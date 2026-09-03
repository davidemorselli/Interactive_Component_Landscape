# Tweet narratives, on demand from the landscape page: retrieval, then the
# LLM summary, its grades, and one rewrite against a critique. One query at a
# time, on either set of prompts — the tab bar's choice travels with every
# call and nothing is stored, which is what lets this page be where a prompt
# is tried before a whole word list is spent on it.

import math
import time

from flask import jsonify, request

from llm.pipeline.acueval import build_acueval_feedback
from llm.pipeline.budget import (TOLERANCE, build_budget_feedback,
                                       word_count)
from llm.pipeline.levels import LEVELS
from llm.pipeline.rubric import build_rubric_feedback
from rag.tweet_retrieval import get_tweets_about

from .app import app
from .params import (DEFAULT_PROMPT_SET, GRADERS, num, payload_strs,
                     phase_filter, read, role_config)
from .state import search_lock, speaker_names

def level_for(payload):
    """The level the call asks for, by the prompt set its page had chosen."""
    # Always the word level — this page writes about one query, never a word
    # list. A payload naming no set (or an unknown one) falls back to the
    # default, so an older page still gets an answer rather than a 500.
    return (LEVELS.get(payload.get("prompts")) or LEVELS[DEFAULT_PROMPT_SET]).word


@app.get("/tweets")
def tweets_endpoint():
    args = request.args
    query = (args.get("query") or "").strip()
    if not query:
        return jsonify({"note": "Type a query.", "tweets": [], "html": None})

    phase = args.get("phase", "all")
    source = args.get("speakers", "all")
    component = read(args, "component")
    n_extreme = read(args, "n_extreme")
    min_similarity = num(args.get("min_similarity"), 0.3)

    with search_lock:
        names = speaker_names(source, component, n_extreme, phase)
        if names is not None and not names:
            return jsonify({"note": f"Component {component} has no {source}.",
                            "tweets": [], "html": None})
        hits = get_tweets_about(query, take_n=num(args.get("n_tweets"), 20, int),
                                phase=phase_filter(phase),
                                speakers=names, min_similarity=min_similarity)

    if not len(hits):
        return jsonify({"note": f"No tweet within {min_similarity} similarity of the query "
                                "— lower the floor or rephrase.",
                        "tweets": [], "html": None})

    scope = "" if names is None else f", from {len(names)} {source} of component {component}"
    return jsonify({"note": f"{len(hits)} tweets{scope}:",
                    "tweets": hits["tweet"].tolist(),
                    # Rounded to the two decimals every other similarity shows,
                    # and without pandas' unnamed index column.
                    "html": hits.round(2).to_html(index=False)})


def timed(fn, *args, **kwargs):
    """What fn answered, and the wall-clock seconds it took to answer it."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, round(time.perf_counter() - t0, 2)


@app.post("/narrative")
def narrative_endpoint():
    payload = request.get_json(silent=True) or {}
    query, = payload_strs(payload, "query")
    tweets = payload.get("tweets") or []
    if not query or not tweets:
        return jsonify({"error": "Nothing to summarise — run a search first."}), 400
    writer = level_for(payload).writer(**role_config(payload, "summary"))
    summary, seconds = timed(writer.write, query, tweets)
    return jsonify({"summary": summary, "seconds": seconds})


@app.post("/grade")
def grade_endpoint():
    payload = request.get_json(silent=True) or {}
    query, summary = payload_strs(payload, "query", "summary")
    tweets = payload.get("tweets") or []
    if not query or not summary or not tweets:
        return jsonify({"error": "Nothing to grade — get a narrative first."}), 400
    level = level_for(payload)
    acueval = level.agent("acueval", **role_config(payload, "acueval"))
    rubric = level.agent("rubric", **role_config(payload, "rubric"))
    # Timed one by one: the two graders run in turn, and the page says how long
    # each of them took on its own card.
    (verified, acu_score), acu_seconds = timed(acueval.evaluate_summary, summary, tweets)
    graded, rubric_seconds = timed(rubric.evaluate_summary, query, summary, tweets)

    supported = verified["supported"]
    grades = None
    if graded is not None:  # None: the grader's answer was not valid JSON
        column = graded["grade"]
        grades = {"criteria": {c: int(g) for c, g in
                               column.drop(["average", "reason"]).items()},
                  "average": float(column["average"]),
                  "reason": str(column["reason"]),
                  # What the summary would be rewritten against; empty at
                  # full marks, and the page then offers no refinement.
                  "feedback": build_rubric_feedback(graded)}
    # The word budget is arithmetic — answered here, no model asked, no time
    # to report; the critique is empty unless the summary overran.
    words = level.words
    budget_feedback = build_budget_feedback(summary, words, TOLERANCE,
                                            level.budget_feedback)
    return jsonify({
        "budget": {"words": word_count(summary), "budget": words,
                   "over": bool(budget_feedback), "feedback": budget_feedback},
        "acueval": {"score": None if math.isnan(acu_score) else acu_score,
                    "supported": int(supported.fillna(False).sum()),
                    "units": int(len(supported)),
                    # One claim per unit; None where the grader answered neither.
                    "claims": [{"unit": unit, "supported": verdict}
                               for unit, verdict in supported.items()],
                    "feedback": build_acueval_feedback(verified),
                    "seconds": acu_seconds},
        # Outside the grades: a grader that answered nothing valid still took
        # its time to do it, and the page shows that much.
        "rubric": grades,
        "rubric_seconds": rubric_seconds,
    })


# The grader that wrote a critique is the one that rewrites the summary against
# it — built for the level the call names, so a frame analysis is rewritten
# under the frame prompt rather than told to open the way a narrative does.
# Which model box each rewrites with is role_config's ROLE_BOX, as everywhere.


@app.post("/refine")
def refine_endpoint():
    """Rewrite a summary against one grader's critique. The critique comes
    back from the page as /grade sent it out, so the rewrite answers the
    grades the user is looking at without grading a second time."""
    payload = request.get_json(silent=True) or {}
    query, summary, feedback = payload_strs(payload, "query", "summary", "feedback")
    tweets = payload.get("tweets") or []
    grader = payload.get("grader")
    if grader not in GRADERS:
        return jsonify({"error": f"grader must be one of {list(GRADERS)}"}), 400
    if not query or not summary or not tweets or not feedback:
        return jsonify({"error": "Nothing to refine — grade a narrative first."}), 400
    agent = level_for(payload).agent(grader, **role_config(payload, grader))
    revised, seconds = timed(agent.refine_summary, query, tweets, summary, feedback)

    if revised is None:  # the rewrite dropped the opening every summary must have
        return jsonify({"error": "The rewrite is not a summary — it does not open the "
                                 "way one must. The summary is left as it was."}), 502
    return jsonify({"summary": revised, "seconds": seconds})
