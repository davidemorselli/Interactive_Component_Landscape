# The bulk page: one background task summarises, grades and refines a whole
# list of words, then writes a summary of the summaries.
#
# The pipeline is parameterized by its storage table and its pair of levels
# (llm.pipeline.levels), so the frames level (webapp.frames_page) reuses
# it whole with its own of each — only the prompts differ, and those are the
# level. The page leads with the choice between the two (PROMPT_SETS): one
# page, one set of controls, and the prompts a task is written with picked like
# any other parameter of it.

import json

from flask import jsonify, render_template, request

from llm.pipeline.levels import LEVELS
from rag.tweet_retrieval import get_tweets_about

from .app import app
from .params import (DEFAULT_PROMPT_SET, RUN_WORD_PARAMS, VIEW_FLOORS, num,
                     page_context, phase_filter, prompt_sets)
from .runs import (delete_run, named_runs, rename_run, row_exists, row_update,
                   run_guarded, fan_out, start_run)
from .state import search_lock
from .versions import (build_agents, refine_versions, selected_index,
                       selected_version, task_speakers)


# The two sets of prompts a task can be written and graded with: what to call
# the run, where its tasks live (the frames tasks keep their own table,
# endpoints and agents — frames_page — so switching sets costs nothing but a
# different URL), and the budgets the refinement tooltip names. The heading
# names the page; the tab bar under it says which set is chosen.
PAGE_TITLE = "Bulk"

PROMPT_SETS = prompt_sets(
    **{key: {"base": base,
             "merge_label": levels.merge.label,
             "word_budget": levels.word.words,
             "merge_budget": levels.merge.words}
       for key, base in (("narratives", "/tasks"), ("frames", "/frame_tasks"))
       for levels in [LEVELS[key]]})


@app.get("/bulk")
def bulk():
    # floors twice: the template writes them into the header tooltips, and the
    # page's script colours the poll-drawn grade cells against PAGE.floors.
    # The prompt sets likewise: the template renders the tabs, and the script
    # follows the chosen one to its endpoints and its budgets.
    return render_template("bulk.html", floors=VIEW_FLOORS,
                           page_title=PAGE_TITLE, prompt_sets=PROMPT_SETS,
                           default_prompt_set=DEFAULT_PROMPT_SET,
                           **page_context(RUN_WORD_PARAMS, floors=VIEW_FLOORS,
                                          prompt_sets=PROMPT_SETS))


def run_word(item, p, writer, graders, speakers_by_phase):
    """Summarize one (phase, word), refining while the settings ask for it."""
    row = {"phase": None, "word": None, "versions": []}
    try:
        phase, word = str(item["phase"]), item["word"]
        row["phase"], row["word"] = phase, word
        with search_lock:
            hits = get_tweets_about(word, take_n=num(p.get("n_tweets"), 20, int),
                                    phase=phase_filter(phase),
                                    speakers=None if speakers_by_phase is None
                                    else speakers_by_phase[phase],
                                    min_similarity=num(p.get("min_similarity"), 0.3))
        tweets = hits["tweet"].tolist()
        if not tweets:
            row["note"] = "no tweets retrieved"
            return row
        first = writer.write(word, tweets)
        # An answer with nothing in it is neither a narrative nor the refusal
        # the prompt allows: grading it would store a version whose every field
        # is empty and count the word as summarized.
        if not (first or "").strip():
            row["note"] = "the summary agent answered nothing"
            return row
        refine_versions(row["versions"], word, first,
                        tweets, graders, p.get("refine") or {})
    except Exception as e:
        # The word, so a note read off the table says which one broke, and the
        # class of the exception, since the message alone can be a bare parse
        # position with nothing saying what failed to parse.
        row["note"] = f"error on '{word}': {type(e).__name__}: {e}"
    return row


def run_merge(p, finals, level):
    """The task's merge: one answer over the final answers of its words,
    graded and refined under the same settings as any one word's, with those
    answers standing where the tweets do."""
    writer, graders = build_agents(p, level)
    versions = []
    # No query: the summaries stand for the task's whole word list.
    refine_versions(versions, "", writer.write("", finals), finals,
                    graders, p.get("refine") or {})
    # The selected version is what the task stands on; the rest of the run
    # rides under `attempts`, oldest first, with `selected` saying where the
    # version above was taken from — without it the run cannot be put back
    # together, the selection no longer always being last.
    chosen = selected_index(versions)
    return versions[chosen] | {"attempts": versions[:chosen] + versions[chosen + 1:],
                               "selected": chosen}


# Below this many narratives, a summary of summaries would be a statement about
# whatever few words happened to land rather than about the component.
MERGE_MIN_NARRATIVES = 3


def coverage(results):
    """What a task's words actually yielded — narratives, refusals, failures.
    It is the narrative count, not the progress bar's done/total, that says
    how much of the component the task has anything to say about."""
    counts = {"narratives": 0, "refusals": 0, "failed": 0}
    for r in results:
        if not r or not r.get("versions"):
            counts["failed"] += 1
        elif selected_version(r["versions"]).get("abstained"):
            counts["refusals"] += 1
        else:
            counts["narratives"] += 1
    return counts


def finals_of(results):
    """
    The narratives a task selected among its results — what its summary of
    summaries is written from. Only the narratives: summarising a list of
    refusals yields a narrative about the corpus at large that the graders
    cannot catch, since they read the same list. The selected version rather
    than the last written (selected_index), so a run that ended on a worse
    draft does not carry it into the merge.
    """
    return [selected_version(r["versions"])["summary"] for r in results
            if r and r["versions"]
            and not selected_version(r["versions"]).get("abstained")]


def merge_for(p, finals, counts, level):
    """The task's summary of summaries — or, in its place, the reason there
    is none, in the {"error": ...} shape the pages render. The coverage
    travels with it, sparing the task list re-reading every result per poll."""
    if len(finals) < MERGE_MIN_NARRATIVES:
        merged = {"error": f"no summary of summaries: only {counts['narratives']} of "
                           f"{sum(counts.values())} words yielded a narrative "
                           f"({counts['refusals']} refusals, {counts['failed']} failed)."}
    else:
        try:
            merged = run_merge(p, finals, level)
        except Exception as e:
            merged = {"error": f"summary of summaries failed: {e}"}
    return merged | {"coverage": counts}


def run_task(task_id, p, table="tasks", levels=LEVELS["narratives"], finish=None):
    """One task's run, from the word list to the summary of summaries.
    `finish` is a last pass over the finished merge, given (params, finals,
    merged) and answering the merge to store — the frames page groups its
    analysis under topics there (topic_merge); the narrative level passes
    none."""
    def body():
        writer, graders = build_agents(p, levels.word)
        speakers_by_phase = task_speakers(p)

        def progress(results, done):
            # The counts ride along with the tick, so the task list fills its
            # coverage columns as the run goes rather than all at once at the
            # end. A word not back yet is a None here, and a None is not a
            # failure: only what has finished is counted.
            finished = [r for r in results if r is not None]
            row_update(table, task_id, done=done, results=json.dumps(finished),
                       meta=json.dumps({"coverage": coverage(finished)}))

        # The words go through the pool; the results come back in word order.
        results = fan_out(table, task_id,
                          lambda item: run_word(item, p, writer, graders,
                                                speakers_by_phase),
                          p.get("words") or [], progress)
        finals = finals_of(results)
        if row_exists(table, task_id):
            merged = merge_for(p, finals, coverage(results), levels.merge)
            # The last pass reads the summary the merge just wrote and answers
            # what to store, so the task writes its `meta` column once, already
            # holding whatever that pass changed. It is run before the row is
            # touched rather than after: a merge stored and then rewritten
            # would be on screen for the minute the pass takes, as a summary
            # about to change under whoever is reading it.
            if finish is not None:
                merged = finish(p, finals, merged)
            row_update(table, task_id, meta=json.dumps(merged))
        row_update(table, task_id, status="done")

    run_guarded(table, task_id, body)


# The most words one run will take. Each costs a summary and its grades, so a
# list harvested with a wide Top n words is a bill rather than a run.
MAX_WORDS = 300


def start_task(payload, table="tasks", runner=run_task):
    """Validate one create-task payload and set its runner going — shared with
    the frames page, which passes its own table and runner."""
    words = payload.get("words") or []
    if not words:
        return jsonify({"error": "No words — get or add some first."}), 400
    if len(words) > MAX_WORDS:
        return jsonify({"error": f"{len(words)} words is more than one run will take "
                                 f"(limit {MAX_WORDS}) — narrow the search first."}), 400
    return start_run(table, runner, payload, len(words))


@app.post("/tasks")
def create_task():
    return start_task(request.get_json(silent=True) or {})


@app.get("/tasks")
def list_tasks():
    return named_runs("tasks")


@app.patch("/tasks/<int:task_id>")
def rename_task(task_id):
    return rename_run("tasks", task_id)


@app.delete("/tasks/<int:task_id>")
def delete_task(task_id):
    return delete_run("tasks", task_id)
