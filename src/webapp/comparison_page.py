# LLM comparison: several summary agents write a summary of every word, and
# several graders grade every summary. The run reports each agent's average
# grade (the comparison) and each grader's (how lenient it was), through
# fan_out in two stages. Parameterized by storage table and level, so the
# frames comparison (webapp.frame_comparison_page) reuses it whole.

import json
import math

import pandas as pd
from flask import jsonify, render_template, request

from llm.pipeline.agent import Attempts
from llm.pipeline.levels import LEVELS
from llm.prompts.refusals import declines
from rag.tweet_retrieval import get_tweets_about

from .app import app
from .params import (DECIMALS, DEFAULT_PROMPT_SET, RUN_WORD_PARAMS,
                     VIEW_FLOORS, csv_response, display_name, grade_cls,
                     no_such, num, page_context, phase_filter, prompt_sets,
                     stored_retrieval)
from .runs import (delete_run, list_runs, row_update, run_guarded, db,
                   fan_out, start_run)
from .state import search_lock
from .versions import task_speakers

# The level one comparison writes and grades at: the word level of the
# narratives set, the frames comparison passing its own (frame_comparison_page).
# A comparison runs three of a level's four roles — it writes once and never
# rewrites, so it has nothing to measure a length against.
LEVEL = LEVELS["narratives"].word

# The two grader kinds a comparison offers, and the role of the level each is
# built from. The kinds are the stored parameter keys the page sends its
# grader lists under (`acue_graders`, `rubric_graders`) — never rename them.
GRADER_ROLES = {"acue": "acueval", "rubric": "rubric"}


def comparison_graders(p):
    """The judges of one comparison, each named after the CSV column it fills.

    The labels are stored as JSON keys in the comparisons table's results —
    never change them.
    """
    return ([dict(g, kind="acue", label=f"ACUEval_grader_{i}")
             for i, g in enumerate(p.get("acue_graders") or [], 1)]
            + [dict(g, kind="rubric", label=f"RUBRIC_grader_{i}")
               for i, g in enumerate(p.get("rubric_graders") or [], 1)])


def comparison_size(p):
    """
    How many model calls one comparison asks for: a summary per (word, agent),
    and a grade of each of those per grader. It is what the progress counts
    against, and an upper bound — a word nothing was retrieved for costs none.
    """
    return (len(p.get("words") or []) * len(p.get("agents") or [])
            * (1 + len(comparison_graders(p))))


def grade_summary(grader, word, summary, tweets, level=LEVEL, attempts=None):
    """
    The grade one grader gives one summary, on its own scale: 0-1 for an
    'acue' kind, 1-5 for a 'rubric' one; None when the grader answered nothing
    valid. A refusal is graded on the `warranted` scale instead — which scale
    a row is on is read off its `declined` flag, and the two are never
    averaged together (comparison_summary).
    """
    agent = level.agent(GRADER_ROLES[grader["kind"]], model=grader["model"],
                        attempts=attempts)
    if grader["kind"] == "acue":
        _, score = agent.evaluate_summary(summary, tweets)
        return None if math.isnan(score) else score
    graded = agent.evaluate_summary(word, summary, tweets)
    return None if graded is None else float(graded["grade"]["average"])


def run_comparison(run_id, p, table="comparisons", level=LEVEL):
    """Every agent summarises every word, and every grader grades every summary."""
    def body():
        agents = p.get("agents") or []
        graders = comparison_graders(p)
        # Every agent this run builds — writers and graders alike — counts its
        # calls here, so the report is over the run and not over the process.
        attempts = Attempts()

        def progress(offset):
            """The run's progress, counting from what the earlier stages did."""
            return lambda _, done: row_update(table, run_id, done=offset + done)

        # The same speaker restriction a bulk task reads off the same settings:
        # tweets from the strong or extreme speakers of the component, or from
        # every speaker. Resolved before the scans — task_speakers takes the
        # search lock itself.
        speakers_by_phase = task_speakers(p)

        # The corpus scans are serialized by the search lock anyway, so they run
        # one after the other, before the model calls fan out. A key is a
        # (phase, word) pair, as the page collected it.
        tweets = {}
        for item in p.get("words") or []:
            phase = str(item["phase"])
            with search_lock:
                hits = get_tweets_about(
                    item["word"], take_n=num(p.get("n_tweets"), 20, int),
                    min_similarity=num(p.get("min_similarity"), 0.3),
                    phase=phase_filter(phase),
                    speakers=None if speakers_by_phase is None
                    else speakers_by_phase[phase])
            tweets[(phase, item["word"])] = hits["tweet"].tolist()

        # One call that fails must not cost the run every call that succeeded:
        # whatever went wrong, the cell just stays empty.
        def summarize(job):
            key, agent = job
            if not tweets[key]:
                return None
            try:
                writer = level.writer(model=agent["model"], attempts=attempts)
                return writer.write(key[1], tweets[key])
            except Exception:
                return None

        summary_jobs = [(key, agent) for key in tweets for agent in agents]
        written = fan_out(table, run_id, summarize, summary_jobs, progress(0))

        def grade(job):
            i, grader = job
            key, _ = summary_jobs[i]
            try:
                return grade_summary(grader, key[1], written[i], tweets[key],
                                     level, attempts)
            except Exception:
                return None

        grade_jobs = [(i, grader) for i, summary in enumerate(written) if summary
                      for grader in graders]
        # The bar was set on every summary being written and graded; the words
        # nothing was retrieved for, and the calls that failed, take that back.
        row_update(table, run_id, total=len(summary_jobs) + len(grade_jobs))
        graded = fan_out(table, run_id, grade, grade_jobs,
                         progress(len(summary_jobs)))
        got = {}
        for (i, grader), value in zip(grade_jobs, graded):
            got.setdefault(i, {})[grader["label"]] = value

        def average(values):
            values = [v for v in values if v is not None]
            return sum(values) / len(values) if values else None

        # One row per (word, agent): the summary, whether it declined, one
        # column per grader, and the average of each grade scale.
        rows = []
        for i, ((key, agent), summary) in enumerate(zip(summary_jobs, written)):
            grades = {g["label"]: got.get(i, {}).get(g["label"]) for g in graders}
            # An agent is named in the results by the model it runs, as typed.
            rows.append({"phase": key[0], "word": key[1],
                         "agent": agent["model"], "summary": summary,
                         # Which scale this row's RUBRIC grades are on —
                         # stored, so the tables and the CSV read one answer.
                         "declined": bool(summary) and declines(summary),
                         **grades,
                         "average_ACUEval": average(
                             grades[g["label"]] for g in graders if g["kind"] == "acue"),
                         "average_RUBRIC": average(
                             grades[g["label"]] for g in graders if g["kind"] == "rubric")})
        row_update(table, run_id, status="done", results=json.dumps(rows),
                   meta=json.dumps({"calls": attempts.report()}))

    run_guarded(table, run_id, body)


# The two sets of prompts a comparison can be run with. A comparison writes
# once and merges nothing, so unlike the bulk page's PROMPT_SETS there are no
# budgets or merge labels to carry — just where each set's runs live. The
# heading names the page; the tab bar under it says which set is chosen.
PAGE_TITLE = "LLM comparison"

PROMPT_SETS = prompt_sets(narratives={"base": "/comparisons"},
                          frames={"base": "/frame_comparisons"})


@app.get("/comparison")
def comparison_page():
    return render_template("comparison.html",
                           page_title=PAGE_TITLE, prompt_sets=PROMPT_SETS,
                           default_prompt_set=DEFAULT_PROMPT_SET,
                           **page_context(RUN_WORD_PARAMS,
                                          prompt_sets=PROMPT_SETS))


def start_comparison(payload, table="comparisons", runner=run_comparison):
    """Validate one launch payload and set its runner going — shared with the
    frames comparison, which passes its own table and runner."""
    if not payload.get("words"):
        return jsonify({"error": "No words — get or add some first."}), 400
    if not payload.get("agents"):
        return jsonify({"error": "No summary agent — add at least one."}), 400
    if not comparison_graders(payload):
        return jsonify({"error": "No grader — add at least one."}), 400
    return start_run(table, runner, payload, comparison_size(payload))


@app.post("/comparisons")
def create_comparison():
    return start_comparison(request.get_json(silent=True) or {})


@app.get("/comparisons")
def list_comparisons():
    return list_runs("comparisons")


@app.delete("/comparisons/<int:run_id>")
def delete_comparison(run_id):
    return delete_run("comparisons", run_id)


def comparison_results(run_id, table="comparisons"):
    """
    What one comparison found — one row per (word, agent) — and the params it
    was launched with. (None, None) for no such run.
    """
    with db() as conn:
        row = conn.execute(f"SELECT results, params FROM {table} WHERE id = ?",
                           (run_id,)).fetchone()
    if row is None:
        return None, None
    return pd.DataFrame(json.loads(row["results"])), json.loads(row["params"])


def comparison_summary(results, graders):
    """
    The three average tables of one comparison: one per summary agent (the
    comparison itself) and one per grader of each kind, best grade first;
    None when nothing was graded. The averages cover the summaries actually
    written: a refusal's `warranted` grade runs 1-5 like the rubric and means
    something else entirely, so averaging the two would rank first the agent
    that answers least. Refusals are counted in a column of their own instead.
    """
    answered = results[~results["declined"].astype(bool)] if not results.empty else results
    grades = answered.filter(like="_grader_").astype(float)
    if grades.empty or grades.isna().all().all():
        return None
    averages = answered.filter(like="average_").astype(float)
    models = {g["label"]: g["model"] for g in graders}

    def grader_average(kind):
        table = (grades.filter(like=kind).mean().rename_axis("grader")
                 .to_frame(f"average {kind} grade").round(2))
        table.insert(0, "model", [models.get(label, "") for label in table.index])
        return table.sort_values(f"average {kind} grade", ascending=False)

    # The agents rank on ACUEval first, RUBRIC breaking its ties; a run graded
    # on one scale only has nothing but NaN in the other, which sorts last.
    per_agent = averages.groupby(answered["agent"]).mean().round(2)
    ranked = per_agent.sort_values(list(per_agent.columns), ascending=False)
    # Every agent the run had, including one that answered nothing at all and
    # so has no row above: it is the agent the count exists to show.
    refusals = results.groupby("agent")["declined"].sum().astype(int)
    silent = [agent for agent in refusals.index if agent not in ranked.index]
    ranked = ranked.reindex([*ranked.index, *silent])
    # Kept as objects, not as an int column: the page reads the table row by
    # row (colored_html), and a row of mixed dtypes comes back as one float
    # Series — which would print a count of refusals as "1.0".
    ranked["refusals"] = (refusals.reindex(ranked.index).fillna(0)
                          .astype(int).astype(object))
    return {"per_agent": ranked,
            "per_acue": grader_average("ACUEval"),
            "per_rubric": grader_average("RUBRIC")}


def call_stats(run_id, table="comparisons"):
    """
    What one comparison's calls cost, one row per model: calls, retries per
    call, failures. A failure is a call that ran out of tries or hit its token
    ceiling; the run drops what it was for and carries on, so failures are
    invisible in the grades and counted here instead. None for a run with no
    counted meta to read.
    """
    with db() as conn:
        row = conn.execute(f"SELECT meta FROM {table} WHERE id = ?",
                           (run_id,)).fetchone()
    if row is None or not row["meta"]:
        return None
    counted = (json.loads(row["meta"]) or {}).get("calls") or {}
    if not counted:
        return None
    stats = pd.DataFrame([
        {"model": model,
         "calls": counted[model]["calls"],
         "retries per call": round(
             (counted[model]["tries"] - counted[model]["calls"])
             / counted[model]["calls"], DECIMALS),
         "failures": counted[model]["failures"],
         "failure %": round(100 * counted[model]["failures"]
                            / counted[model]["calls"], DECIMALS)}
        for model in sorted(counted)])
    # Worst first: the reason to read this table is to find the model that is
    # costing the run, and counts stay counts (see comparison_summary).
    stats = stats.sort_values(["failure %", "retries per call"], ascending=False)
    stats = stats.set_index("model")
    for column in ("calls", "failures"):
        stats[column] = stats[column].astype(int).astype(object)
    return stats


def averages_of(run_id, table="comparisons"):
    """The summary tables the page shows, as the HTML it lays out."""
    results, params = comparison_results(run_id, table)
    if results is None:
        return no_such("comparison")
    tables = comparison_summary(results, comparison_graders(params))
    calls = call_stats(run_id, table)
    if tables is None:
        # The call report still stands, and is most worth reading precisely
        # when nothing was graded: it is what says why.
        answer = {"note": "Nothing was graded."}
        if calls is not None:
            answer["per_call"] = colored_html(calls)
        return jsonify(answer)
    shown = {key: colored_html(value) for key, value in tables.items()}
    if calls is not None:
        shown["per_call"] = colored_html(calls)
    return jsonify(shown)


@app.get("/comparisons/<int:run_id>")
def comparison_averages(run_id):
    return averages_of(run_id)


def summary_csv_of(run_id, table, filename):
    """The same summary tables as one CSV, a titled block each."""
    results, params = comparison_results(run_id, table)
    if results is None:
        return no_such("comparison")
    tables = comparison_summary(results, comparison_graders(params)) or {}
    calls = call_stats(run_id, table)
    if calls is not None:
        tables = {**tables, "per_call": calls}
    titles = {"per_agent": "Average per summary agent",
              "per_acue": "Average per ACUEval grader",
              "per_rubric": "Average per RUBRIC grader",
              "per_call": "Calls, retries and failures per model"}
    body = "\n".join(titles[key] + "\n" + table.to_csv()
                     for key, table in tables.items())
    return csv_response(body or "Nothing was graded.\n", filename)


@app.get("/comparisons/<int:run_id>/summary.csv")
def comparison_summary_csv(run_id):
    return summary_csv_of(run_id, "comparisons", f"comparison_{run_id}_summary.csv")


def grade_class(column, value):
    """"ok"/"bad" for a grade cell against its scale's floor, "" for the rest."""
    for family in VIEW_FLOORS:
        if family in column.lower():
            return grade_cls(family, value)
    return ""


def pretty(label):
    """A stored key as the pages show it — the grade families in their proper
    spelling, spaces for the underscores. Display only: the stored keys, which
    the CSVs keep, never change."""
    return display_name(str(label)).replace("_", " ")


def colored_html(table):
    """The table as one HTML string, each grade cell coloured as in the
    detailed view — rendered by the _grade_table partial, so every value is
    escaped on its way in."""
    def cell(column, value):
        value = "" if pd.isna(value) else value
        return value, grade_class(column, value)

    # The grader tables are indexed by the stored grader labels; the agent one
    # by the model identifiers as typed, which stay as they are.
    fix_index = table.index.name == "grader"
    rows = [(pretty(idx) if fix_index else idx,
             [cell(c, v) for c, v in row.items()])
            for idx, row in table.iterrows()]
    return render_template("_grade_table.html", index_name=table.index.name or "",
                           columns=[pretty(c) for c in table.columns],
                           rows=rows).rstrip("\n")


# The results read down a word instead of across a
# table. One card per word holds what every agent answered for it, side by
# side, which is the shape a summary is actually judged in — two answers to the
# same tweets, read against each other.


def text_of(row, column):
    """
    One text field of a result row, as a string. A summary the run never got —
    a failed call — reads back from the stored JSON as a NaN, which is neither
    falsy nor a string, so it is turned into the empty one here.
    """
    value = row.get(column)
    return "" if value is None or pd.isna(value) else str(value)


def comparison_answers(graders, word_rows):
    """
    What each agent answered for one word: the summary, the working note behind
    it, and its grades. In the order the run wrote them, so an agent keeps the
    same column in every card.
    """
    answers = []
    for _, row in word_rows.iterrows():
        summary = text_of(row, "summary")
        grades = []
        for grader in graders:
            value = row.get(grader["label"])
            text = "" if value is None or pd.isna(value) else round(float(value), DECIMALS)
            grades.append({"label": grader["label"].replace("_", " "),
                           "value": text,
                           "cls": grade_class(grader["label"], text)})
        answers.append({"agent": row["agent"],
                        "summary": summary,
                        # A refusal is an absence, not a bad answer: the page
                        # greys it as the rest of the app does — the summary
                        # and, with it, grades that are on the `warranted`
                        # scale rather than the rubric's.
                        "declined": bool(row.get("declined")),
                        "grades": grades})
    return answers


def side_by_side_of(run_id, table="comparisons", runs_base="/comparisons",
                    **page_vars):
    """One card per word, holding every agent's answer to it."""
    results, params = comparison_results(run_id, table)
    if results is None:
        return no_such("comparison")
    graders = comparison_graders(params)
    # A run still going has written no rows yet, and so has no columns to group
    # by: the page says as much rather than the grouping raising.
    cards = [{"phase": phase, "word": word,
              "answers": comparison_answers(graders, rows)}
             for (phase, word), rows in results.groupby(["phase", "word"], sort=False)
             ] if not results.empty else []
    return render_template(
        "comparison_side_by_side.html", run_id=run_id, cards=cards,
        runs_base=runs_base, retrieval=stored_retrieval(params), **page_vars)


@app.get("/comparisons/<int:run_id>/side-by-side")
def comparison_side_by_side(run_id):
    return side_by_side_of(run_id)


def detail_csv_of(run_id, table, filename):
    results, _ = comparison_results(run_id, table)
    if results is None:
        return no_such("comparison")
    return csv_response(results.round(DECIMALS).to_csv(index=False), filename)


@app.get("/comparisons/<int:run_id>/csv")
def comparison_csv(run_id):
    return detail_csv_of(run_id, "comparisons", f"comparison_{run_id}.csv")
