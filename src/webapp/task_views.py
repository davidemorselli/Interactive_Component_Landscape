# One task's results, as the browser table and as the CSV — parameterized by
# the table the task lives in, so the frames page (webapp.frames_page) reuses
# both views whole for its own runs.

import json

import pandas as pd
from flask import render_template

from llm.prompts.refusals import declines

from .app import app
from .params import (DECIMALS, col_class, csv_response, display_name,
                     grade_cls, no_such, stored_retrieval)
from .runs import db
from .versions import FIELDS, GRADE_FIELDS, rubric_grade_of, selected_index


def versions_of(r):
    """One word's summary versions — or, for a word that produced nothing, a
    single pseudo-version carrying the reason in place of the summary."""
    return r["versions"] or [{"summary": f"({r.get('note', 'nothing')})"}]


def merge_versions(merged):
    """The merge's run, in the order it was written: the drafts stored under
    `attempts` with the version the task stands on put back where it was
    taken from."""
    attempts, k = merged["attempts"], merged["selected"]
    return attempts[:k] + [merged] + attempts[k:]


def run_depth(results, merged=None):
    """
    How many versions of a summary the table shows beside the one it selected:
    as many as its most rewritten summary went through — the run entire, since
    the selection is not always the last version written (selected_index).
    The merge counts when passed (it is a row of the same table); the CSV
    passes none.
    """
    depths = [len(r["versions"]) for r in results]
    # A merge is a row only once its summary is stored: mid-run the meta
    # carries the coverage alone, and there is no run of drafts to measure.
    if merged and merged.get("summary") and not merged.get("error"):
        depths.append(len(merge_versions(merged)))
    depth = max([*depths, 0])
    # One version is not a run: a task nothing was rewritten in has the version
    # it selected and nothing to set beside it. Floored the same way for a task
    # whose every word came back with no summary at all.
    return depth if depth > 1 else 0


def csv_value(version, field):
    """One field of one version as the CSV shows it: grades are rounded."""
    value = version.get(field)
    if field in GRADE_FIELDS and value is not None:
        return round(float(value), DECIMALS)
    return value


def task_table(task_id, table="tasks"):
    """The (rows, columns) of the task's CSV table, None for no such task.

    Only the CSV keeps the flat `field_k` column shape (and the warranted_*
    columns); the browser view is built from the version dicts directly. The
    `final_` columns hold the selected version, `selected` says which attempt
    that was, and the numbered columns are the run itself, attempt 1 first —
    a column means the same attempt on every row.
    """
    with db() as conn:
        row = conn.execute(f"SELECT results, params FROM {table} WHERE id = ?",
                           (task_id,)).fetchone()
    if row is None:
        return None
    results = json.loads(row["results"])
    depth = run_depth(results)
    rows = []
    for r in results:
        versions = versions_of(r)
        chosen = selected_index(versions)
        out = {"phase": r["phase"], "word": r["word"],
               "final_summary": versions[chosen].get("summary")}
        out.update({f: csv_value(versions[chosen], f) for f in FIELDS[1:]})
        out["selected"] = f"{chosen + 1} of {len(versions)}"
        for k in range(1, depth + 1):  # 1 is the first version written
            attempt = versions[k - 1] if len(versions) >= k else {}
            out.update({f"{f}_{k}": csv_value(attempt, f) for f in FIELDS})
        rows.append(out)
    columns = (["phase", "word", "final_summary"] + FIELDS[1:] + ["selected"]
               + [f"{f}_{k}" for k in range(1, depth + 1) for f in FIELDS])
    return rows, columns


def csv_of(task_id, table, filename):
    """One task's CSV download, from whichever table it lives in."""
    data = task_table(task_id, table)
    if data is None:
        return no_such("task")
    rows, columns = data
    return csv_response(pd.DataFrame(rows, columns=columns)
                        .rename(columns=display_name).to_csv(index=False),
                        filename)


@app.get("/tasks/<int:task_id>/csv")
def task_csv(task_id):
    return csv_of(task_id, "tasks", f"task_{task_id}.csv")


# The view's columns: phase and word name the row, `refined` (the one thing
# that belongs to the whole run, so it leads rather than standing in a block)
# says which version the row stands on and what sent the others back, then one
# block of these six per version. The browser folds a refusal's grade into the
# RUBRIC pair tagged "(warranted)"; `words` is a length, not a grade, and
# wears no colour.
LEAD_COLUMNS = ["phase", "word", "refined"]
VERSION_COLUMNS = ["summary", "RUBRIC", "RUBRIC comment", "ACUEval",
                   "ACUEval comment", "words"]


def refined_label(key, version):
    """One step of the trail: what asked for the rewrite, and the reading that
    fell short. The budget is not a grader and its rewrite is not about quality
    — it is named for what it measured and what it wanted less of."""
    if key == "rubric":
        grade = rubric_grade_of(version)
        return None if grade is None else f"RUBRIC {round(grade, DECIMALS)}"
    if key == "acueval":
        grade = version.get("acue_grade")
        return None if grade is None else f"ACUEval {round(grade, DECIMALS)}"
    if key == "budget":
        words = version.get("budget_words")
        return "length" if words is None else f"length {words}w"
    return None


def refinement(versions):
    """
    Which attempt the task stands on, and what sent each of the others back —
    "kept #1 of 3 · RUBRIC 3.1 → length 312w" for a summary the rubric put back
    at 3.1 and the word budget put back again at 312 words, whose first draft
    outgraded both rewrites in the end.

    The trail reads left to right in the order the rewrites happened, which is
    the order the attempt columns stand in, so a step and the version it sent
    back are read under one another. "kept #3 of 3" is the run that improved
    all the way through; anything else is a run that did not. Each rewrite
    stores what asked for it (`refined_for`), so nothing here is guesswork.
    """
    trail = []
    for i, version in enumerate(versions[:-1]):
        trail.append(refined_label(versions[i + 1].get("refined_for"), version))
    named = [step for step in trail if step]
    kept = f"kept #{selected_index(versions) + 1} of {len(versions)}"
    # "0" where nothing was written twice: there is no run to say which of, and
    # no trail. A summary written once can still have something said about it,
    # so the notes below stand under either line.
    line = (kept + (" · " + " → ".join(named) if named else "")
            if trail else "0")
    # What a version says about itself — the frame topic pass's account of what it
    # arranged — under the trail rather than in a column of its own: this is
    # the cell that says how the summary came to look the way it does, and a
    # column would cost every row in the table a blank one. Each note is
    # numbered by the version that carries it.
    notes = [f"#{i + 1}: {version['note'].strip()}"
             for i, version in enumerate(versions) if version.get("note")]
    return "\n\n".join([line, *notes]) if notes else line


def version_cells(version):
    """
    The six (text, colour) view cells of one summary version — an absent
    version (a word with fewer attempts than the table is deep) is `{}` and
    yields six empty cells. Both grading scales share the one RUBRIC column
    pair, a refusal's grade tagged "(warranted)".
    """
    def cell(field, cls):
        value = csv_value(version, field)
        return ("" if value is None else value), cls

    summary = version.get("summary")
    # A summary cell holding a refusal rather than a narrative is greyed out:
    # an absence is not a judgment, so it never wears the grades' green/red.
    declined = isinstance(summary, str) and declines(summary)
    cells = [("" if summary is None else summary, "none" if declined else "")]

    warranted = csv_value(version, "warranted_grade")
    if warranted is not None:
        cls = grade_cls("warranted", warranted)
        cells.append((f"{warranted} (warranted)", cls))
        cells.append((version.get("warranted_comment") or "", cls))
    else:
        cls = grade_cls("rubric", csv_value(version, "rubric_grade"))
        cells += [cell("rubric_grade", cls), cell("rubric_comment", cls)]
    cls = grade_cls("acue", csv_value(version, "acue_grade"))
    cells += [cell("acue_grade", cls), cell("acue_comment", cls)]
    # How long the summary is — measured, not judged, so it has no floor to
    # clear and stays uncoloured. A version stored before it was measured has
    # none, and shows an empty cell like any absent value.
    cells.append(cell("budget_words", ""))
    return cells


def version_row(lead, versions, depth, refined):
    """
    One row of the view table: the row-naming cells and the refinement trail,
    the selected version (`sel`), then — down to `depth` — the run it was
    selected from, oldest first (`prior`). The two hold the same versions, so
    the detail switch trades one for the other; the attempt the selection came
    from is marked `chosen` where it stands rather than lifted out, since the
    point of showing a run is its order. A table with no run at all carries no
    `sel` — its one block stands under both settings. The cell opening each
    block also carries group-start, so the boundary runs the whole table.
    """
    cells = list(lead) + [(refined, col_class("refined"))]
    chosen = selected_index(versions)
    for k in range(depth + 1):  # 0 is the selected version, 1 the first written
        if not k:
            version = versions[chosen]
        else:
            version = versions[k - 1] if len(versions) >= k else {}
        # The column each cell stands in, and which half of the trade it is on:
        # the selected version, or an attempt of the run — and, of those, the
        # one the selection was taken from.
        mark = (" sel" if depth else "") if not k else (
            " prior" + (" chosen" if k - 1 == chosen else ""))
        cells += [(text, f"{cls} {col_class(column)}{mark}".strip())
                  for (text, cls), column
                  in zip(version_cells(version), VERSION_COLUMNS)]
    if depth:
        i = len(LEAD_COLUMNS)
        for _ in range(depth + 1):
            text, cls = cells[i]
            cells[i] = (text, f"{cls} group-start".strip())
            i += len(VERSION_COLUMNS)
    return cells


def view_rows(results, depth):
    """The view table's body: one row per word."""
    return [version_row([(r["phase"], ""), (r["word"], "")], versions, depth,
                        refinement(versions))
            for r in results for versions in [versions_of(r)]]


# What the word column says on the summary-of-summaries row, which names no
# word. The view's script reads the `meta` class, not this, to tell the row
# apart.
MERGE_LABEL = "summary of summaries"


def merge_row(merged, depth):
    """
    The merge as the table's first row: the same six cells as a word's, so
    the summary written from the words is read against the words themselves.
    Every cell carries `meta`, which sets the row apart and keeps the tweets
    button off it (it was written from summaries, not tweets). None when the
    task has no summary of summaries to put there.
    """
    if not merged or merged.get("error") or not merged.get("summary"):
        return None
    versions = merge_versions(merged)
    cells = version_row([("", ""), (MERGE_LABEL, "meta-label")], versions,
                        depth, refinement(versions))
    return [(text, f"{cls} meta".strip()) for text, cls in cells]


def stored_merge(stored):
    """
    The task's stored merge — its summary of summaries, or its merged frame
    analysis — or None while it has none. The summary itself and its grades go
    into the table (merge_row); what is left for the box above it is the
    coverage the summary was written from, and — when there is no summary —
    why there is none.
    """
    return json.loads(stored) if stored else None


def view_of(task_id, table="tasks", **page_vars):
    """One task's browser view, from whichever table it lives in. `page_vars`
    reach the template: the frames page names its heading there, the bulk page
    leaves it to the template's default."""
    with db() as conn:
        row = conn.execute(f"SELECT results, params, meta, error FROM {table} "
                           "WHERE id = ?", (task_id,)).fetchone()
    if row is None:
        return no_such("task")
    params = json.loads(row["params"])
    results = json.loads(row["results"])
    merged = stored_merge(row["meta"])
    depth = run_depth(results, merged)
    # The two halves of the detail switch's trade (version_row): the selected
    # version, and the run it came out of, attempts numbered forwards from the
    # first version written.
    groups = [{"label": "Selected", "cls": "final sel", "columns": VERSION_COLUMNS}]
    groups += [{"label": f"Attempt {k}", "cls": "prior", "columns": VERSION_COLUMNS}
               for k in range(1, depth + 1)]
    # The single-row head of an ungrouped table, in the CSV's spelling. Its one
    # version is neither selected from a run nor an attempt of one, so it is
    # headed as the summary the task came to and nothing is traded away.
    flat = ["final_summary"] + [display_name(f) for f in FIELDS[1:]
                                if not f.startswith("warranted")]
    return render_template(
        "view.html", task_id=task_id, columns=LEAD_COLUMNS + flat,
        lead=LEAD_COLUMNS if depth else None, groups=groups if depth else None,
        meta=merged,
        # Why the task stopped early, when it did — the page says so instead
        # of leaving an interrupted task looking merely unfinished.
        run_error=row["error"],
        retrieval=stored_retrieval(params),
        # What the task was called on the list it was started from; a task
        # never named has none and the heading is what it always was.
        task_name=(params.get("name") or "").strip() or None,
        # One (text, colour) pair per cell: the page only lays them out.
        rows=([r for r in [merge_row(merged, depth)] if r]
              + view_rows(results, depth)), **page_vars)


@app.get("/tasks/<int:task_id>/view")
def task_view(task_id):
    return view_of(task_id)
