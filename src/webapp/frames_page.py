# The frames level of the bulk page: the bulk_page machinery and views
# parameterized, not copied — only the levels (and so the prompts), the
# storage table and the URLs differ. There is no page of its own: /bulk posts
# to the endpoints of whichever set its tab bar chose, and this is the frames
# end of them.

from flask import request

from llm.pipeline.levels import LEVELS
from llm.prompts.frame_topics import frame_topics_prompt

from .app import app
from .params import role_config
from .runs import delete_run, named_runs, rename_run
from .bulk_page import run_task, start_task
from .task_views import csv_of, view_of

# The two levels a frames task is written and graded at — the frames end of
# the one table both pages pick their levels out of.
FRAMES = LEVELS["frames"]


def run_frames_task(task_id, p):
    # `finish` is what the frames level does to its merged analysis that the
    # narrative level does not: put its frames under topics (topic_merge).
    # The bulk narratives run passes none — a narrative has no frames to
    # gather under one.
    run_task(task_id, p, table="frames", levels=FRAMES, finish=topic_merge)


@app.post("/frame_tasks")
def create_frame_task():
    return start_task(request.get_json(silent=True) or {}, "frames",
                      run_frames_task)


@app.get("/frame_tasks")
def list_frame_tasks():
    return named_runs("frames")


@app.patch("/frame_tasks/<int:task_id>")
def rename_frame_task(task_id):
    return rename_run("frames", task_id)


@app.delete("/frame_tasks/<int:task_id>")
def delete_frame_task(task_id):
    return delete_run("frames", task_id)


@app.get("/frame_tasks/<int:task_id>/view")
def frame_task_view(task_id):
    return view_of(task_id, "frames",
                   view_title=f"Frames task {task_id} results")


@app.get("/frame_tasks/<int:task_id>/csv")
def frame_task_csv(task_id):
    return csv_of(task_id, "frames", f"frames_{task_id}.csv")


# --- Frame topics: the merged analysis arranged by subject -------------------
#
# A step of the run, not a button: the merge writes the summary of summaries
# and this puts its frames in topic order before the task is stored. It stays
# BLIND — one call handed the merged analysis and nothing else — which is what
# makes it safe to run unattended, and it is deliberately not a round of the
# refinement loop. Its answer is taken as given except for two whole-analysis
# checks: a NO TOPICS answer, and one holding a different number of frames,
# leave the summary as the merge wrote it — the grades ride along, and a grade
# earned by a frame no longer on the page would be a grade about nothing.


def split_reasoning(answer):
    """The pass's answer as (analysis, reasoning). The reasoning is stored
    beside the analysis, never inside it — a grader would read "I grouped
    frames 3 and 7" as an unsupported claim about the discourse. An answer
    that never wrote the separator keeps its whole text as the analysis."""
    analysis, _, reasoning = answer.partition("---REASONING---")
    reasoning = reasoning.strip()
    if reasoning.upper().startswith("REASONING:"):
        reasoning = reasoning[len("REASONING:"):].strip()
    return analysis.strip(), reasoning


def topic_merge(p, finals, merged):
    """
    The task's summary of summaries with its frames put in topic order, or the
    merge untouched when they fall into no groups. The grouped analysis
    REPLACES the summary rather than standing beside it as one more version —
    it holds the same claims in a different order, so it keeps the grades and
    the place in the run; the drafts under `attempts` stay as the run wrote
    them. Returns what it was given rather than raising: an arrangement is
    never worth losing a summary over.
    """
    summary = merged.get("summary")
    if not summary or merged.get("error"):
        return merged
    try:
        writer = FRAMES.merge.writer(**role_config(p, "summary"))
        grouped, reasoning = split_reasoning(
            writer.ask(frame_topics_prompt(summary), temperature=writer.temperature))
        if (grouped.upper().startswith("NO TOPICS:")
                or grouped.count("FRAME:") != summary.count("FRAME:")):
            return merged
    except Exception as e:
        # Under `note`, never `error`: an error on the merge hides its row
        # (merge_row, the tasks-table cell), and the ungrouped analysis is a
        # summary worth showing — only its arrangement failed.
        return merged | {"note": f"the frames were not grouped: {e}"}
    # Why the frames were arranged as they were, beside the analysis and never
    # inside it — what the trail cell on the page shows.
    return merged | {"summary": grouped, "note": reasoning}
