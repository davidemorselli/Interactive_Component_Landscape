# The frames level of the LLM comparison: the comparison machinery with the
# Entman frame-analysis agents swapped in. Same three agent tables, same run
# table, same average tables and side-by-side cards — everything here is the
# comparison_page module and its views parameterized, not copied: only the
# agents (and so the prompts), the storage table and the URLs differ.
#
# There is no page of its own. /comparison offers both sets of prompts and
# posts to the endpoints of the one its tab bar has chosen
# (comparison_page.PROMPT_SETS), so what is left here is the frames end of
# those endpoints — exactly as frames_page stands to bulk_page.

from flask import request

from llm.pipeline.levels import LEVELS

from .app import app
from .comparison_page import (averages_of, detail_csv_of, run_comparison,
                              side_by_side_of, start_comparison,
                              summary_csv_of)
from .runs import delete_run, list_runs

# The level the comparison runs at, in its frames form: the word level of the
# frames set. Its budget role goes unused — a comparison writes once and never
# rewrites (comparison_page.LEVEL).
LEVEL = LEVELS["frames"].word

TABLE = "frame_comparisons"
RUNS_BASE = "/frame_comparisons"


def run_frame_comparison(run_id, p):
    run_comparison(run_id, p, table=TABLE, level=LEVEL)


@app.post("/frame_comparisons")
def create_frame_comparison():
    return start_comparison(request.get_json(silent=True) or {}, TABLE,
                            run_frame_comparison)


@app.get("/frame_comparisons")
def list_frame_comparisons():
    return list_runs(TABLE)


@app.delete("/frame_comparisons/<int:run_id>")
def delete_frame_comparison(run_id):
    return delete_run(TABLE, run_id)


@app.get("/frame_comparisons/<int:run_id>")
def frame_comparison_averages(run_id):
    return averages_of(run_id, TABLE)


@app.get("/frame_comparisons/<int:run_id>/summary.csv")
def frame_comparison_summary_csv(run_id):
    return summary_csv_of(run_id, TABLE, f"frame_comparison_{run_id}_summary.csv")


@app.get("/frame_comparisons/<int:run_id>/side-by-side")
def frame_comparison_side_by_side(run_id):
    return side_by_side_of(run_id, TABLE, RUNS_BASE,
                           view_title=f"Frames comparison {run_id} side by side")


@app.get("/frame_comparisons/<int:run_id>/csv")
def frame_comparison_csv(run_id):
    return detail_csv_of(run_id, TABLE, f"frame_comparison_{run_id}.csv")
