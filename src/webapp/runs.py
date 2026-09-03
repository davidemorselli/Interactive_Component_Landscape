# The background runs of the pages that have them — the bulk narratives, the
# bulk frames and the two LLM comparisons. All live in SQLite so they survive
# a page reload and can be downloaded as CSV later, each page in a table of
# the same shape, and each operation opens its own connection, so threads
# never share one.

import json
import sqlite3
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

from flask import jsonify, request

from config import DATA_DIR

DB_PATH = DATA_DIR / "tasks.db"

# One row per run: what it was asked for, how far along it is, and what it found.
RUN_COLUMNS = """
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    created TEXT DEFAULT (datetime('now')),
    params  TEXT NOT NULL,
    status  TEXT DEFAULT 'running',
    done    INTEGER DEFAULT 0,
    total   INTEGER,
    error   TEXT,
    results TEXT DEFAULT '[]'"""


def db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


with db() as conn:
    conn.execute(f"CREATE TABLE IF NOT EXISTS comparisons ({RUN_COLUMNS}, meta TEXT)")
    # The frames comparison keeps its own table rather than a column telling
    # the two apart: the rows are read whole by the page that wrote them, and
    # a run of one kind has nothing to say to the other.
    conn.execute("CREATE TABLE IF NOT EXISTS frame_comparisons "
                 f"({RUN_COLUMNS}, meta TEXT)")
    # The task tables' `meta` carries the merge — the summary of summaries, or
    # the merged frame analysis — and the comparison tables' carries what their
    # calls cost (comparison_page.call_stats). The column is named `meta` in
    # every stored row, so it keeps that name however the code around it
    # renames the thing it holds.
    conn.execute(f"CREATE TABLE IF NOT EXISTS tasks ({RUN_COLUMNS}, meta TEXT)")
    conn.execute(f"CREATE TABLE IF NOT EXISTS frames ({RUN_COLUMNS}, meta TEXT)")
    # A run lives in its thread, and threads do not survive the process:
    # anything still 'running' at boot was left mid-flight, and nothing is
    # coming back to finish it — said here once rather than left polling
    # forever on a page. Results already stored stay.
    for table in ("tasks", "frames", "comparisons", "frame_comparisons"):
        conn.execute(f"UPDATE {table} SET status = 'error', "
                     "error = 'interrupted: the app restarted while this run "
                     "was in flight' WHERE status = 'running'")


def row_update(table, row_id, **cols):
    with db() as conn:
        sets = ", ".join(f"{c} = ?" for c in cols)
        conn.execute(f"UPDATE {table} SET {sets} WHERE id = ?", (*cols.values(), row_id))


def row_exists(table, row_id):
    with db() as conn:
        return conn.execute(f"SELECT 1 FROM {table} WHERE id = ?",
                            (row_id,)).fetchone() is not None


def list_runs(table, extra_select="", row_fix=None):
    """Every run of one table, newest first, without the results they hold."""
    with db() as conn:
        rows = conn.execute("SELECT id, created, status, done, total, error"
                            f"{extra_select} FROM {table} ORDER BY id DESC").fetchall()
    runs = [dict(r) for r in rows]
    if row_fix is not None:
        runs = [row_fix(r) for r in runs]
    return jsonify({"runs": runs})


def named_runs(table):
    """Like list_runs, plus each task's name and summary of summaries for the table."""
    return list_runs(
        table, extra_select=", meta, json_extract(params, '$.name') AS name",
        row_fix=lambda r: r | {"meta": json.loads(r["meta"]) if r["meta"] else None})


def rename_run(table, task_id):
    """
    Rename one task. The name is the one thing about a task that changes after
    it is created, and it lives among the parameters the page sent — a task
    from before there were names simply has none.
    """
    name = ((request.get_json(silent=True) or {}).get("name") or "").strip()
    with db() as conn:
        conn.execute(f"UPDATE {table} SET params = json_set(params, '$.name', ?) "
                     "WHERE id = ?", (name, task_id))
    return "", 204


def start_run(table, worker, params, total):
    """Insert one run and set its worker going; the id is what polls it after."""
    with db() as conn:
        run_id = conn.execute(f"INSERT INTO {table} (params, total) VALUES (?, ?)",
                              (json.dumps(params), total)).lastrowid
    threading.Thread(target=worker, args=(run_id, params), daemon=True).start()
    return jsonify({"id": run_id})


def delete_run(table, run_id):
    """Forget one run — which is also what stops it, if it is still going."""
    with db() as conn:
        conn.execute(f"DELETE FROM {table} WHERE id = ?", (run_id,))
    return "", 204


def run_guarded(table, run_id, body):
    """One background worker's whole run: any failure lands on its row."""
    try:
        body()
    except Exception:
        row_update(table, run_id, status="error", error=traceback.format_exc())


# The model calls of one run, this many at a time. The agents are safe to share
# across the threads: they are read-only configuration, and every call builds its
# own client. All the writes stay in the run's own thread.
PARALLEL_CALLS = 16


def fan_out(table, run_id, fn, jobs, on_done):
    """
    What fn answers for each job, in job order, PARALLEL_CALLS at a time.

    Deleting the run lets the calls in flight finish and drops the rest, and
    on_done(answers, done) reports the progress as each one lands.
    """
    answers = [None] * len(jobs)

    def guarded(job):
        if not row_exists(table, run_id):
            return None
        try:
            return fn(job)
        except Exception:
            return None  # one job's failure is one empty answer, not the run's

    with ThreadPoolExecutor(max_workers=PARALLEL_CALLS) as pool:
        futures = {pool.submit(guarded, job): i for i, job in enumerate(jobs)}
        for done, future in enumerate(as_completed(futures), start=1):
            answers[futures[future]] = future.result()
            on_done(answers, done)
    return answers
