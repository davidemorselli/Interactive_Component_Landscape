// The bulk page, over shared.js and runs.js: creating tasks from the harvested
// words, and the task table with its polls.
//
// The page runs one set of prompts at a time — the tab bar leading Narrative &
// grading, whose machinery is runs.js's, shared with the comparison page. The
// narrative tasks live behind /tasks and the frames tasks behind /frame_tasks,
// so everything here that touches a task asks `runs()` where it once held a
// constant, and switching the tab is switching which runs the table below
// is of.

async function createTask() {
  $("t_error").textContent = "";
  // One task per click: a double-click must not file the task twice.
  $("create_task").disabled = true;
  try {
    const resp = await fetch(runs(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        // What the task is called until it is renamed: the component it came from.
        name: `component ${$("component").value}`,
        words: wordItems(),
        n_tweets: $("b_n_tweets").value,
        min_similarity: $("b_min_similarity").value,
        // The speaker restriction reads the component the words were harvested
        // from — the Selection box above, the same one the name reads.
        speakers: radio("b_speakers"),
        component: $("component").value,
        n_extreme: $("b_n_extreme").value,
        agents: agentConfig(),
        refine: {
          max_attempts: $("max_attempts").value,
          rubric: { enabled: $("rubric_refine").checked, threshold: $("rubric_threshold").value },
          acueval: { enabled: $("acueval_refine").checked, threshold: $("acueval_threshold").value },
          // No threshold: a summary is over its word budget or it is not.
          budget: { enabled: $("budget_refine").checked },
        },
      }),
    });
    const data = await resp.json();
    if (data.error) { $("t_error").textContent = data.error; return; }
    loadTasks();
  } catch (e) {
    $("t_error").textContent = String(e);
  } finally {
    onWordsChange();  // re-enabled the moment there are words to file
  }
}

async function loadTasks() {
  try {
    const RUNS = runs();  // one set per redraw, whatever the tab does meanwhile
    const text = await (await fetch(RUNS)).text();
    // Redrawn only when something changed: the 3-second poll must not wipe a
    // selection being copied out of the table.
    if (text === lastPayload) return;
    lastPayload = text;
    const data = JSON.parse(text);
    $("tasks_body").replaceChildren(...data.runs.map(t => {
      const ren = Object.assign(document.createElement("button"), { textContent: "rename" });
      ren.addEventListener("click", async () => {
        // A prompt rather than a field in the cell: the table redraws every
        // three seconds, which would wipe one mid-edit.
        const name = prompt("Task name", t.name || "");
        if (name === null) return;
        await fetch(`${RUNS}/${t.id}`, {
          method: "PATCH",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name }),
        });
        loadTasks();
      });
      const nameCell = td(t.name || ""); nameCell.append(" ", ren);
      const meta = t.meta || {};
      const gradeTd = (g, floor) => {
        const cell = td(g == null ? "" : (+g).toFixed(2));
        if (g != null) cell.className = g >= floor ? "ok" : "bad";
        return cell;
      };
      // What the words yielded, which done/total does not say: a task can get
      // through every word and still have a narrative for only a few of them.
      // A cell per bucket, so the header says which number is which.
      const c = meta.coverage || {};
      const countTd = key => {
        const cell = td(c[key] == null ? "" : String(c[key]));
        cell.className = "count";
        return cell;
      };
      const mergeCell = td(meta.error || meta.summary || "");
      mergeCell.className = "merge";
      // Why a run stopped, visible under its status — the last line of the
      // stored error is the message; the full traceback stays on the tooltip.
      const statusCell = td(t.status);
      if (t.error)
        statusCell.append(Object.assign(document.createElement("div"),
          { className: "run-error", textContent: t.error.trim().split("\n").pop() }));
      const tr = document.createElement("tr");
      tr.append(td(t.id), nameCell, td(t.created), statusCell, td(`${t.done}/${t.total}`),
                countTd("narratives"), countTd("refusals"), countTd("failed"),
                mergeCell,
                gradeTd(meta.rubric_grade, PAGE.floors.rubric),
                gradeTd(meta.acue_grade, PAGE.floors.acue),
                cell(link(`${RUNS}/${t.id}/view`, "visualize")),
                cell(link(`${RUNS}/${t.id}/csv`, "CSV")),
                cell(deleteButton(`${RUNS}/${t.id}`, loadTasks)));
      if (t.error) tr.title = t.error;  // hover an errored task for its traceback
      return tr;
    }));
  } catch (e) { /* the next poll will say */ }
}

function syncDisabled() {
  syncWordDisabled();
  $("b_n_extreme").disabled = radio("b_speakers") !== "extreme speakers";
  $("rubric_threshold").disabled = !$("rubric_refine").checked;
  $("acueval_threshold").disabled = !$("acueval_refine").checked;
  // With no refinement there is one version per word, so there is nothing for
  // a second attempt to be.
  $("max_attempts").disabled =
    !$("rubric_refine").checked && !$("acueval_refine").checked
    && !$("budget_refine").checked;
  syncAutoDisabled();  // again, after the retrieval box just decided
}

onWordsChange = () => { $("create_task").disabled = !wordItems().length; };
$("create_task").addEventListener("click", createTask);
// What the chosen set names on this page beyond the heading runs.js renames:
// the merge level's two labels, and the budgets the refinement tooltip quotes.
// Written from PAGE rather than rendered per set, so the markup stays one
// page's worth (templates/bulk.html).
function renamed() {
  const set = promptSet();
  $("merge_head").textContent = set.merge_label.toLowerCase();
  $("budget_refine_label").title =
    `Rewrite a summary longer than the word budget its prompt asked for — ` +
    `${set.word_budget} words for a word's, ${set.merge_budget} for the ` +
    `${set.merge_label.toLowerCase()}, each with a fifth over allowed. Counted ` +
    `here, not judged by a model. It is the last refinement tried: what a ` +
    `summary says comes before how long it is.`;
}

// `restored` rather than `syncDisabled`: the labels follow the set the state
// brought back, and nothing about them is enabled or greyed.
const restored = wirePromptTabs({ body: "tasks_body", load: loadTasks, renamed });
initRunPage({ store: "bulk_page", speakersRadio: "b_speakers",
              syncDisabled, restored, load: loadTasks });
