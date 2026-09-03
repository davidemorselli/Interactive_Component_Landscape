// The LLM-comparison page, over shared.js and runs.js: the three agent tables,
// launching runs, and the run table with its summary views.
//
// The page compares models on one set of prompts at a time — the tab bar
// leading Models & judges, whose machinery is runs.js's, shared with the bulk
// page. The narrative runs live behind /comparisons and the frames runs behind
// /frame_comparisons, so everything here that touches a run asks `runs()`, and
// switching the tab is switching which runs the table below is of. The agent
// tables themselves are the page's own and stand under both: whichever prompts
// are chosen, it is the same models being compared.

// --- The models compared and the judges ------------------------------------
// Three tables of the same shape, so one function draws all three. Each starts
// on the model the agent boxes of the other pages start on. The rows carry no
// id, so they ride along with the saved state rather than being read off it.
const TABLES = ["agents", "acue", "rubric"];
let lists = { agents: [{ ...DEFAULT_AGENT }],
              acue: [{ ...DEFAULT_ACUE }],
              rubric: [{ ...DEFAULT_GRADER }] };

function renderTable(kind) {
  $(kind + "_body").replaceChildren(...lists[kind].map((row, i) => {
    const model = Object.assign(document.createElement("input"),
      { type: "text", value: row.model, placeholder: "model identifier" });
    model.addEventListener("change", () => { row.model = model.value.trim(); saveState(); });
    const del = Object.assign(document.createElement("button"), { textContent: "×" });
    del.addEventListener("click", () => { lists[kind].splice(i, 1); renderTable(kind); });
    const tr = document.createElement("tr");
    tr.append(cell(model), cell(del));
    return tr;
  }));
  saveState();
  syncLaunch();
}

// There is a comparison to run once there are words, someone to write the
// summaries and someone to grade them.
function syncLaunch() {
  $("launch").disabled = !(wordItems().length && lists.agents.length
                           && (lists.acue.length || lists.rubric.length));
}

// --- Runs ------------------------------------------------------------------
async function launch() {
  $("r_error").textContent = "";
  // One run per click: a double-click must not launch the grading twice.
  $("launch").disabled = true;
  try {
    const resp = await fetch(runs(), {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        words: wordItems(),
        n_tweets: $("c_n_tweets").value,
        min_similarity: $("c_min_similarity").value,
        // The speaker restriction reads the component the words were harvested
        // from — the Selection box above, as on the bulk page.
        speakers: radio("c_speakers"),
        component: $("component").value,
        n_extreme: $("c_n_extreme").value,
        agents: lists.agents,
        acue_graders: lists.acue,
        rubric_graders: lists.rubric,
      }),
    });
    const data = await resp.json();
    if (data.error) { $("r_error").textContent = data.error; return; }
    loadRuns();
  } catch (e) {
    $("r_error").textContent = String(e);
  } finally {
    syncLaunch();  // re-enabled the moment there is a comparison to run
  }
}

async function loadRuns() {
  try {
    const RUNS = runs();  // one set per redraw, whatever the tab does meanwhile
    const text = await (await fetch(RUNS)).text();
    // Redrawn only when something changed: the 3-second poll must not wipe a
    // selection being copied out of the table.
    if (text === lastPayload) return;
    lastPayload = text;
    const data = JSON.parse(text);
    $("runs_body").replaceChildren(...data.runs.map(r => {
      const show = Object.assign(document.createElement("button"),
                                 { textContent: "show summary" });
      show.addEventListener("click", () => showAverages(r.id));
      // Why a run stopped, visible under its status — the last line of the
      // stored error is the message; the full traceback stays on the tooltip.
      const statusCell = td(r.status);
      if (r.error)
        statusCell.append(Object.assign(document.createElement("div"),
          { className: "run-error", textContent: r.error.trim().split("\n").pop() }));
      const tr = document.createElement("tr");
      tr.append(td(r.id), td(r.created), statusCell, td(`${r.done}/${r.total}`),
                cell(show),
                cell(link(`${RUNS}/${r.id}/summary.csv`, "download summary")),
                cell(link(`${RUNS}/${r.id}/side-by-side`, "compare side by side")),
                cell(link(`${RUNS}/${r.id}/csv`, "download detailed results")),
                cell(deleteButton(`${RUNS}/${r.id}`, loadRuns)));
      if (r.error) tr.title = r.error;  // hover an errored run for its traceback
      return tr;
    }));
  } catch (e) { /* the next poll will say */ }
}

// The comparison itself is the first table: the average grade of each agent.
// The next two say how lenient each judge was, and the last what the run's
// calls cost — retries and failures per model, which the grades cannot show
// because a failed call leaves no row to grade.
async function showAverages(id) {
  $("r_error").textContent = "";
  $("averages").replaceChildren();
  try {
    const data = await (await fetch(`${runs()}/${id}`)).json();
    if (data.error) {
      $("r_error").textContent = data.error;
      return;
    }
    // A run that graded nothing still reports what its calls cost, and that
    // report is often what says why nothing was graded — so the note is shown
    // beside whatever tables came back, not instead of them.
    if (data.note) $("r_error").textContent = data.note;
    // A run stored before its calls were counted has no per-model table, so
    // the blocks are filtered rather than assumed.
    const blocks = [
      ["Average per summary agent", data.per_agent],
      ["Average per ACUEval grader", data.per_acue],
      ["Average per RUBRIC grader", data.per_rubric],
      ["Calls, retries and failures per model", data.per_call],
    ].filter(([, table]) => table);
    if (!blocks.length) return;
    // Led by the run the tables belong to — clicking another run swaps them
    // silently otherwise — and brought on screen, since they render below a
    // table that can be taller than the window.
    $("averages").innerHTML = `<h3 class="avg-head">Run ${id}</h3>` +
      blocks.map(([heading, table]) => `<div><h3>${heading}</h3>${table}</div>`).join("");
    $("averages").scrollIntoView({ block: "nearest", behavior: "smooth" });
  } catch (e) {
    $("r_error").textContent = String(e);
  }
}

// --- Wiring ----------------------------------------------------------------
function syncDisabled() {
  syncWordDisabled();
  $("c_n_extreme").disabled = radio("c_speakers") !== "extreme speakers";
  syncAutoDisabled();  // again, after the retrieval box just decided
}

extraState = () => ({ lists });
applyExtraState = saved => { lists = saved.lists || lists; };
onWordsChange = syncLaunch;
for (const kind of TABLES)
  $(kind + "_add").addEventListener("click", () => {
    lists[kind].push({ ...{ agents: DEFAULT_AGENT, acue: DEFAULT_ACUE,
                            rubric: DEFAULT_GRADER }[kind] });
    renderTable(kind);
  });
$("launch").addEventListener("click", launch);
// The averages of a run of the other set would be read under this one's
// heading, so they go with the table they were opened from.
const restored = wirePromptTabs({
  body: "runs_body", load: loadRuns,
  renamed: () => { $("averages").replaceChildren(); $("r_error").textContent = ""; }});
initRunPage({ store: "comparison_page", speakersRadio: "c_speakers", syncDisabled,
              restored: () => { restored(); TABLES.forEach(renderTable); },
              load: loadRuns });
