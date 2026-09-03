// The landscape page, over shared.js: the plot and its word tables, the
// drawn-token export, the per-token component table, and the on-demand tweet
// narrative with its grades and refinements.

// Same enable/disable wiring as the notebook widgets: the nearest-word rules
// every page shares, plus the two overlays only this one draws.
function syncDisabled() {
  syncWordDisabled();
  $("n_extreme_speakers").disabled = !$("show_extreme_speakers").checked;
  $("n_extreme_words").disabled = !$("extreme_labels").checked;
  // The two label families carry the same three settings, each live only
  // while its own overlay is drawn, and each floor only while its box is on.
  for (const [family, drawn] of [["strong", "strong_labels"],
                                 ["extreme", "extreme_labels"]]) {
    const on = $(drawn).checked;
    $(family + "_min_count_filter").disabled = !on;
    $(family + "_min_count").disabled = !on || !$(family + "_min_count_filter").checked;
    $(family + "_order").disabled = !on;
  }
  $("n_strong_words").disabled = !$("strong_labels").checked;
  syncAutoDisabled();  // again, after the two display boxes just decided
}

let seq = 0;
async function update() {
  const my = ++seq;
  $("status").textContent = "rendering…";
  $("plot").classList.add("stale");
  $("error").textContent = "";
  try {
    const resp = await fetch("/render?" + new URLSearchParams(collect(PARAMS)));
    const data = await resp.json();
    if (my !== seq) return;  // superseded by a newer request
    if (data.error) { $("error").textContent = data.error; }
    else {
      $("plot").src = "data:image/png;base64," + data.image;
      // The figure's own title, for whoever cannot see the image.
      $("plot").alt = `Component ${$("component").value} landscape — `
                    + `phase ${$("phase").value}`;
      // The same cards the run pages keep, drawn from the same function —
      // read-only here, so no row carries a × to drop it.
      tables = data.tables;
      $("tables").replaceChildren(
        ...(data.note ? [Object.assign(document.createElement("p"),
                                       { textContent: data.note })] : []),
        ...tables.map(t => phaseCard(t, false)));
      refreshExport();  // the same parameters the plot was just drawn from
    }
  } catch (e) {
    if (my === seq) $("error").textContent = String(e);
  } finally {
    if (my === seq) { $("status").textContent = ""; $("plot").classList.remove("stale"); }
  }
}

// --- Export of the drawn tokens --------------------------------------------
// The table under the plot and the CSV hold exactly the tokens on it: both are
// asked for with the parameters of the render that drew them.
// The plot legend's colours, so the type column reads back to it.
const TYPE_COLOURS = { "strong speaker": "orange", "extreme speaker": "deeppink",
                       "strong word": "#9a3412", "extreme word": "deeppink",
                       "nearest word": "black" };

function typeCell(type) {
  const dot = Object.assign(document.createElement("i"), { className: "dot" });
  dot.style.background = TYPE_COLOURS[type] ?? "#999";
  const cell = td(type);
  cell.prepend(dot);
  return cell;
}

let eseq = 0;
async function refreshExport() {
  const my = ++eseq;
  try {
    const resp = await fetch("/export?" + new URLSearchParams(collect(PARAMS)));
    const data = await resp.json();
    if (my !== eseq || data.error) return;  // superseded, or the plot's error box says it
    $("export_box").hidden = false;
    $("export_count").textContent = data.rows.length;
    $("export_note").hidden = data.rows.length > 0;
    $("export_btn").disabled = !data.rows.length;
    const head = document.createElement("tr");
    head.append(...["phase", "type", "token", "occurrences", "value"].map(text =>
      Object.assign(document.createElement("th"), { textContent: text })));
    head.children[3].title = "how often the corpus writes the token — the "
      + "count the frequency floors read";
    // No lone header over an empty table, as with the token table below. The
    // value cell carries the server's colour class — the token table's scale.
    $("export_table").replaceChildren(...(data.rows.length ? [head] : []),
      ...data.rows.map(r => {
        const tr = document.createElement("tr");
        tr.append(td(r.phase), typeCell(r.type), td(r.token), td(r.count),
                  Object.assign(td(r.value.toFixed(2)), { className: r.cls }));
        return tr;
      }));
  } catch { /* leave the last table standing; the next render tries again */ }
}

$("export_btn").addEventListener("click", () =>
  location.href = "/export.csv?" + new URLSearchParams(collect(PARAMS)));


// Coalesced redraws: every arrow-click on a number box fires 'change', and
// each render costs the server a full figure — so the update waits for the
// controls to settle for a beat instead of queueing one render per click.
let updateTimer = null;
function scheduleUpdate() {
  clearTimeout(updateTimer);
  updateTimer = setTimeout(update, 300);
}

// Scoped to the landscape controls: the tweet controls below must not
// trigger a re-render of the plot. A new component or phase first re-points
// the auto boxes at its own tail, so the render already draws with them.
document.querySelectorAll("#landscape-controls input, #landscape-controls select").forEach(el =>
  el.addEventListener("change", async () => {
    syncDisabled();
    saveState();
    const boxes = autoBoxesFor(el.id);
    if (boxes.length) await applyAuto(boxes);
    scheduleUpdate();
  }));
// The controls come back as they were left, like the run pages'; the plot is
// always redrawn from them, so only the settings persist, never a stale figure.
STORE = "landscape_page";
RADIOS.push("q_speakers");  // persisted with the controls, like the word radios
RADIOS.push("prompts");     // and the prompt set the narrative is written with
loadState();
syncDisabled();
// The retrieval box re-runs the tweet search instead of the plot — the same
// thing changing it by hand does.
wireAuto(id => id === "q_n_extreme" ? search() : update());
applyAuto().then(update);


// --- Token on the components -----------------------------------------------
// The server sends each cell its colour class along with its value, the way the
// task view does: what a value means depends on the component's direction, and
// that is the server's to know.
const tokenCell = (tag, text, cls) =>
  Object.assign(document.createElement(tag), { textContent: text, className: cls ?? "" });

function syncTokenDisabled() {
  $("t_min_count").disabled = !$("t_min_count_filter").checked;
}

let tseq = 0;
async function showToken() {
  const my = ++tseq;
  $("t_error").textContent = "";
  try {
    // Unticked, the floor is one: every word the tokenizer kept is a neighbour.
    const resp = await fetch("/token?" + new URLSearchParams(
      { token: $("t_query").value, topn: $("t_topn").value,
        min_count: $("t_min_count_filter").checked ? $("t_min_count").value : 1 }));
    const data = await resp.json();
    if (my !== tseq) return;  // superseded by a newer token
    if (data.error) { $("t_error").textContent = data.error; return; }
    $("t_note").textContent = data.note ?? "";
    // A minus sign rather than a hyphen: it is read as a sign, next to a number.
    const head = document.createElement("tr");
    head.append(...["word", "sim",
                    ...data.components.map(c => `${c.component} ${c.direction === "+" ? "+" : "−"}`)]
                .map(text => tokenCell("th", text)));
    // No lone header over an empty table: nothing found clears it entirely.
    $("t_table").replaceChildren(...(data.rows.length ? [head] : []), ...data.rows.map(r => {
      const tr = document.createElement("tr");
      tr.append(tokenCell("td", r.word), tokenCell("td", r.similarity.toFixed(2)),
                ...r.cells.map(c => tokenCell("td", c.value.toFixed(2), c.cls)));
      return tr;
    }));
  } catch (e) {
    if (my === tseq) $("t_error").textContent = String(e);
  }
}


// --- Tweet narratives ------------------------------------------------------
let lastSearch = null;  // {query, tweets} the narrative button summarises

function collectTweets() {
  return {
    query: $("q_query").value,
    n_tweets: $("q_n_tweets").value,
    phase: $("q_phase").value,
    speakers: radio("q_speakers"),
    component: $("q_component").value,
    n_extreme: $("q_n_extreme").value,
    min_similarity: $("q_min_similarity").value,
  };
}

function syncTweetDisabled() {
  const source = radio("q_speakers");
  $("q_component").disabled = source === "all";
  $("q_n_extreme").disabled = source !== "extreme speakers";
  syncAutoDisabled();  // its auto button greys out with it
}

let qseq = 0;
async function search() {
  const my = ++qseq;
  lastSearch = null;
  $("q_narrative_btn").disabled = true;
  $("q_narrative").textContent = "";
  $("q_grades").replaceChildren();
  $("q_error").textContent = "";
  $("q_status").textContent = "searching…";
  try {
    const resp = await fetch("/tweets?" + new URLSearchParams(collectTweets()));
    const data = await resp.json();
    if (my !== qseq) return;  // superseded
    if (data.error) { $("q_error").textContent = data.error; return; }
    $("q_note").textContent = data.note ?? "";
    $("q_tweets").innerHTML = data.html ?? "";
    if (data.tweets.length) {
      lastSearch = { query: collectTweets().query.trim(), tweets: data.tweets };
      $("q_narrative_btn").disabled = false;
    }
  } catch (e) {
    if (my === qseq) $("q_error").textContent = String(e);
  } finally {
    if (my === qseq) $("q_status").textContent = "";
  }
}

// A duration, no more precisely than it is worth reading: 4.2s, then 37s.
const secs = s => s < 10 ? `${s.toFixed(1)}s` : `${Math.round(s)}s`;

// One grade per card; textContent throughout, the values are model output.
// The time the grader took sits at the right of the card's title line.
function gradeCard(title, value, detail, seconds) {
  const div = document.createElement("div");
  div.className = "grade-card";
  const head = Object.assign(document.createElement("span"), { textContent: title });
  if (seconds != null)
    head.append(Object.assign(document.createElement("i"),
                              { className: "timing", textContent: secs(seconds) }));
  div.append(head,
             Object.assign(document.createElement("b"), { textContent: value }),
             Object.assign(document.createElement("small"), { textContent: detail }));
  return div;
}

// One line per atomic claim, ticked or crossed; textContent, it is model output.
function claimList(claims) {
  const ul = document.createElement("ul");
  for (const c of claims) {
    const li = document.createElement("li");
    li.className = c.supported === null ? "unjudged" : c.supported ? "yes" : "no";
    li.textContent = c.unit;
    ul.append(li);
  }
  return ul;
}

// The button that rewrites the summary against this grader's critique. Nothing
// is added when the critique is empty: a grade with nothing to fix has nothing
// to say, and the model would only be asked to rewrite for the sake of it.
function addRefineButton(card, grader, feedback, summary) {
  if (!feedback) return;
  const button = document.createElement("button");
  button.className = "refine";
  button.textContent = "Refine summary";
  button.title = "Rewrite the summary against this critique, then grade it again";
  button.addEventListener("click", () => refine(grader, feedback, summary));
  card.append(button);
}

async function grade(summary) {
  $("q_status").textContent = "grading the summary… (one LLM call per claim)";
  const resp = await fetch("/grade", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...lastSearch, summary, agents: agentConfig(),
                           prompts: radio("prompts") }),
  });
  const data = await resp.json();
  if (data.error) { $("q_error").textContent = data.error; return; }
  const a = data.acueval;
  const acu = gradeCard("ACUEval", a.score === null ? "—" : a.score.toFixed(2),
                        `${a.supported}/${a.units} claims supported by the tweets`,
                        a.seconds);
  acu.append(claimList(a.claims));  // the claims themselves, inside the card
  addRefineButton(acu, "acueval", a.feedback, summary);
  const rub = data.rubric
    ? gradeCard("RUBRIC",`${data.rubric.average.toFixed(1)}/5`,
                Object.entries(data.rubric.criteria).map(([c, g]) => `${c} ${g}/5`).join(" · ")
                + " — " + data.rubric.reason, data.rubric_seconds)
    : gradeCard("RUBRIC","—", "the grader did not answer with valid grades",
                data.rubric_seconds);
  if (data.rubric) addRefineButton(rub, "rubric", data.rubric.feedback, summary);
  // No timing: the count is arithmetic on this side of the wire, and no model
  // was asked. The button appears only when the summary is over budget, which
  // is the only case the critique is non-empty.
  const b = data.budget;
  const len = gradeCard("LENGTH", `${b.words} word${b.words === 1 ? "" : "s"}`,
                        b.over ? `over the ${b.budget}-word budget`
                               : `within the ${b.budget}-word budget`);
  addRefineButton(len, "budget", b.feedback, summary);
  $("q_grades").replaceChildren(acu, rub, len);
}

// The summary on the page is the one the buttons act on: a refinement takes
// the place of what it rewrites, and is graded in its turn. Under it, how long
// it is and how long the model took to write it — a rewrite says so, being a
// second pass. The length is worth a glance: the prompt asks for 50 words.
function showSummary(text, seconds, verb = "written") {
  const p = document.createElement("p");
  p.textContent = text;  // textContent, not innerHTML: the summary is model output.
  const words = (text.match(/\S+/g) || []).length;
  const meta = [`${words} word${words === 1 ? "" : "s"}`];
  if (seconds != null) meta.push(`${verb} in ${secs(seconds)}`);
  $("q_narrative").replaceChildren(
    p, Object.assign(document.createElement("div"),
                     { className: "timing", textContent: meta.join(" · ") }));
}

async function refine(grader, feedback, summary) {
  // The cards stay up while the model writes — a failed rewrite leaves the
  // grades that were on screen, and the buttons that offer it again.
  const buttons = $("q_grades").querySelectorAll("button.refine");
  buttons.forEach(b => { b.disabled = true; });
  $("q_narrative_btn").disabled = true;
  $("q_error").textContent = "";
  $("q_status").textContent = "rewriting the summary…";
  try {
    const resp = await fetch("/refine", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...lastSearch, summary, grader, feedback,
                             agents: agentConfig(), prompts: radio("prompts") }),
    });
    const data = await resp.json();
    if (data.error) { $("q_error").textContent = data.error; return; }
    showSummary(data.summary, data.seconds, "rewritten");
    $("q_grades").replaceChildren();  // these grades are about the old summary
    await grade(data.summary);  // and these are about the one now on the page
  } catch (e) {
    $("q_error").textContent = String(e);
  } finally {
    $("q_status").textContent = "";
    $("q_narrative_btn").disabled = !lastSearch;
    buttons.forEach(b => { b.disabled = false; });
  }
}

async function narrative() {
  if (!lastSearch) return;
  $("q_narrative_btn").disabled = true;
  $("q_error").textContent = "";
  $("q_grades").replaceChildren();
  $("q_status").textContent = "asking the LLM…";
  try {
    const resp = await fetch("/narrative", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...lastSearch, agents: agentConfig(),
                             prompts: radio("prompts") }),
    });
    const data = await resp.json();
    if (data.error) { $("q_error").textContent = data.error; }
    else {
      showSummary(data.summary, data.seconds);
      await grade(data.summary);
    }
  } catch (e) {
    $("q_error").textContent = String(e);
  } finally {
    $("q_status").textContent = "";
    $("q_narrative_btn").disabled = !lastSearch;
  }
}

// The agent column is left out: naming another model is not a new search. A
// new retrieval component or phase first re-points the N extreme box at its
// tail, so the search already runs with it.
document.querySelectorAll("#tweet-controls .col:not(#agent-col) :is(input, select)").forEach(el =>
  el.addEventListener("change", async () => {
    syncTweetDisabled();
    saveState();
    const boxes = autoBoxesFor(el.id);
    if (boxes.length) await applyAuto(boxes);
    search();
  }));
// The agent and token boxes only have to be remembered.
document.querySelectorAll("#agent-col input, #token-controls input").forEach(el =>
  el.addEventListener("change", saveState));
// So is the prompt set: it changes nothing already on the page — the tweets
// retrieved are the same tweets either way — and is read at the moment the
// next narrative is asked for. Wired here rather than with the retrieval
// boxes above, which re-run the search on every change.
document.querySelectorAll("input[name=prompts]").forEach(el =>
  el.addEventListener("change", saveState));
$("t_query").addEventListener("keydown", e => { if (e.key === "Enter") showToken(); });
$("t_topn").addEventListener("change", showToken);
$("t_min_count").addEventListener("change", showToken);
$("t_min_count_filter").addEventListener("change", () => { syncTokenDisabled(); showToken(); });
$("t_show").addEventListener("click", showToken);
$("q_query").addEventListener("keydown", e => { if (e.key === "Enter") search(); });
$("q_search").addEventListener("click", search);
$("q_narrative_btn").addEventListener("click", narrative);
syncTokenDisabled();
syncTweetDisabled();
