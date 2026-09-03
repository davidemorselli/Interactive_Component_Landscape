// The helpers every page shares. Loaded after the page defined `const PAGE`
// (its server values: parameters, agent defaults) and before the page's own
// script. The run pages add runs.js for the word list they edit and the
// state they persist.

const $ = id => document.getElementById(id);
const radio = name => document.querySelector(`input[name=${name}]:checked`).value;
const td = text => Object.assign(document.createElement("td"), { textContent: text });

// The parameters this page's endpoint reads, in webapp params.PARAMS order.
const PARAMS = PAGE.params;

// No card offers these any more — the search always runs on unit-normed,
// mean-centred vectors — but they stay on the wire: the endpoints read them
// and the export CSV writes them.
const FIXED = { unit_norm: 1, mean_centre: 1 };

// One value per parameter name, read from the control that carries it: a radio
// group goes by name, anything else by id. The names come from the server, so a
// page cannot send its endpoint less than the endpoint reads.
function collect(names) {
  return Object.fromEntries(names.map(name => {
    if (name in FIXED) return [name, FIXED[name]];
    if (document.querySelector(`input[name=${name}]`)) return [name, radio(name)];
    const el = $(name);
    return [name, el.type === "checkbox" ? (el.checked ? 1 : 0) : el.value];
  }));
}

// --- Agents ----------------------------------------------------------------
// What an agent box starts on, for the page that builds its rows in JS rather
// than from the _agents.html macro: one default per role.
const DEFAULT_AGENT = PAGE.default_agent;
const DEFAULT_ACUE = PAGE.default_acue;
const DEFAULT_GRADER = PAGE.default_grader;

// The agents the pages configure, as the endpoints read them. A page only
// holds the boxes it needs, so the helper skips the ids it does not have.
const AGENTS = { summary: "a_summary", acueval: "a_acueval", rubric: "a_rubric" };

function agentConfig() {
  return Object.fromEntries(Object.entries(AGENTS)
    .filter(([, id]) => $(id + "_model"))
    .map(([name, id]) => [name, { model: $(id + "_model").value.trim() }]));
}

// --- The word controls -----------------------------------------------------
// The enable/disable rules of the nearest-word search, which all three pages
// carry, plus the run pages' word-source tabs: the chosen source's pane is
// the one on screen, so the other sources' settings leave with their tab
// rather than grey out. The landscape page has no tabs and always stands on
// the centroid search. A page with controls beyond these calls it from its
// own syncDisabled.
function syncWordDisabled() {
  const src = document.querySelector("input[name=word_source]:checked");
  if (src) {
    for (const [value, pane] of [["centroid", "pane_centroid"],
                                 ["strong words", "pane_strong"],
                                 ["extreme words", "pane_extreme"]])
      $(pane).hidden = value !== src.value;
    for (const family of ["strong", "extreme"])
      $(family + "_min_count").disabled = !$(family + "_min_count_filter").checked;
  }
  $("n_extreme").disabled = radio("centroid_source") === "strong speakers";
  $("k_extreme").disabled = radio("word_filter") !== "extreme words";
  $("min_count").disabled = !$("min_count_filter").checked;
  $("min_similarity").disabled = !$("similarity_filter").checked;
  // Both halves of the tweet floor follow the box: with it off no count is
  // taken at all, so neither how many tweets it takes to keep a word nor how
  // close they must be has anything left to act on.
  $("word_min_tweets").disabled = !$("tweet_filter").checked;
  $("word_min_similarity").disabled = !$("tweet_filter").checked;
  $("group_similarity").disabled = !$("group_words").checked;
  syncAutoDisabled();
}

// --- Auto top-K ------------------------------------------------------------
// The auto >3σ buttons: pressed, a top-K box follows the component's tail
// (/suggested); typing a value takes the box back. On by default except the
// label count, whose tail is more words than a plot can label. A box names
// the selects it follows when they are not the page's own; `pooledAll` marks
// boxes whose consumer ranks speakers pooled under 'all'; `retrieval` marks
// boxes that feed retrieval, so the run pages do not re-search on re-follow.
const AUTO_BOXES = {
  n_extreme:          { family: "speakers" },
  n_extreme_speakers: { family: "speakers" },
  k_extreme:          { family: "words" },
  n_extreme_words:    { family: "words" },
  q_n_extreme:        { family: "speakers", component: "q_component",
                        phase: "q_phase", pooledAll: true, retrieval: true },
  b_n_extreme:        { family: "speakers", retrieval: true },
  c_n_extreme:        { family: "speakers", retrieval: true },
};
const AUTO_DEFAULT = { n_extreme_words: false };
const autoOn = Object.fromEntries(Object.keys(AUTO_BOXES).filter(id => $(id))
  .map(id => [id, AUTO_DEFAULT[id] ?? true]));
const suggestedCache = {};  // "component|phase" -> /suggested's answer

const autoButton = id => document.querySelector(`button.auto[data-auto=${id}]`);

function paintAuto() {
  for (const id in autoOn) {
    const button = autoButton(id);
    if (!button) continue;
    button.classList.toggle("on", autoOn[id]);
    // The pressed state is painted as a colour; aria-pressed says it in words.
    button.setAttribute("aria-pressed", String(autoOn[id]));
  }
}

// The buttons grey out with their boxes; called from syncWordDisabled, and
// again by the pages whose own rules disable boxes after it ran.
function syncAutoDisabled() {
  for (const id in autoOn) {
    const button = autoButton(id);
    if (button) button.disabled = $(id).disabled;
  }
}

// The (component, phase) one box follows right now.
function autoSource(id) {
  const cfg = AUTO_BOXES[id];
  let phase = $(cfg.phase || "phase").value;
  if (cfg.pooledAll && phase === "all") phase = "pooled";
  return [$(cfg.component || "component").value, phase];
}

// Cached forever: the tail of a (component, phase) never changes in a session.
async function fetchSuggested(component, phase) {
  const key = component + "|" + phase;
  if (!(key in suggestedCache)) {
    const resp = await fetch("/suggested?" + new URLSearchParams({ component, phase }));
    const data = await resp.json();
    if (data.error) return null;  // uncached — the next ask tries again
    suggestedCache[key] = data;
  }
  return suggestedCache[key];
}

// Set every auto-on box among `ids` — every auto box when not given — from
// its component's tail. Deterministic per source, so no sequence guard: two
// racing calls write the same numbers.
async function applyAuto(ids) {
  await Promise.all((ids ?? Object.keys(autoOn)).map(async id => {
    if (!autoOn[id]) return;
    const data = await fetchSuggested(...autoSource(id));
    if (!data) return;  // the box keeps its value
    const el = $(id);   // within the box's own bounds
    el.value = Math.min(Math.max(data[AUTO_BOXES[id].family], +el.min || 1),
                        +el.max || Infinity);
  }));
}

// The boxes a change of this select must refresh — the page handlers ask
// before they redraw, so the redraw already runs with the new values.
function autoBoxesFor(sourceId) {
  return Object.keys(autoOn).filter(id => {
    const cfg = AUTO_BOXES[id];
    return (cfg.component || "component") === sourceId
        || (cfg.phase || "phase") === sourceId;
  });
}

// redraw(id): what the page does when a drawn-from box changes — update() on
// the landscape page, getWords() on the run pages, nothing for a retrieval box.
function wireAuto(redraw) {
  for (const id in autoOn) {
    // Only a user edit fires 'change'; applyAuto's writes do not, so setting
    // a value by hand is exactly what releases the box.
    $(id).addEventListener("change", () => {
      autoOn[id] = false; paintAuto(); if (STORE) saveState();
    });
    autoButton(id)?.addEventListener("click", async () => {
      autoOn[id] = !autoOn[id];
      paintAuto();
      if (STORE) saveState();
      if (!autoOn[id]) return;  // released: the typed value stands
      await applyAuto([id]);
      redraw(id);
    });
  }
  paintAuto();
  syncAutoDisabled();
}

// --- Persistence ------------------------------------------------------------
// localStorage, per browser; saved on every change, restored before wiring.
// Every page names its own store (the run pages in initRunPage, the landscape
// page directly) and fills the two hooks when it keeps state beyond the
// controls and the word tables.
let STORE = null;
let extraState = () => ({});     // page state saved alongside the words
let applyExtraState = () => {};  // and restored from it
const RADIOS = ["centroid_source", "word_filter", "word_source"];

function saveState() {
  const vals = {};
  // Anything without an id — the comparison page's table rows — stays out of
  // this and goes through extraState instead.
  document.querySelectorAll("input[id], select[id]").forEach(el =>
    vals[el.id] = el.type === "checkbox" ? el.checked : el.value);
  // Only the radio groups the page has: the landscape page carries no
  // word-source tabs.
  RADIOS.forEach(name => {
    if (document.querySelector(`input[name=${name}]`)) vals["radio_" + name] = radio(name);
  });
  localStorage.setItem(STORE, JSON.stringify({ vals, tables, auto: autoOn,
                                               ...extraState() }));
}

function loadState() {
  const saved = JSON.parse(localStorage.getItem(STORE) || "null");
  if (!saved) return;
  for (const [id, v] of Object.entries(saved.vals || {})) {
    const el = $(id);
    if (!el) continue;
    // A choice this page no longer offers is dropped rather than restored:
    // assigning a <select> a value it has no option for empties it, and the
    // empty is then rejected by the endpoint. The rendered default stands.
    if (el.tagName === "SELECT" && ![...el.options].some(o => o.value === v)) continue;
    if (el.type === "checkbox") el.checked = v; else el.value = v;
  }
  for (const name of RADIOS) {
    const el = document.querySelector(`input[name=${name}][value="${saved.vals["radio_" + name]}"]`);
    if (el) el.checked = true;
  }
  // The sliders' printed values follow their inline oninput handlers.
  document.querySelectorAll("input[type=range]").forEach(el =>
    el.dispatchEvent(new Event("input")));
  // Which boxes were on auto; a box this browser has never seen keeps its
  // default. Repainted here: the pages wire the buttons before they restore
  // the state behind them.
  for (const id in autoOn)
    if (saved.auto && id in saved.auto) autoOn[id] = saved.auto[id];
  paintAuto();
  // A list saved before the cards is dropped rather than migrated: one click
  // on the search brings it back.
  tables = saved.tables || [];
  applyExtraState(saved);
}

// --- The word tables -------------------------------------------------------
// What /render and /words both answer with: one card per drawn phase, its rows
// already in centroid order. The three pages draw them with the same function,
// so a word list is the same thing wherever it is read — the run pages hand
// each row a × to drop it, the landscape page does not.
let tables = [];

// A word, how far it sits from the centroid, and how much the corpus has to
// say about it. The <wbr>s let a long compound break after its underscores
// rather than mid-syllable.
function wordChip(w, table) {
  const el = document.createElement("span");
  w.word.split(/(?<=_)/).forEach(part => el.append(part, document.createElement("wbr")));
  // No similarity when no centroid chose the word: only the centroid lists
  // carry one. Every list carries the word's value on the component,
  // coloured exactly as the token table colours its cells.
  if (w.similarity != null) {
    el.append(Object.assign(document.createElement("small"),
      { textContent: w.similarity.toFixed(2),
        title: "cosine similarity with the centroid" }));
  }
  if (w.value != null) {
    el.append(Object.assign(document.createElement("small"), {
      className: "ica " + (w.cls || "lvl0"), textContent: w.value.toFixed(2),
      title: `value on component ${table.component} — a z-score within the `
             + "component; green towards its direction, red against it, "
             + "grey when too small to mean anything" }));
  }
  // And how often the corpus writes it — the count the frequency floor and
  // the corpus-frequency order read.
  if (w.count != null) {
    el.append(Object.assign(document.createElement("small"), {
      textContent: `${w.count}×`,
      title: `written ${w.count} times in the corpus — the count the `
             + "frequency floor and the corpus-frequency order read" }));
  }
  // No count without the tweet filter: with its box off nothing was scanned,
  // so the word carries no number rather than a stale or zero one.
  if (w.tweets != null) {
    const where = table.phase === "pooled" ? "in the corpus" : `in phase ${table.phase}`;
    el.append(Object.assign(document.createElement("small"), {
      className: "tweets", textContent: `· ${w.tweets}`,
      title: `${w.tweets} tweets ${where} are within `
             + `${table.tweet_similarity.toFixed(2)} of "${w.word}"`
             + " — what the corpus has to say about it" }));
  }
  return el;
}

function wordRow(r, table, onDrop) {
  const li = document.createElement("li");
  // A row holding several words is one thing, not several: it is boxed, tinted
  // and counted, so a group is told from its neighbours at a glance.
  if (r.words.length > 1) {
    li.className = "grouped";
    li.append(Object.assign(document.createElement("b"), {
      className: "count", textContent: `${r.words.length}×`,
      title: `${r.words.length} words retrieved as one group` }));
  }
  const chips = Object.assign(document.createElement("span"), { className: "chips" });
  chips.append(...r.words.map(w => wordChip(w, table)));
  li.append(chips);
  // The cohesion pair only says something about a group: how alike its words
  // are next to how alike the whole find they came from is. Single words have
  // no cohesion of their own and show nothing.
  if (r.cohesion != null)
    li.append(Object.assign(document.createElement("small"), {
      className: "cohesion",
      textContent: `${r.cohesion.toFixed(2)} vs ${r.list_cohesion.toFixed(2)}`,
      title: `these words agree with each other at ${r.cohesion.toFixed(2)},`
             + ` the whole list they were found in at ${r.list_cohesion.toFixed(2)}`
             + " (average pairwise cosine similarity)" }));
  // A grouped row is one retrieval query, so its tweet count sits on the
  // row: how many tweets are close to the words as one joined query, not to
  // each word on its own.
  if (r.tweets != null) {
    const where = table.phase === "pooled" ? "in the corpus" : `in phase ${table.phase}`;
    li.append(Object.assign(document.createElement("small"), {
      className: "tweets", textContent: `· ${r.tweets}`,
      title: `${r.tweets} tweets ${where} are within `
             + `${table.tweet_similarity.toFixed(2)} of these words as one query`
             + " — the query retrieval will run" }));
  }
  if (onDrop) {
    const del = Object.assign(document.createElement("button"),
      { className: "drop", textContent: "×", title: "drop this row" });
    del.addEventListener("click", onDrop);
    li.append(del);
  }
  return li;
}

// deletable is false where the list is only read — the landscape page.
function phaseCard(t, deletable) {
  const card = Object.assign(document.createElement("div"), { className: "phase-card" });
  card.append(Object.assign(document.createElement("h3"), { textContent: t.title }),
              Object.assign(document.createElement("p"),
                            { className: "meta", textContent: t.meta }));
  // No rows at all is the search being off; an empty list is the floors' work.
  if (t.rows) {
    const ul = document.createElement("ul");
    ul.append(...t.rows.map((r, i) =>
      wordRow(r, t, deletable && (() => dropRow(t, i)))));
    card.append(t.rows.length ? ul
      : Object.assign(document.createElement("p"),
          { className: "meta", textContent: "no words cleared the floors" }));
  }
  return card;
}
