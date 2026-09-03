// The run pages' half, over shared.js: the editable word list, the run-table
// cell helpers, and the page bootstrap. The persistence behind the controls
// lives in shared.js, where the landscape page shares it.

// --- The editable word list ------------------------------------------------
let onWordsChange = () => {};  // the page enables whatever the words unlock

// The rows as the run endpoints read them: one {phase, word} each, a grouped
// row's words joined by a space — one retrieval query, one narrative.
const wordItems = () =>
  tables.flatMap(t => (t.rows || []).map(
    r => ({ phase: t.phase, word: r.words.map(w => w.word).join(" ") })));

// A phase with nothing left in it is not one the run has anything to ask
// about, so the card goes with its last row.
function dropRow(table, i) {
  table.rows.splice(i, 1);
  if (!table.rows.length) tables.splice(tables.indexOf(table), 1);
  renderTables();
}

// The cards, what they add up to, and the state saved behind them. Every
// change to the list goes through here.
function renderTables() {
  $("tables").replaceChildren(...tables.map(t => phaseCard(t, true)));
  const items = wordItems();
  const words = items.reduce((n, item) => n + item.word.split(" ").length, 0);
  const count = (n, one, many) => `${n} ${n === 1 ? one : many}`;
  // Grouped, the two differ: one query per row, several words inside it.
  $("words_count").textContent =
    !items.length ? "no words yet"
    : items.length === words ? count(words, "word", "words")
    : `${count(items.length, "query", "queries")} · ${count(words, "word", "words")}`;
  saveState();
  onWordsChange();
}

let wseq = 0;
async function getWords() {
  const my = ++wseq;
  $("q_status").textContent = "searching for the nearest words…";
  $("error").textContent = "";
  try {
    const data = await (await fetch("/words?" + new URLSearchParams(collect(PARAMS)))).json();
    if (my !== wseq) return;  // superseded by a newer set of parameters
    if (data.error) { $("error").textContent = data.error; return; }
    $("error").textContent = data.note || "";
    // One search, one list: the find takes the place of whatever was there, so
    // a list only ever holds one component's words under one set of floors. A
    // search that failed leaves the previous list standing.
    tables = data.tables;
    renderTables();
  } catch (e) {
    if (my === wseq) $("error").textContent = String(e);
  } finally {
    if (my === wseq) $("q_status").textContent = "";
  }
}

// The wiring the word controls always want, plus the per-page syncDisabled the
// controls share.
function wireWords(syncDisabled) {
  // The word controls redraw the list, the way the landscape page's redraw its
  // plot: what the controls say is what is shown, with nothing to press. Rows
  // dropped by hand go with it — the search is the list. A new component or
  // phase first re-points the auto boxes at its own tail, so the search that
  // follows already runs with them.
  document.querySelectorAll("#word-controls input, #word-controls select").forEach(el =>
    el.addEventListener("change", async () => {
      syncDisabled();
      const boxes = autoBoxesFor(el.id);
      if (boxes.length) await applyAuto(boxes);
      getWords();
    }));
  // A retrieval box re-following changes no word search — only the task the
  // page will create; the wiring above already saved it.
  wireAuto(id => { if (!AUTO_BOXES[id].retrieval) getWords(); });
  // The narrative and grading boxes only have to be remembered.
  document.querySelectorAll(".controls:not(#word-controls) input, " +
                            ".controls:not(#word-controls) select").forEach(el =>
    el.addEventListener("change", () => { syncDisabled(); saveState(); }));
}

// A list restored from the last visit stands as it was left, dropped rows and
// all — its controls with it, so the auto boxes are not refreshed over them.
// A browser with none first points the auto boxes at the component, then gets
// the search its controls describe.
function startWords() {
  if (tables.length) renderTables(); else applyAuto().then(getWords);
}

// --- The runs table --------------------------------------------------------
const cell = el => { const c = document.createElement("td"); c.append(el); return c; };
const link = (href, text) =>
  Object.assign(document.createElement("a"), { href, textContent: text });

// The × that forgets a run — which also stops it, if it is still going.
function deleteButton(url, reload) {
  const del = Object.assign(document.createElement("button"), { textContent: "×" });
  del.addEventListener("click", async () => {
    await fetch(url, { method: "DELETE" });
    reload();
  });
  return del;
}

// --- The prompt set --------------------------------------------------------
// Both run pages offer the same two sets and run one at a time. The set is a
// URL: everything a page files, polls, views and deletes follows the chosen
// set to its `base` (params.prompt_sets, reaching the page as
// PAGE.prompt_sets).

// The chosen set, as the page was given it.
const promptSet = () => PAGE.prompt_sets[radio("prompts")];

// Where this set's runs live.
const runs = () => promptSet().base;

// The payload the run table was last drawn from, so a poll that changed
// nothing does not redraw a table someone is selecting text in. Cleared on a
// tab switch: two sets can answer the very same bytes (two empty lists do).
let lastPayload = null;

// The tab bar's wiring; `renamed` is also handed back for initRunPage's
// `restored`, since what a page renames must follow the restored set too.
// The heading is not touched — the tab bar already says which set is chosen.
// A switch empties `body` before it reloads: rows left on screen would show
// a run under the wrong prompts, links and all.
function wirePromptTabs({ body, load, renamed = () => {} }) {
  document.querySelectorAll("input[name=prompts]").forEach(el =>
    el.addEventListener("change", () => {
      renamed();
      saveState();
      lastPayload = null;
      $(body).replaceChildren();
      load();
    }));
  // Persisted with the controls, like the word radios and the speaker one: the
  // page opens on the set it was left on.
  RADIOS.push("prompts");
  return renamed;
}

// The wiring both run pages end on, once their own listeners are registered:
// restore the saved state, draw everything from it, and keep the run table
// polling. `restored` is the page's chance to redraw what loadState brought
// back beyond the controls.
function initRunPage({ store, speakersRadio, syncDisabled, restored, load }) {
  STORE = store;
  RADIOS.push(speakersRadio);  // persisted with the controls, like the word radios
  wireWords(syncDisabled);
  loadState();
  syncDisabled();
  restored?.();
  startWords();
  load();
  setInterval(load, 3000);
}
