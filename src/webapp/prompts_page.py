# Every prompt the app sends, on one page to be read. The texts are built by
# the same code that sends them — the level table (llm.pipeline.levels) and
# the critique templates — with ⟨placeholders⟩ standing where a run's data
# goes, so the page cannot drift from what the runs say. The placeholders are
# the same at every level, which is also what lets a prompt one level reuses
# from another build the identical text and be shown once (_dedup).

from flask import render_template

from llm.pipeline.acueval import FEEDBACK as ACUEVAL_FEEDBACK
from llm.pipeline.acueval import decomposition_messages
from llm.pipeline.levels import LEVELS
from llm.pipeline.rubric import FEEDBACK_PROMPT, REDO_PROMPT
from llm.prompts.frame_topics import frame_topics_prompt
from llm.prompts.narrative import source_block

from .app import app
from .params import DEFAULT_PROMPT_SET, PROMPT_LABELS

# What stands where a run's data would. Three sources, so the "---" joins
# every prompt shows its sources in are visible; the prompts themselves say
# what the sources are at each level (tweets, narratives, frame analyses).
QUERY = "⟨query⟩"
SOURCES = ["⟨source 1⟩", "⟨source 2⟩", "⟨source 3⟩"]
ANSWER = "⟨the answer being graded⟩"


def _turns(conversation):
    """A conversation's turns as (role, text) pairs, ready to lay out."""
    return [(turn["role"], turn["content"]) for turn in conversation]


def _user(text):
    """A one-string prompt, as the single user turn it is sent as."""
    return [("user", text)]


def _level_prompts(level):
    """One level's prompts, each grading followed by the refinement critique
    it can send back — the critique is not a conversation of its own but the
    next user turn of the writing conversation, replayed with the answer as
    the model's own turn (llm.pipeline.refining; the blurb says so once for
    the page). RUBRIC and ACUEval are spelled as the run pages spell them."""
    return [
        ("Writer", "asks for the answer", _turns(level.prompt(QUERY, SOURCES))),
        ("RUBRIC grading", "grades an answer against the level's rubric",
         _user(level.grading_prompt(QUERY, ANSWER, SOURCES))),
        ("RUBRIC refinement", "asks for a rewrite over the criteria short of 5",
         _user(FEEDBACK_PROMPT.format(
             grades="- ⟨a criterion⟩: ⟨its grade⟩/5",
             reason="⟨the grader's reason⟩"))),
        ("RUBRIC grading of a refusal", "grades the decision to decline",
         _user(level.abstention_prompt(QUERY, SOURCES))),
        ("RUBRIC refinement after refusal", "asks again after a refusal "
         "graded wrong",
         _user(REDO_PROMPT.format(grade="⟨its grade⟩",
                                  reason="⟨the grader's reason⟩"))),
        ("ACUEval decomposition", "splits an answer into its content units",
         _turns(decomposition_messages(level.decomposition, ANSWER))),
        ("ACUEval verification", "checks one unit; one call per unit",
         _user(level.verification.format(context=source_block(SOURCES),
                                         unit="⟨one unit of the answer⟩"))),
        ("ACUEval refinement", "asks for a rewrite over the units the "
         "sources do not bear out",
         _user(ACUEVAL_FEEDBACK.format(
             claims="- ⟨a unit the sources do not support⟩"))),
    ]


def _dedup(groups):
    """The same text shown once per tab: a prompt a later level reuses
    unchanged — the merges' abstention and decomposition are the word
    levels' — becomes a line saying so, in place of the second copy. Per tab,
    never across tabs: a pointer must never point at a pane that is off
    screen."""
    seen = {}
    for group in groups:
        for prompt in group["prompts"]:
            key = tuple(prompt["turns"])
            if key in seen:
                prompt["same"] = ("The {} prompt of {}, reused unchanged."
                                  .format(*seen[key]))
                prompt["turns"] = []
            else:
                seen[key] = (prompt["name"], group["title"])
    return groups


def _entries(entries):
    """(name, note, turns) rows as the dicts the template lays out."""
    return [{"name": name, "note": note, "turns": turns, "same": None}
            for name, note, turns in entries]


def _sets():
    """One tab per prompt set, in the order every tab bar offers them: the
    set's word and merge levels, the frames set's topic pass, then the budget
    critique — the one refinement with no grading conversation to follow."""
    sets = []
    for key, label in PROMPT_LABELS.items():
        pair = LEVELS[key]
        groups = [
            {"title": pair.word.label,
             "note": "the word level — one answer per word of a task, "
                     f"held to about {pair.word.words} words",
             "prompts": _entries(_level_prompts(pair.word))},
            {"title": pair.merge.label,
             "note": "the merge level — the one answer over all of a "
                     f"task's words, held to about {pair.merge.words} words",
             "prompts": _entries(_level_prompts(pair.merge))}]
        if key == "frames":
            groups.append(
                {"title": "Frame topics",
                 "note": "one pass over the merged analysis: the same frames "
                         "in topic order, under headings. Graded by nobody; "
                         "only its frame count is checked",
                 "prompts": [{"name": "Topic pass", "note": None, "same": None,
                              "turns": _turns(frame_topics_prompt(
                                  "⟨the merged frame analysis⟩"))}]})
        groups.append(
            {"title": "Budget",
             "note": "no model grades the length — the word count is "
                     "arithmetic, and an answer past the level's budget is "
                     "rewritten against this critique, the same at both "
                     "levels",
             "prompts": _entries([
                 ("Budget refinement", "asks a too-long answer rewritten "
                  "to budget",
                  _user(pair.word.budget_feedback.format(
                      words="⟨count⟩", budget="⟨the level's budget⟩")))])})
        sets.append({"key": key, "label": label, "groups": _dedup(groups)})
    return sets


@app.get("/prompts")
def prompts():
    return render_template("prompts.html", sets=_sets(),
                           default_set=DEFAULT_PROMPT_SET)
