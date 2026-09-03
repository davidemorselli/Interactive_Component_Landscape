# The four levels this app writes and grades at, declared once: a narrative of
# one word's tweets and the merge of a task's narratives, a frame analysis and
# its merge. Every one is the same four agents making the same calls; what
# differs is data, and this table is where the data lives. (It used to be
# sixteen classes and three mixins, one per (level, role) cell — a fifth level
# is now a fifth entry below and nothing else.)

from dataclasses import dataclass
from typing import Callable, NamedTuple

from llm.prompts import frames, frames_merge, narrative, narrative_merge

from .acueval import ACUEvalGrader
from .agent import MAX_TOKENS
from .budget import ANALYSIS_FEEDBACK, BudgetGrader, SUMMARY_FEEDBACK
from .rubric import RubricGrader, rubric_criteria
from .writer import Writer

# The four roles a level is run by. Stored task settings name their model
# boxes with these keys — never rename them.
ROLES = {"summary": Writer, "acueval": ACUEvalGrader,
         "rubric": RubricGrader, "budget": BudgetGrader}


def _turns(build):
    """A prompt builder that answers one string, as a conversation of one user
    turn; the merge prompts answer their own turns and pass through as is."""
    return lambda query, sources: [{"role": "user",
                                    "content": build(query, sources)}]


@dataclass(frozen=True)
class Level:
    """
    One (prompt set, level) cell: everything the four agents need that is not
    the same at every level. Frozen read-only configuration, safe to share
    across the threads a run fans out over.
    """

    # What this level is called in the code, and what the pages call its answer.
    name: str
    label: str

    # --- writing ------------------------------------------------------------
    # The conversation that asks for this level's answer, given (query,
    # sources). The query is empty at the merge levels, whose sources stand
    # for a whole word list.
    prompt: Callable[[str, list[str]], list[dict]]

    # What a compliant answer opens with, given the query — or None where the
    # prompt dictates no layout. The counterpart of the opening the prompt
    # itself mandates: change them together.
    openings: Callable[[str], tuple[str, ...]] | None

    # How long the answer is asked to be, and held to by the budget grader.
    words: int

    # --- ACUEval ------------------------------------------------------------
    # The decomposition prompt: its instruction and worked examples.
    decomposition: tuple[str, list[tuple[str, str]]]

    # The Yes/No prompt one unit is verified against, naming this level's
    # sources. Formatted with {context} and {unit}.
    verification: str

    # --- the rubric grader --------------------------------------------------
    rubric: str
    # (query, answer, sources, rubric) -> the prompt that grades the answer.
    grading: Callable[[str, str, list[str], str], str]
    # (query, sources) -> the prompt that grades a refusal, on the one
    # criterion `warranted`.
    abstention: Callable[[str, list[str]], str]

    # --- the budget grader --------------------------------------------------
    # The critique an over-budget answer is rewritten against, worded in this
    # level's own register: SUMMARY at the narrative levels, ANALYSIS at the
    # frames levels, which never call their answer anything else.
    budget_feedback: str = SUMMARY_FEEDBACK

    # --- transport ----------------------------------------------------------
    # The ceiling one answer of this level is held to; only the frames merge
    # needs more than the default.
    max_tokens: int = MAX_TOKENS

    # What a FIRST answer must open with, where that is looser than what a
    # rewrite must. Defaults to `openings`; only the frames word level sets it
    # — a bad rewrite is discarded for free, but a first answer that keeps
    # missing the same punctuation loses the word altogether.
    first_openings: Callable[[str], tuple[str, ...]] | None = None

    # --- what the agents ask of it -----------------------------------------

    def accepts_answer(self, query: str, answer: str) -> bool:
        """Whether a first answer opens the way this level's prompt dictates."""
        openings = (self.first_openings or self.openings)
        return openings is None or answer.strip().upper().startswith(openings(query))

    def accepts_revision(self, query: str, revised: str) -> str | None:
        """The rewrite if it opens a way this level's answers may, else None —
        matched stripped and case-blind, so a capital letter never costs a
        rewrite."""
        if self.openings is None:
            return revised or None
        return revised if revised.strip().upper().startswith(
            self.openings(query)) else None

    def grading_prompt(self, query: str, answer: str, sources: list[str]) -> str:
        """The prompt that grades one answer of this level against its own
        rubric."""
        return self.grading(query, answer, sources, self.rubric)

    def abstention_prompt(self, query: str, sources: list[str]) -> str:
        """The prompt that grades a refusal at this level: whether there was
        anything in these sources to report."""
        return self.abstention(query, sources)

    @property
    def criteria(self) -> frozenset[str]:
        """The criteria a grader's answer is checked against — the rubric's
        own, so the rubric stays the one place a criterion is declared."""
        return rubric_criteria(self.rubric)

    def agent(self, role: str, **config):
        """One of this level's four agents, by role. Every role shares the
        level's token ceiling: a grader that rewrites what the writer wrote
        needs the room the writer had."""
        return ROLES[role](level=self, max_tokens=self.max_tokens, **config)

    def writer(self, **config) -> Writer:
        """This level's writer — the one role called on its own often enough
        to be worth naming."""
        return self.agent("summary", **config)


# --- The narrative levels ---------------------------------------------------

NARRATIVE = Level(
    name="narrative",
    label="Narrative",
    prompt=_turns(narrative.build_prompt),
    openings=lambda query: (f"AROUND {query.upper()}, THE MESSAGES",
                            "NO NARRATIVE:"),
    words=narrative.SUMMARY_WORDS,
    decomposition=(narrative.DECOMPOSITION_INSTRUCTION,
                   narrative.DECOMPOSITION_EXAMPLES),
    verification=narrative.VERIFICATION_PROMPT,
    rubric=narrative.RUBRIC,
    grading=narrative.grading_prompt,
    abstention=narrative.abstention_prompt,
)

NARRATIVE_MERGE = Level(
    name="narrative_merge",
    label="Summary of summaries",
    # The original PONS prompt, a system persona and one user turn; it names
    # the query nowhere — the summaries stand for the word list.
    prompt=lambda _query, summaries: narrative_merge.merge_prompt(summaries),
    openings=None,  # the PONS prompt dictates no opening
    words=narrative_merge.MERGE_SUMMARY_WORDS,
    # The tweet level's: what a unit of a narrative looks like does not depend
    # on what the narrative was written from.
    decomposition=(narrative.DECOMPOSITION_INSTRUCTION,
                   narrative.DECOMPOSITION_EXAMPLES),
    verification=narrative_merge.VERIFICATION_PROMPT,
    rubric=narrative_merge.RUBRIC,
    grading=lambda _query, summary, sources, rubric:
        narrative_merge.grading_prompt(summary, sources, rubric),
    # All but unreachable — narratives that all say something leave no honest
    # reason to decline — but the cell is filled: the word level's stands
    # behind it rather than nothing at all.
    abstention=narrative.abstention_prompt,
)

# --- The frames levels ------------------------------------------------------

FRAMES = Level(
    name="frames",
    label="Frame analysis",
    prompt=_turns(frames.build_prompt),
    openings=lambda _query: ("FRAME:", "NO FRAMES:"),
    # Measured over stored runs, the colon is most of what a first answer
    # misses ("FRAME 1:", "NO FRAMES."); opening on the bare prefix still
    # rejects the answers actually wrong — JSON, fenced blocks, the prompt's
    # own example handed back.
    first_openings=lambda _query: ("FRAME", "NO FRAMES"),
    words=frames.FRAME_SUMMARY_WORDS,
    decomposition=(frames.DECOMPOSITION_INSTRUCTION,
                   frames.DECOMPOSITION_EXAMPLES),
    verification=frames.VERIFICATION_PROMPT,
    rubric=frames.RUBRIC,
    # No query at either frames level: the writer was told to ignore the
    # retrieval term, so the grader is never shown it either.
    grading=lambda _query, analysis, tweets, rubric:
        frames.grading_prompt(analysis, tweets, rubric),
    abstention=lambda _query, tweets:
        frames.abstention_prompt(tweets, frames.FRAME_ABSTENTION_RUBRIC),
    budget_feedback=ANALYSIS_FEEDBACK,
)

FRAMES_MERGE = Level(
    name="frames_merge",
    label="Merged analysis",
    prompt=lambda _query, analyses: frames_merge.merge_prompt(analyses),
    openings=None,  # the merge prompt dictates no opening
    words=frames_merge.FRAME_MERGE_WORDS,
    decomposition=(frames.DECOMPOSITION_INSTRUCTION,
                   frames.DECOMPOSITION_EXAMPLES),
    verification=frames_merge.VERIFICATION_PROMPT,
    rubric=frames_merge.RUBRIC,
    grading=lambda _query, analysis, analyses, rubric:
        frames_merge.grading_prompt(analysis, analyses, rubric),
    # The word level's, as at the narrative merge — declining here is all but
    # unreachable. (While the levels were classes, the NARRATIVE abstention
    # prompt stood here unnoticed; a table has one cell for it.)
    abstention=lambda _query, analyses:
        frames.abstention_prompt(analyses, frames.FRAME_ABSTENTION_RUBRIC),
    budget_feedback=ANALYSIS_FEEDBACK,
    # The one answer legitimately longer than the rest: 500 words carrying
    # the structure of every frame it folded in.
    max_tokens=8192,
)


class PromptSet(NamedTuple):
    """One set of prompts at both its levels: the word level writes one answer
    per word of a task, the merge level the one answer over all of them."""

    word: Level
    merge: Level


# Keyed as the pages key their tab bar (params.PROMPT_LABELS), and as a task's
# endpoint names the set it was created under.
LEVELS = {"narratives": PromptSet(word=NARRATIVE, merge=NARRATIVE_MERGE),
          "frames": PromptSet(word=FRAMES, merge=FRAMES_MERGE)}
