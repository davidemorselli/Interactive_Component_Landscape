# Holding an answer to its word budget. The other graders ask a model what it
# thinks; this one grades in code — the budget is arithmetic — and is an agent
# only because the rewrite it asks for is a model call.

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from llm.prompts.refusals import declines

from .agent import Agent
from .refining import Refines

if TYPE_CHECKING:
    from .levels import Level

# How far past the budget an answer may go before it is over it. The prompts ask
# for "about" their number of words, and a writer that lands near it has done
# what was asked; only an answer visibly longer than the budget is one that
# ignored it.
TOLERANCE = 0.20

SUMMARY_FEEDBACK = """The SUMMARY is {words} words long. It was to be about {budget} words, and that budget was not respected.

Write a new SUMMARY of about {budget} words, saying what this one says in the words available, and still following all the instructions above. Return only the SUMMARY, do not add other text."""

# The same critique in the frames register: those prompts call their answer
# an ANALYSIS and never a SUMMARY ("Keep the whole analysis under about N
# words"), and a rewrite asked for under the wrong name invites prose in
# place of the FRAME layout. Which register a level rewrites in is the
# level's own field (llm.pipeline.levels, `budget_feedback`).
ANALYSIS_FEEDBACK = """The ANALYSIS is {words} words long. It was to be kept under about {budget} words, and that budget was not respected.

Write a new ANALYSIS of about {budget} words, saying what this one says in the words available, and still following all the instructions above. Return only the ANALYSIS, do not add other text."""


def word_count(summary: str) -> int:
    """How long an answer is, in words: runs of non-space, which is the count
    the page already prints under a narrative (index.js `showSummary`)."""
    return len(summary.split())


def build_budget_feedback(summary: str, budget: int, tolerance: float,
                          template: str = SUMMARY_FEEDBACK) -> str:
    """
    The comment on an answer that overran its budget: how long it is and how
    long it was to be, worded by `template` — the level's own register, the
    SUMMARY wording by default. Empty when it did not overrun — there is
    nothing to rewrite for, and that emptiness is what stops the refinement.

    An answer shorter than its budget is left alone: the budget is a ceiling on
    what the writer may spend, not a quota it must fill. A refusal is likewise
    left alone — it is the one line the prompt dictates, and its length is not
    the writer's to choose.
    """
    if declines(summary):
        return ""
    words = word_count(summary)
    if words <= round(budget * (1 + tolerance)):
        return ""
    return template.format(words=words, budget=budget)


@dataclass
class BudgetGrader(Refines, Agent):
    """A word budget, and a model that rewrites an answer which overran it.
    Configured as the Agent it extends, plus the level whose budget it holds
    an answer to and how far past it an answer may go. It has no grading
    temperature: it never asks a model what it thinks."""

    level: "Level" = field(kw_only=True)
    tolerance: float = TOLERANCE

    @property
    def budget(self) -> int:
        """What this level's prompt asks the writer for, so the answer is held
        to the number it was given."""
        return self.level.words

    def ceiling(self) -> int:
        """The longest an answer may be before it is over budget."""
        return round(self.budget * (1 + self.tolerance))

    def evaluate_summary(self, summary: str) -> int:
        """How long the answer is, in words. No model is asked: unlike the
        other graders' `evaluate_summary`, this one costs nothing and cannot
        fail."""
        return word_count(summary)

    def feedback(self, summary: str) -> str:
        """What the answer would be rewritten against, empty when it is
        within budget."""
        return build_budget_feedback(summary, self.budget, self.tolerance,
                                     self.level.budget_feedback)
