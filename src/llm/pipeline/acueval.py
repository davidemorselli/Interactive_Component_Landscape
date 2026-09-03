# ACUEval: grading an answer through the atomic content units (ACUs) it holds.
#
# ACUEval (Wan et al., Findings of ACL 2024, https://aclanthology.org/2024.findings-acl.597/)
# grades a summary in two structured steps instead of asking for one score:
# (1) decompose the generated summary into ACUs — "elementary information units,
# which no longer need to be further split" — and (2) verify each unit against
# the source.
#
# One class for every level. The decomposition wording and the verification
# wording are the level's (llm.pipeline.levels); the parsing and the
# scoring below are the paper's, and are the same whatever was decomposed.

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd
from openai.types.chat import ChatCompletionMessageParam

from llm.prompts.narrative import source_block
from llm.prompts.refusals import declines

from .agent import Agent
from .refining import Refines

if TYPE_CHECKING:
    from .levels import Level

FEEDBACK = """The SUMMARY is not consistent with the MESSAGES. The messages do not support the following claims:
{claims}

Rewrite the SUMMARY so that it no longer claims these things, making the least amount of changes, and still following all the instructions above. Return only the SUMMARY, do not add other text."""


def decomposition_messages(decomposition, answer: str) -> list[ChatCompletionMessageParam]:
    """The decomposition prompt as chat turns: one user/assistant pair per
    worked example of a level's decomposition, then the answer to break down.
    A module function rather than a method so the prompts page can show the
    turns without an agent to send them."""
    instruction, examples = decomposition
    messages: list[ChatCompletionMessageParam] = []
    for passage, facts in examples:
        messages.append({"role": "user", "content": instruction + passage})
        messages.append({"role": "assistant", "content": facts})
    messages.append({"role": "user", "content": instruction + answer})
    return messages


def _parse_acus(answer: str | None) -> list[str]:
    """
    The units listed in the model's answer, one per line. As in the authors'
    code, a line counts as a unit if it is bulleted ("- fact") or numbered
    ("1. fact"); anything else the model says around the list is dropped.
    """
    units = []
    for line in (answer or "").splitlines():
        line = line.strip()
        if line.startswith("- "):
            units.append(line[2:].strip())
        elif re.match(r"\d+\.\s", line):
            units.append(line.split(".", 1)[1].strip())
    return units


def build_acueval_feedback(verified: pd.DataFrame) -> str:
    """The comment on an answer, from the verified units: every claim the
    sources refute. Empty when they bear all of them out, and units left
    unjudged are not listed — only a No is a reason to rewrite."""
    refuted = [unit for unit, supported in verified["supported"].items()
               if supported is not None and not supported]
    if not refuted:
        return ""
    return FEEDBACK.format(claims="\n".join(f"- {unit}" for unit in refuted))


@dataclass
class ACUEvalGrader(Refines, Agent):
    """
    A model that grades an answer by decomposing it into atomic content units
    and verifying each against the sources it was written from, and that
    rewrites the answer from the units the sources refute. Configured as the
    Agent it extends, plus the level it grades for.
    """

    level: "Level" = field(kw_only=True)

    # Greedy by default, as the paper decodes for a reproducible split and
    # for its Yes/No verdicts.
    temperature: float = 0.0

    def get_atomic_content_unit_from_summary(self, summary: str) -> list[str]:
        """Break an answer into the atomic content units it puts forward, one
        elementary fact per item in the order the model listed them. Empty
        when the model answered with no list at all."""
        return _parse_acus(self.ask(
            decomposition_messages(self.level.decomposition, summary),
            temperature=self.temperature))

    def verify_atomic_units_using_tweets(self, units: list[str],
                                         sources: list[str]) -> pd.DataFrame:
        """Label each atomic content unit — one model call per unit — by
        whether the sources bear it out: a 'supported' column indexed by unit,
        True where they do and None where the model answered neither Yes nor
        No. `score` turns this into the ACUEval score."""
        def parse_yes_no(answer: str) -> bool | None:
            """
            True when the model answered Yes, False when it answered No, None
            when it answered neither, which leaves the unit unjudged rather
            than refuted.
            """
            said = re.search(r"\b(yes|no)\b", answer.lower())
            return said.group(1) == "yes" if said else None

        context = source_block(sources)
        supported = [parse_yes_no(self.ask(
            [{"role": "user",
              "content": self.level.verification.format(context=context, unit=unit)}],
            temperature=self.temperature))
            for unit in units]
        return pd.Series(supported, index=pd.Index(units, name="unit"),
                         name="supported").to_frame()

    def score(self, verified: pd.DataFrame) -> float:
        """The ACUEval score of an answer — the paper's final score, the share
        of its units the sources bear out, in [0, 1]. Units left unjudged are
        left out of the average, and an answer with no judged unit at all
        scores nan."""
        judged = verified["supported"].dropna()
        return float(judged.mean()) if len(judged) else float("nan")

    def evaluate_summary(self, summary: str,
                         sources: list[str]) -> tuple[pd.DataFrame, float]:
        """Grade an answer over the given sources end to end — decompose,
        verify, score — as (per-unit verdicts, ACUEval score). An answer that
        declines — a NO NARRATIVE or NO FRAMES line — puts no unit forward:
        its frame is empty and it scores nan, which reports it as unscored
        rather than wrong."""
        # An abstention makes no claim about the sources, so there is nothing
        # to verify and nothing to rewrite. Decomposing it would score a
        # correct refusal near zero, since the sentences it splits into are not
        # borne out by sources it says nothing about — and the low score would
        # then buy another round of refinement of an answer that is already
        # right.
        if declines(summary):
            units = []  # skips the decomposition call and every verification call
        else:
            units = self.get_atomic_content_unit_from_summary(summary)
        verified = self.verify_atomic_units_using_tweets(units, sources)
        return verified, self.score(verified)
