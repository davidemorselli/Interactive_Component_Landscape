# LLM-as-a-judge grading: an answer held against a rubric of criteria.
#
# Idea from (Croxford et al, 2025), 10.1038/s41746-025-02005-2
#
# Original source code : https://git.doit.wisc.edu/smph-public/dom/uw-icu-data-science-lab-public/pdsqi-9/-/blob/main/02_LLM_as_a_Judge/pdsqi_create_prompt.ipynb?ref_type=heads
#
# One class for every level. The rubric and the two prompts that carry it are
# the level's (llm.pipeline.levels); the parsing, the averaging and the
# critique below read any rubric, because they read the criteria off the rubric
# itself.

import functools
import json
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from llm.prompts.refusals import declines

from .agent import Agent
from .refining import Refines

if TYPE_CHECKING:
    from .levels import Level

# The one criterion a refusal is graded on, at every level: each level's
# abstention prompt defines `warranted` and nothing else, and
# `build_rubric_feedback` reads a refusal by that name.
ABSTENTION_CRITERIA = frozenset({"warranted"})


@functools.lru_cache(maxsize=None)
def rubric_criteria(rubric_set: str) -> frozenset[str]:
    """The criteria a rubric defines, read off the tags that open its blocks
    ("<accurate>" on a line of its own; closing tags don't match — a backslash
    is not a word character). Cached: four rubrics, read on every grade."""
    return frozenset(re.findall(r"^<(\w+)>\s*$", rubric_set, re.MULTILINE))


def _parse_grades(answer: str | None,
                  criteria: frozenset[str]) -> tuple[dict, str] | None:
    """
    The (grades, reason) in the model's answer, or None for anything but a
    JSON object holding an integer grade for every criterion of the rubric.

    Checked against the rubric's own criteria, never against whatever keys
    came back: a model that corrupts its output can return valid JSON with
    mangled keys (a real run held a criterion named ": 1, "), and keys taken
    on trust get averaged into the grade. Surplus keys are dropped, not
    rejected — a grader that adds a key of its own has still graded.
    """
    try:
        grades = json.loads(answer or "")
    except json.JSONDecodeError:
        return None
    if not isinstance(grades, dict):
        return None
    reason = grades.pop("reason", "")  # the one value that is not a grade
    grades = {k: v for k, v in grades.items() if k in criteria}
    # `type` rather than isinstance: bool is a subclass of int, and a true/false
    # is not a grade. Every criterion must be there: a missing one is what a
    # mangled key leaves behind, and averaging over what survived would report
    # a grade the grader never gave.
    if set(grades) != set(criteria) or not all(type(g) is int for g in grades.values()):
        return None
    return grades, str(reason)


FEEDBACK_PROMPT = """The SUMMARY was graded against a rubric, out of 5 per criterion. It did not receive full marks on these criteria:
{grades}

The grader explains its grades as follows:
{reason}

Rewrite the SUMMARY so that it addresses this critique, making the least amount of changes, and still following all the instructions above. Return only the SUMMARY, do not add other text."""

# Worded for either level: the narrative writer declined over messages that do
# not discuss the query, the frame writer over messages that attest no
# recurrent frame. The closing sentence is the load-bearing one — without it a
# grader that judged the refusal wrong is an instruction to invent.
REDO_PROMPT = """You answered that the messages hold nothing to report, and wrote no answer. A grader read the same messages and judged that decision wrong, giving it {grade} out of 5.

The grader explains its grade as follows:
{reason}

Write now the answer you were asked for, over the messages that do bear on the query, following all the instructions above. Return only that answer, do not add other text. If, reading them again, you still find nothing to report, answer the same refusal line once more rather than inventing something: a refusal the grader disagreed with is a better answer than content the messages do not carry."""


def build_rubric_feedback(graded: pd.DataFrame) -> str:
    """The comment on an answer, from what `evaluate_summary` returned: the
    criteria it lost marks on, and the reason the grader gave. Empty when
    every criterion got the top grade — there is nothing to rewrite for."""
    grades = graded["grade"].drop(["average", "reason"], errors="ignore")
    reason = graded["grade"].get("reason", "")
    short = grades[grades < 5]  # 5 is the grade of a criterion with nothing left to fix
    if not len(short):
        return ""
    # A refusal holds nothing to rewrite: what it is asked for is the answer
    # it declined to write, not the least amount of changes to the refusal.
    if "warranted" in grades.index:  # the frame an abstention is graded in
        return REDO_PROMPT.format(grade=int(grades["warranted"]), reason=reason)
    lines = "\n".join(f"- {criterion}: {int(grade)}/5"
                      for criterion, grade in short.items())
    return FEEDBACK_PROMPT.format(grades=lines, reason=reason)


@dataclass
class RubricGrader(Refines, Agent):
    """A model that grades an answer against this level's rubric set, and
    rewrites the answer against its own critique. Configured as the Agent it
    extends, plus the level it grades for and a (low) grading temperature."""

    level: "Level" = field(kw_only=True)
    temperature: float = 0.0

    def evaluate_summary(self, token: str, summary: str,
                         sources: list[str]) -> pd.DataFrame | None:
        """
        Grade an answer over the given sources: a 'grade' column indexed by
        criterion, with an 'average' row (the mean of the grades) and a
        'reason' row (what the model says of its grading). An answer that
        declines — a NO NARRATIVE or NO FRAMES line — is not held against the
        rubric: its frame has the single criterion 'warranted', the grade of
        the decision to decline. None when the model did not answer with a
        JSON object of integer grades, which marks the answer invalid.
        """
        # An abstention has no content to hold against the rubric; what is
        # graded instead is the decision to decline.
        if declines(summary):
            prompt = self.level.abstention_prompt(token, sources)
            criteria = ABSTENTION_CRITERIA
        else:
            prompt = self.level.grading_prompt(token, summary, sources)
            criteria = self.level.criteria

        parsed = _parse_grades(self.ask([{"role": "user", "content": prompt}],
                                        temperature=self.temperature,
                                        response_format={"type": "json_object"}),
                               criteria)
        if parsed is None:
            return None
        grades, reason = parsed

        graded = pd.Series(grades, name="grade", dtype=float)
        graded["average"] = graded.mean()  # before the reason: the mean of the grades only
        graded["reason"] = reason
        return graded.rename_axis("criterion").to_frame()
