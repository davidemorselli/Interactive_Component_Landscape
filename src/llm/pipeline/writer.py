# The agent that writes an answer: the one model call the graders then judge.
#
# One class for every level. What a narrative of a word's tweets and a merge of
# a task's frame analyses have in common is the whole of this file; what they
# differ in — the prompt, the openings the answer must take, the room it is
# given — is the Level it is built for (llm.pipeline.levels).

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .agent import Agent

if TYPE_CHECKING:
    from .levels import Level


@dataclass
class Writer(Agent):
    """A model that writes this level's answer over this level's sources.
    Configured as the Agent it extends, plus the level it writes for and a
    (low) sampling temperature."""

    # Keyword-only and without a default: every writer is a writer of some
    # level, and there is no level it makes sense to fall back to.
    level: "Level" = field(kw_only=True)
    temperature: float = 0.1

    def write(self, query: str, sources: list[str]) -> str:
        """
        This level's answer over the given sources. The query names what was
        retrieved and is empty at the merge levels. An answer that does not
        open the way the level's prompt dictates is asked for again
        (Agent.ask's `valid`): the layout was dictated, the model simply did
        not follow it this time.
        """
        return self.ask(self.level.prompt(query, sources),
                        temperature=self.temperature,
                        valid=(None if self.level.openings is None else
                               lambda answer: self.level.accepts_answer(query, answer)))
