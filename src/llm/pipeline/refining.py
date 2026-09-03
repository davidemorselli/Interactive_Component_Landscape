# Rewriting an answer against a grader's critique. All three graders do it
# the same way and differ only in the critique they hand it; everything
# per-level is read off the grader's Level.

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .levels import Level

# The temperature the answer was written at: rewriting is writing, not judging.
REFINE_TEMPERATURE = 0.1


class Refines:
    """
    Rewriting, for an agent that also grades.

    The rewrite replays the conversation the answer was written in — the
    writing prompt as the first turn, the answer as the model's own, the
    critique as what is said to it next — so the rules the answer was written
    under are the rules it is revised under.
    """

    if TYPE_CHECKING:
        # Supplied by the Agent class this mixin is combined with, and by the
        # Level field every one of them carries; stubs only, so they never
        # shadow the real attributes in the MRO.
        level: "Level"

        def ask(self, messages: list, temperature: float = 0.0,
                **options) -> str: ...

    def refine_summary(self, token: str, sources: list[str], summary: str,
                       feedback: str) -> str | None:
        """
        Rewrite an answer against this grader's critique. An empty critique
        returns the answer untouched, without a call; None marks a rewrite
        that does not open any of the ways this level's answers may, which is
        an invalid answer and is thrown away silently — the version before it
        stands.

        A rewrite may itself be a refusal: told its narrative is not grounded,
        concluding there is none to write is a compliant answer, not a failure.
        """
        if not feedback.strip():
            return summary
        revised = self.ask(
            self.level.prompt(token, sources)
            + [{"role": "assistant", "content": summary},
               {"role": "user", "content": feedback}],
            temperature=REFINE_TEMPERATURE)
        return self.level.accepts_revision(token, revised)
