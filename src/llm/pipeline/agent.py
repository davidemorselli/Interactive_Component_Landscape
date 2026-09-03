# The one way this project talks to a model: every agent extends Agent and
# differs only in what it sends — which is its Level (llm.pipeline.levels).

import os
import threading
import time
from dataclasses import dataclass

from openai import APIError, OpenAI

import config  # noqa: F401 — importing config loads .env for the API key

BASE_URL = "https://openrouter.ai/api/v1"

# Roughly one call in ten comes back as an error payload; without retries a
# multi-call job like an ACUEval grade rarely survives.
ATTEMPTS = 4
BACKOFF = 1.5  # seconds before the second try, doubled before each one after
# TIMEOUT counts silence between bytes, so on its own it bounds nothing;
# DEADLINE bounds the whole call however busy the line looks.
TIMEOUT = 180
DEADLINE = 300

# The temperatures a call is retried at after its answer ran to the token
# ceiling. A ceiling hit is a repetition loop, and a loop is a fixed point of
# the decoding — the same try at the same temperature loops again — so each
# try after a ceiling is hotter than the last (the caller's own temperature is
# the first).
LOOP_TEMPERATURES = (0.5, 0.8, 1.0)

# The default ceiling on one answer, in tokens. Not a budget (the word budgets
# hold answers to a length) but a stop for the failure neither clock catches:
# a repetition loop streams steadily, so only DEADLINE would end it — 300s and
# tens of thousands of tokens later. Set from measurement, reasoning included:
# the largest healthy answer generated 5,010 tokens, the loops ran to 65,536+.
# 6144 sits between — above every answer worth having, below every one worth
# cutting.
MAX_TOKENS = 6144


class Attempts:
    """
    What one run's calls cost in tries, counted by model.

    One collector is shared by every agent of a run, across the threads the
    run fans out over, so the counting is locked. A run given none is simply
    not counted (Agent.attempts defaults to None).
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._models = {}

    def record(self, model: str, tries: int, failed: bool) -> None:
        """One finished call: which model was asked, how many tries it took,
        and whether it ended with no answer at all."""
        with self._lock:
            row = self._models.setdefault(
                model, {"calls": 0, "tries": 0, "failures": 0})
            row["calls"] += 1
            row["tries"] += tries
            row["failures"] += int(failed)

    def report(self) -> dict:
        """{model: {calls, tries, failures}}, plain enough to store as JSON."""
        with self._lock:
            return {model: dict(row) for model, row in self._models.items()}


class AgentError(RuntimeError):
    """A failure whose message is written for the user — a missing API key, a
    model past its output ceiling. The webapp answers it as a 503 with the
    message as it is, where anything else gets a traceback."""


class Truncated(AgentError):
    """An answer that ran to its token ceiling, so there is none to read.
    `ask` treats it as neither failure nor answer but a reason to try again
    hotter (LOOP_TEMPERATURES); an AgentError still, for the run with no
    tries left."""


@dataclass
class Agent:
    """A model behind OpenRouter: a model identifier as OpenRouter names it
    (e.g. 'openai/gpt-oss-20b:nitro')."""

    model: str

    # Where this agent's calls are counted, when a run is counting them —
    # shared by every agent of the run.
    attempts: Attempts | None = None

    # The ceiling this agent's answers are held to. A field, so a level whose
    # answers are longer names its own (Level.max_tokens).
    max_tokens: int = MAX_TOKENS

    def ask(self, messages: list, temperature: float = 0.0,
            valid=None, **options) -> str:
        """
        What the model answers to a conversation of chat turns; `options` go
        to the API as they are. The answer comes back stripped; a call still
        answering nothing after ATTEMPTS tries raises, as does one cut off by
        the output ceiling with no tries left.

        No call is waited on longer than DEADLINE, so the answer comes back —
        or stops being waited for — inside a bounded time however the provider
        behaves; that bound is what lets a caller fan work out. A try that ran
        to the token ceiling is retried hotter (LOOP_TEMPERATURES), so
        `temperature` names where the retrying starts.

        `valid` is a predicate naming what the caller accepts as an answer at
        all — for the prompts that dictate how their answer opens. An answer
        it rejects is retried like an empty one.
        """
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            raise AgentError("set OPENROUTER_API_KEY in the environment or in .env")
        # setdefault, not an argument: a caller naming its own ceiling keeps it.
        options.setdefault("max_tokens", self.max_tokens)
        why = "no attempt was made"
        tries, ceilings = 0, 0
        for attempt in range(ATTEMPTS):
            tries = attempt + 1
            if attempt:
                time.sleep(BACKOFF * 2 ** (attempt - 1))
            # One attempt is one worker thread, and the connection never
            # leaves it: past DEADLINE the attempt is abandoned where it is,
            # never interrupted. Closing its socket from out here (a timer
            # once did) frees the descriptor number while the worker's TLS
            # stack still holds it; the number is recycled to whatever file
            # another thread opens next, and the TLS stack writes that file
            # as though it were the wire — which is how tasks.db lost its
            # header to an encrypted TLS alert. An abandoned worker ends on
            # its own; the daemon flag keeps one that never does from holding
            # up interpreter shutdown.
            outcome = {}
            done = threading.Event()

            # outcome and done are bound as defaults: an abandoned worker
            # finishing late must land in its own attempt's slots.
            def call(outcome=outcome, done=done, temperature=temperature):
                # max_retries=0: the retrying is this loop's — the client's
                # own silent three-tries would break the ATTEMPTS accounting.
                client = OpenAI(base_url=BASE_URL, api_key=key,
                                timeout=TIMEOUT, max_retries=0)
                try:
                    outcome["response"] = client.chat.completions.create(
                        model=self.model, temperature=temperature,
                        messages=messages, **options)
                except Exception as error:
                    outcome["error"] = error
                finally:
                    client.close()  # a client left open holds its connections
                    done.set()

            threading.Thread(target=call, daemon=True).start()
            if not done.wait(DEADLINE):
                why = f"no answer within {DEADLINE}s; the attempt was abandoned"
                continue
            error = outcome.get("error")
            if isinstance(error, APIError):  # refused, dropped, rate-limited
                why = f"{type(error).__name__}: {error}"
                continue
            if error is not None:  # not the provider answering badly: our bug
                self._count(tries, failed=True)
                raise error
            try:
                answer = self._answer(outcome["response"])
            except Truncated as cut:
                # Not an answer, and not a failure while there are tries left:
                # what hit the ceiling is a loop, and a loop comes good on a
                # hotter try. The last message is kept for `why`.
                ceilings += 1
                why = f"{cut} (tries at the ceiling: {ceilings})"
                temperature = LOOP_TEMPERATURES[
                    min(ceilings, len(LOOP_TEMPERATURES)) - 1]
                continue
            except Exception:  # our bug, or a provider answering nonsense
                self._count(tries, failed=True)
                raise
            if answer is not None and valid is not None and not valid(answer):
                why = "the answer did not open the way the prompt dictates"
                continue
            if answer is not None:
                self._count(tries, failed=False)
                return answer
            why = "the provider answered with no content"
        self._count(tries, failed=True)
        raise RuntimeError(f"{self.model} gave no answer in {ATTEMPTS} tries — {why}")

    def _count(self, tries: int, failed: bool) -> None:
        """One finished call, told to the run's collector if it has one."""
        if self.attempts is not None:
            self.attempts.record(self.model, tries, failed)

    @staticmethod
    def _answer(response) -> str | None:
        """
        The text of one response; None where there is none to read and the
        call is worth making again (an error payload without choices, or an
        empty content). A response cut off by the output ceiling raises
        Truncated instead — since MAX_TOKENS the ceiling sits well above any
        real answer, so what reached it is a repetition loop.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return None
        choice = choices[0]
        if choice.finish_reason == "length":
            raise Truncated(
                "the answer ran to its token ceiling with none of it ended: "
                "past MAX_TOKENS an answer has generally stopped saying "
                "anything — see the ceiling on the agent that asked")
        return (choice.message.content or "").strip() or None
