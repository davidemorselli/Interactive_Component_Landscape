# Token helpers — the one home of the agent{phase}_ speaker-token conventions.

import re

# A speaker token is 'agent<phase>_<name>'. The unprefixed 'agent_' form (no
# digit) also counts as a speaker: the corpus carries a few such junk words
# (agent_agitator, ...), and treating them as speakers keeps them out of the
# word vocabularies, as it always has.
_SPEAKER = re.compile(r"^agent(_|\d)")
_PHASE_PREFIX = re.compile(r"^agent(\d)_")
# The strip is laxer than the phase match on purpose: it also bares the
# unprefixed agent_ tokens.
BARE_NAME_PATTERN = r"^agent\d?_"


def is_speaker_token(word):
    """True when a vocabulary token is a speaker (agent) token."""
    return _SPEAKER.match(word) is not None


def token_phase(token):
    """The phase '1'..'4' of a speaker token's agent prefix, or None for
    tokens without one (non-speakers, and the unprefixed agent_ junk)."""
    match = _PHASE_PREFIX.match(token)
    return match.group(1) if match else None


def bare_name(token):
    """A speaker token without its agent prefix — the name the tweet table
    stores. Non-speaker tokens pass through unchanged."""
    return re.sub(BARE_NAME_PATTERN, "", token)


def phase_prefix(phase):
    """The token prefix of one phase's speakers."""
    return f"agent{phase}_"


def drawn_phases(phase):
    """The phases one drawn view spans: all four for 'all', else the one
    asked for ('1'..'4' or 'pooled')."""
    return ["1", "2", "3", "4"] if phase == "all" else [phase]


def is_valid_token(word):
    """True for a usable word: no mentions, urls, digits or short strings."""
    return (
        "@" not in word
        and "http" not in word
        and len(word) >= 4
        and not word.isdigit()
    )
