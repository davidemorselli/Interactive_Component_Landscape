# Speaker selection: strong or extreme speakers of a component

from .decomposition import (get_strong_words, dominant_direction,
                            ranked_extremes)
from .token_helper import is_speaker_token, phase_prefix


def get_strong_speakers(ica_embedding, emb, component_index, strongest=None):
    """
    The speakers whose strongest component is the given one.

    A speaker is a strong speaker of component c when c is its strongest
    component, i.e. the one with the largest absolute value in its ICA
    representation, so each speaker is a strong speaker of exactly one
    component. Taken from the component's dominant direction, since one-sided
    components are the interpretable ones.
    """
    direction = dominant_direction(ica_embedding, emb, component_index, strongest)

    strong_words = get_strong_words(ica_embedding, emb, component_index, direction, strongest)
    return [word for word in strong_words if is_speaker_token(word)]


def get_extreme_speakers(ica_embedding, emb, component_index, k=50, phase=None, strongest=None):
    """
    The k speakers ranked at the end of the component axis, in its dominant
    direction — unlike strong speakers, a speaker can be extreme on several
    components. Ranked within `phase` when given (per-phase centroids stay
    comparable), over all speakers pooled when None. Selection is by rank,
    never magnitude: ICA values have no natural scale, "the k highest" is
    scale-free.
    """
    direction = dominant_direction(ica_embedding, emb, component_index, strongest)

    # A phase's speakers are exactly the tokens carrying its agent prefix.
    keep = (is_speaker_token if phase is None
            else lambda word: word.startswith(phase_prefix(phase)))

    return ranked_extremes(ica_embedding, emb, component_index, direction, keep, k)
