import numpy as np
from sklearn.decomposition import FastICA

from config import ICA_CACHE_PATH

from .pickle_cache import load_or_fit
from .token_helper import is_valid_token, is_speaker_token

ICA_RANDOM_STATE = 1234
ICA_MAX_ITER = 2000


def load_or_compute_ica_embeddings(word_vectors, cache_path=ICA_CACHE_PATH,
                                   random_state=ICA_RANDOM_STATE, max_iter=ICA_MAX_ITER):
    """Fit FastICA on word_vectors — or load a cached fit made with the same
    parameters on a matrix of the same shape."""
    params = {"random_state": random_state, "max_iter": max_iter,
              "shape": tuple(word_vectors.shape)}

    def fit():
        ica = FastICA(random_state=random_state, max_iter=max_iter)
        return {"embedding": ica.fit_transform(word_vectors)}

    return load_or_fit(cache_path, params, fit)["embedding"]


def strongest_components(ica_embedding):
    """For every word, the component it is strongest on and the sign it has
    there — (components, signs) arrays aligned with emb.index_to_key."""
    components = np.argmax(np.abs(ica_embedding), axis=1)
    signs = np.where(ica_embedding[np.arange(len(ica_embedding)), components] >= 0, '+', '-')
    return components, signs


def get_strong_words(ica_embedding, emb, component_index, direction, strongest=None):
    """
    The strong words (speaker tokens included) of one component and direction.

    A word is a strong word of component c in direction dir when c is the
    word's strongest component, i.e. the one with the largest absolute value
    in its ICA representation, and the sign of that value matches dir (Musil
    and Mareček, 2024). Each word is therefore a strong word of exactly one
    component/direction, unlike extreme words which are ranked per component.
    """
    components, signs = strongest if strongest is not None else strongest_components(ica_embedding)

    strong_word_indices = np.where((components == component_index) & (signs == direction))[0]
    strong_word_indices = [i for i in strong_word_indices if is_valid_token(emb.index_to_key[i])]

    return [emb.index_to_key[i] for i in strong_word_indices]


def take_extremes(ranked, k, direction):
    """The k most extreme entries of a ranking sorted by descending component
    value: the top of the list for '+', the bottom for '-'.

    '-' keeps the ranking's order, most extreme LAST — asymmetric with '+',
    but visible in export row order, so it stays as it has always been.
    """
    if direction == '+':
        return ranked[:k]
    if direction == '-':
        return ranked[-k:] if k > 0 else []
    raise ValueError("Direction must be '+' or '-'")


def ranked_extremes(ica_embedding, emb, component_index, direction, keep, k):
    """The k tokens `keep` admits, ranked at the `direction` end of the
    component axis — the shared skeleton of the extreme words and the extreme
    speakers (take_extremes' order either way)."""
    ranked = np.argsort(ica_embedding[:, component_index])[::-1]
    indices = [i for i in ranked if keep(emb.index_to_key[i])]
    return [emb.index_to_key[i] for i in take_extremes(indices, k, direction)]


def get_extreme_words(ica_embedding, emb, component_index, direction, k=10,
                      min_count=None):
    """
    The k extreme words of one component and direction.

    The extreme words of a component are the k words with the highest
    (direction +) or lowest (direction -) value on that component, i.e. the
    words ranked at the ends of the component axis (Musil and Mareček, 2024).
    Unlike strong words, extreme words are ranked per component, so a word can
    be extreme on several components, and speaker tokens are excluded here.

    `min_count` drops the words the corpus writes fewer times than that
    before the k are taken, so the floor costs no labels: the ranking simply
    goes further down for its k.
    """
    def keep(word):
        return (is_valid_token(word) and not is_speaker_token(word)
                and (min_count is None
                     or emb.get_vecattr(word, "count") >= min_count))

    return ranked_extremes(ica_embedding, emb, component_index, direction, keep, k)


def tail_count(values, direction, tau=3.0):
    """
    How many of one family's component values lie beyond `tau` bulk σ on the
    dominant side — the k that family supports at that strictness.

    The scale is a robust one — 1.4826 × MAD around the median, the spread of
    the near-Gaussian bulk rather than of the whole sample — so the heavy tail
    the count is there to find does not inflate the σ it is cut at.
    """
    if not len(values):
        return 0
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    sigma = 1.4826 * mad if mad > 0 else (values.std() or 1.0)
    sign = 1 if direction == "+" else -1
    return int(((values - median) * sign > tau * sigma).sum())


# Retained components

def get_components_with_strong_words(ica_embedding, emb, threshold=0.9, strong_word_ratio_threshold=0.01):
    """The retained components: those whose strong words are at least
    `strong_word_ratio_threshold` of the vocabulary, with at least
    `threshold` of them on one side of zero (the one-sided, interpretable
    ones)."""
    selected_components = []
    total_words = len(emb.key_to_index)
    strongest = strongest_components(ica_embedding)

    for i in range(ica_embedding.shape[1]):
        # A strong word's sign on its component is its direction, so the two
        # list lengths are the side counts.
        n_plus = len(get_strong_words(ica_embedding, emb, i, '+', strongest))
        n_minus = len(get_strong_words(ica_embedding, emb, i, '-', strongest))
        n_strong_words = n_plus + n_minus
        if n_strong_words / total_words < strong_word_ratio_threshold:
            continue
        if max(n_plus, n_minus) / n_strong_words >= threshold:
            selected_components.append(i)

    return selected_components

# Dominant direction of a component

def dominant_direction(ica_embedding, emb, component_index, strongest=None):
    """The direction ('+' or '-') holding most of a component's strong words
    — its interpretable pole."""
    if strongest is None:
        strongest = strongest_components(ica_embedding)

    n_plus = len(get_strong_words(ica_embedding, emb, component_index, '+', strongest))
    n_minus = len(get_strong_words(ica_embedding, emb, component_index, '-', strongest))
    return '+' if n_plus >= n_minus else '-'