# 2D UMAP projection of the speaker landscape, with pickle cache

import warnings

import numpy as np
import umap.umap_ as umap

from config import TRAINING_DATA_PATH, UMAP_CACHE_PATH

from .pickle_cache import load_or_fit


def load_speakers_tokens(path=TRAINING_DATA_PATH, retain_threshold=5):
    """The speaker tokens with more than retain_threshold tweets, in
    first-seen order. Each line of the training file is one tweet:
    '<speaker_token> <word> <word> ...'."""
    counts = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split(" ")
            if parts and parts[1:]:
                counts[parts[0]] = counts.get(parts[0], 0) + 1
    return [speaker for speaker, n in counts.items() if n > retain_threshold]


def compute_umap_projection(emb, speaker_tokens, cache_path=UMAP_CACHE_PATH,
                            random_state=42, min_dist=0.01, n_neighbors=40):
    """Fit a 2D UMAP on the speakers' vectors — or load a cached fit, refit
    when the parameters or the speaker list change. Returns the aligned
    (reducer, speaker_tokens, coords)."""
    params = {"random_state": random_state, "min_dist": min_dist,
              "n_neighbors": n_neighbors, "n_speakers": len(speaker_tokens)}

    def fit():
        vectors = np.array([emb[token] for token in speaker_tokens])
        reducer = umap.UMAP(metric="cosine", min_dist=min_dist,
                            n_neighbors=n_neighbors, random_state=random_state)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            coords = reducer.fit_transform(vectors)
        return {"reducer": reducer, "speakers": speaker_tokens, "coords": coords}

    cache = load_or_fit(cache_path, params, fit)
    return cache["reducer"], cache["speakers"], cache["coords"]


def project_tokens(reducer, emb, tokens):
    """Project tokens into the fitted 2D speaker landscape: (n, 2)
    coordinates, empty (0, 2) for no tokens."""
    if not tokens:
        return np.empty((0, 2))
    return project_vectors(reducer, np.array([emb[token] for token in tokens]))


def project_vectors(reducer, vectors):
    """
    Project raw 250D vectors into the fitted 2D landscape — for vectors
    without a vocabulary token, e.g. a centroid. The reducer was fitted on
    raw speaker vectors, so the projection of a mean-centred vector is an
    approximation: where the direction lands, not an exact position.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return reducer.transform(vectors)
