# Centroid computation: average a set of token vectors into a single unit-norm
# direction, with optional unit-normalization and vocabulary mean-centring.

import functools
import numpy as np


@functools.lru_cache(maxsize=None)
def vocabulary_mean(emb, normed=True):
    """The mean of all vocabulary vectors (the shared corpus direction), over
    the unit-normalized vectors when `normed` — which should match the
    unit_norm of compute_centroid, so the mean is subtracted from vectors of
    the same scale it was computed on."""
    vectors = emb.get_normed_vectors() if normed else emb.vectors
    return vectors.mean(axis=0)


def preprocessed_vectors(emb, tokens, mean_centre=True, unit_norm=True):
    """Token vectors after the centroid's preprocessing steps (see
    compute_centroid). Shared with the word clustering, so both always work
    in the same space."""
    # Look up the raw token vectors.
    vectors = np.array([emb.get_vector(token) for token in tokens])

    # Optional step 1: unit-normalize each vector so every token weighs the same.
    if unit_norm:
        vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

    # Optional step 2: mean-centre to remove the shared corpus direction,
    # then renormalize each centred vector back to unit length.
    if mean_centre:
        vectors = vectors - vocabulary_mean(emb, normed=unit_norm)
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors


# Centroid in the original 250D space

def compute_centroid(emb, tokens, mean_centre=True, unit_norm=True):
    """
    The (centroid, concentration) of a set of tokens in the embedding space.
    Unit normalization makes every token weigh the same; mean-centring removes
    the shared corpus direction. The centroid comes back unit-norm, so dot
    products with unit vectors are cosine similarities. The concentration is
    the mean resultant length of directional statistics, in [0, 1]: near 1
    the tokens share a genuine direction, near 0 the centroid is mostly noise.
    """
    vectors = preprocessed_vectors(emb, tokens, mean_centre=mean_centre, unit_norm=unit_norm)

    # Concentration on unit copies, so it stays a mean resultant length in
    # [0, 1] even when unit_norm=False and mean_centre=False leave raw vectors.
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    concentration = np.linalg.norm(unit.mean(axis=0))

    # Average and renormalize the centroid to unit length.
    centroid = vectors.mean(axis=0)
    return centroid / np.linalg.norm(centroid), concentration
