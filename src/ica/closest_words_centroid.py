# Closest words to a centroid: build the content-word vocabulary matrix, rank
# its words by cosine similarity to a centroid, and group the ranked words

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage

from .centroid_computation import preprocessed_vectors, vocabulary_mean
from .token_helper import is_valid_token, is_speaker_token


def content_vocabulary(emb, mean_centre=True, min_count=100, unit_norm=True):
    """
    The (words, matrix) of valid, frequent, non-speaker content word vectors.

    unit_norm must match the unit_norm the centroid was computed with — it
    decides both the vectors and which vocabulary mean is centred on — or the
    two are centred on different origins and the cosine ranking is against a
    shifted cloud.
    """
    words = [word for word in emb.index_to_key
             if is_valid_token(word) and not is_speaker_token(word)
             and emb.get_vecattr(word, "count") >= min_count]
    matrix = np.array([emb.get_vector(word, norm=unit_norm) for word in words])
    if mean_centre:
        # normed must match, as compute_centroid does it: the mean of the unit
        # vectors is not the mean of the raw ones.
        matrix = matrix - vocabulary_mean(emb, normed=unit_norm)
        matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.array(words), matrix


def nearest_words(centroid, words, matrix, topn=20):
    """The topn words closest to the centroid by cosine similarity, as a
    DataFrame of 'word' and 'similarity', closest first."""
    similarities = matrix @ centroid
    ranked = np.argsort(similarities)[::-1][:topn]
    return pd.DataFrame({"word": words[ranked], "similarity": similarities[ranked]})


def average_similarity(emb, words, mean_centre=True, unit_norm=True):
    """The average pairwise cosine similarity of a set of words, in the same
    preprocessed space cluster_words groups them in; None for fewer than two
    words."""
    if len(words) < 2:
        return None
    vectors = preprocessed_vectors(emb, words, mean_centre=mean_centre, unit_norm=unit_norm)
    # Cosine regardless of the preprocessing kept the vectors unit or not,
    # as the clustering metric takes them.
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = vectors @ vectors.T
    n = len(words)
    return float((similarities.sum() - n) / (n * (n - 1)))


def cluster_words(emb, words, threshold, mean_centre=True, unit_norm=True):
    """
    Group words into clusters of similar words: average-linkage agglomerative
    clustering, merging while the average pairwise cosine similarity clears
    the threshold. Deterministic and order-independent, in the same space the
    centroid search preprocesses its vocabulary. Clusters come in the order
    of their best-ranked member, members in the order given; no words is no
    clusters, never one empty cluster.
    """
    if len(words) < 2:
        return [[word] for word in words]
    merges = linkage(preprocessed_vectors(emb, words, mean_centre=mean_centre,
                                          unit_norm=unit_norm),
                     method="average", metric="cosine")
    labels = fcluster(merges, t=1 - threshold, criterion="distance")
    clusters = {}  # insertion-ordered: first-member order is rank order
    for word, label in zip(words, labels):
        clusters.setdefault(label, []).append(word)
    return list(clusters.values())