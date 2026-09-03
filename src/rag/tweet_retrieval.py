# Retrieval of the corpus tweets, by exact words or by meaning

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from ica.token_helper import bare_name

from .corpus import MODEL_NAME, clean, load

# torch and BLAS both take a thread per core and fight over them, which costs
# more than the encoder gains: it only ever encodes a query or a page of
# words here. Measured on 12 cores a search takes ~150 ms at the default and
# ~85 ms at two threads. Set before the encoder is built; the corpus encoding
# is a GPU job in encode_corpus, which does not import this module.
torch.set_num_threads(2)

_ROWS, _TEXTS, _VECTORS = load()
_ENCODER = SentenceTransformer(MODEL_NAME)


def _vector_ids(phase=None, speakers=None):
    """The distinct corpus vectors a phase/speaker restriction leaves."""
    rows = _ROWS
    if phase is not None:
        rows = rows[rows["phase"] == phase]
    if speakers is not None:
        rows = rows[rows["speaker"].isin(speakers)]
    return rows["vec_id"].unique()


def _similarities(queries):
    """
    (n_vectors, n_queries) cosine similarity of the corpus to each query.

    One encode and one matrix product for the whole list: in a batch of ten a
    query costs about a fifth of what it costs on its own, which is what makes
    scanning a page of words affordable.
    """
    vectors = _ENCODER.encode([clean(query) for query in queries],
                              normalize_embeddings=True)
    return _VECTORS @ vectors.T


def bare_names(tokens):
    """Speaker tokens ('agent1_name') -> the bare names the tweet table stores.

    The `speakers` argument of get_tweets_about is matched against those bare
    names, so speaker tokens coming from the ICA side have to be stripped first.
    """
    return {bare_name(token) for token in tokens}


def get_tweets_about(query, take_n=20, phase=None, speakers=None, min_similarity=0.3):
    """The tweets closest in meaning to the query — a DataFrame of 'tweet'
    and 'similarity', closest first, at most take_n above the cosine floor,
    optionally restricted to one phase and/or a set of speakers. Scoring is
    over distinct texts, so a tweet posted many times appears once."""
    vec_ids = _vector_ids(phase, speakers)
    similarities = _similarities([query])[vec_ids, 0]

    above = np.flatnonzero(similarities >= min_similarity)
    if take_n < len(above):
        cut = -np.partition(-similarities[above], take_n - 1)[take_n - 1]
        above = above[similarities[above] >= cut]
    ranked = above[np.argsort(-similarities[above], kind="stable")[:take_n]]
    return pd.DataFrame({"tweet": _TEXTS["tweet"].values[vec_ids[ranked]],
                         "similarity": similarities[ranked]})


# Queries per product. One product's result is (n_vectors, n_queries) float32 —
# about 1.7 MB a query on this corpus — so a long table is scanned in batches
# rather than in one allocation.
_SCAN_BATCH = 64


def count_close_tweets(queries, phases=(None,), min_similarity=0.3, speakers=None):
    """
    How many distinct tweets are within `min_similarity` of each query, as
    {phase: [count per query]} — a None phase counts the whole corpus. The
    corpus product is what costs and a phase is only a row selection on it,
    so one call answers every phase asked at once.
    """
    queries = list(queries)
    ids = {phase: _vector_ids(phase, speakers) for phase in phases}
    counts = {phase: [] for phase in phases}
    for start in range(0, len(queries), _SCAN_BATCH):
        # The floor is applied before the phase slice: the copy a slice makes
        # is then a byte a cell instead of four, and it is made once per phase.
        close = _similarities(queries[start:start + _SCAN_BATCH]) >= min_similarity
        for phase in phases:
            counts[phase] += close[ids[phase]].sum(axis=0).tolist()
    return counts

