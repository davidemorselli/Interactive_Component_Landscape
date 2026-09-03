# Build the sentence-transformer vectorization of the tweet corpus.
#
# A one-off job, run on Colab for the GPU (see Encode_corpus_colab.ipynb) or
# standalone: python build_tweet_vectors.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# For standalone runs only: the web app and the notebooks already have src/
# on the path.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import TWEET_ROWS_PATH, TWEET_TEXTS_PATH, TWEET_VECTORS_PATH
from rag.corpus import (MODEL_NAME, clean, is_contentless, load,  # noqa: F401
                        load_corpus_rows, placeholder_links, strip_links)

# `load` is re-exported: the Colab notebook imports build and load from here.


def encode_texts(texts, model_name=MODEL_NAME):
    """Encode texts to float16 unit vectors, on the GPU when there is one.

    Vectors are unit-normalized, making a dot product a cosine similarity.
    fp16 doubles throughput on a tensor-core GPU (T4 and later)
    """
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(model_name, device=device)
    batch_size = 64

    if device == "cuda":
        batch_size = 256
        if torch.cuda.get_device_capability(0) >= (7, 0):
            model = model.half()
        print(f"encoding on {torch.cuda.get_device_name(0)}", flush=True)
    else:
        print("encoding on CPU", flush=True)

    vectors = model.encode([clean(t) for t in texts],
                           normalize_embeddings=True,
                           batch_size=batch_size,
                           show_progress_bar=True)
    return vectors.astype("float16")


def build():
    """Vectorize the corpus and write the vectors, texts and row table.

    The three files are only meaningful together.
    """
    rows = load_corpus_rows()

    # Discard contentless tweets
    rows = rows[~rows["tweet"].map(is_contentless)].reset_index(drop=True)

    # Deduplicate tweets which vary by only their links
    rows["canonical"] = rows["tweet"].map(strip_links)
    groups = rows.drop_duplicates("canonical").reset_index(drop=True)
    texts = groups["tweet"].map(placeholder_links).to_frame("tweet")

    vectors = encode_texts(groups["canonical"].tolist())

    # Pointer into the vector matrix; survives filtering and sorting.
    rows["vec_id"] = rows["canonical"].map(
        pd.Series(groups.index, index=groups["canonical"].values))

    np.save(TWEET_VECTORS_PATH, vectors)
    texts.to_csv(TWEET_TEXTS_PATH, index=False)
    rows.drop(columns=["tweet", "canonical"]).to_csv(TWEET_ROWS_PATH, index=False)

    return rows, texts, vectors


if __name__ == "__main__":
    build()
