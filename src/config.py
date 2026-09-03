
# Everything sits flat in DATA_DIR: the embedding, the training data, the
# speaker labels, and the caches the code writes there itself. DATA_DIR is the
# only thing to configure

import os
from pathlib import Path

from dotenv import load_dotenv

# Read .env here, before DATA_DIR below is resolved: config is imported before
# every other module, so this is the one call that lets .env configure it.
load_dotenv()

_REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("DATA_DIR", _REPO_ROOT / "data"))

EMBEDDING_NAME = "2021_SA_vaccine-total_phase-annotated"
LABEL_COLUMN = "label_svm_word2vec_3agree"

# Inputs
EMBEDDING_PATH = DATA_DIR / f"{EMBEDDING_NAME}.emb"
TRAINING_DATA_PATH = DATA_DIR / f"{EMBEDDING_NAME}_training-data.txt"
SPEAKER_LABELS_CSV = DATA_DIR / "WORD2VEC-SPEAKER-LABELS.csv"

# Caches, written on first use
ICA_CACHE_PATH = DATA_DIR / "ica_embedding.pkl"
UMAP_CACHE_PATH = DATA_DIR / "umap_projection.pkl"

# Tweet vectorization, built by rag/encode_corpus (the three are only
# meaningful together)
TWEET_VECTORS_PATH = DATA_DIR / "tweet_vectors.npy"
TWEET_TEXTS_PATH = DATA_DIR / "tweet_vector_texts.csv"
TWEET_ROWS_PATH = DATA_DIR / "tweet_rows.csv"

DATA_DIR.mkdir(parents=True, exist_ok=True)
