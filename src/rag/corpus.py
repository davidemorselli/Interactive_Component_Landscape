# The tweet corpus as the retrieval side reads it: the text conventions the
# encoder needs, and the loader of the built vectorization. The one-off GPU job
# that writes those files lives in encode_corpus/build_tweet_vectors.py.

import re

import numpy as np
import pandas as pd

from config import (TRAINING_DATA_PATH, TWEET_ROWS_PATH, TWEET_TEXTS_PATH,
                    TWEET_VECTORS_PATH)
from ica.token_helper import BARE_NAME_PATTERN

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

# The corpus is lowercased and joins n-grams with '_', so a link reads as
# 'httpstcozhhip7z2q2', sometimes glued to the previous word.
LINK = re.compile(r"_?httpstco\w*")
MENTION = re.compile(r"@\w+")


def clean(text):
    """Unjoin the corpus n-grams so the sentence encoder reads plain text.

    Apply to queries too.
    """
    return text.replace("_", " ")


def _relink(text, repl):
    return " ".join(LINK.sub(repl, text).split())


def strip_links(text):
    """Delete the t.co codes."""
    return _relink(text, " ")


def placeholder_links(text):
    """Replace the t.co codes by <URL_LINK>."""
    return _relink(text, " <URL_LINK> ")


def is_contentless(text):
    """True for tweets that are nothing but mentions and links."""
    return not MENTION.sub(" ", strip_links(text)).split()


def load_corpus_rows(path=TRAINING_DATA_PATH):
    """Read the corpus into one row per tweet, speaker token split off.

    Each line is '<speaker_token> <word> ...' Phase comes from the token prefix.
    """
    speakers, texts = [], []
    with open(path) as f:
        for line in f:
            tokens = line.strip().split(" ")
            if len(tokens) > 1:
                speakers.append(tokens[0])
                texts.append(" ".join(tokens[1:]))

    rows = pd.DataFrame({"phase_agent": speakers, "tweet": texts})
    rows["phase"] = rows["phase_agent"].str.extract(r"^agent(\d)")
    # Vectorized ica.token_helper.bare_name — the same pattern, per Series.
    rows["speaker"] = rows["phase_agent"].str.replace(BARE_NAME_PATTERN, "", regex=True)
    return rows


def load():
    """Load a built vectorization as (rows, texts, vectors).

    rows['vec_id'] indexes both texts and vectors; vectors are float32.
    """
    rows = pd.read_csv(TWEET_ROWS_PATH)
    texts = pd.read_csv(TWEET_TEXTS_PATH)
    vectors = np.load(TWEET_VECTORS_PATH).astype("float32")
    return rows, texts, vectors
