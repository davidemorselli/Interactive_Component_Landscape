from gensim.models import KeyedVectors

from config import EMBEDDING_PATH


def load_embedding():
    """The speaker landscape embedding, from config.EMBEDDING_PATH."""
    return KeyedVectors.load(str(EMBEDDING_PATH))
