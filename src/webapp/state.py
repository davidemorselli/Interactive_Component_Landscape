# What boots once, before the first request: the landscape pipeline, the two
# locks every request shares, and the boot-time caches that keep /token and
# /suggested lock-free.

import threading
import time

import matplotlib
import numpy as np

# Before anything touches pyplot: the webapp renders from waitress worker
# threads, and the interactive backend a desktop picks (TkAgg) cannot draw off
# the main thread — the process dies on Tcl_AsyncDelete after a few renders.
# The notebooks import ica.landscape directly and keep their own backend.
matplotlib.use("Agg")

from ica.landscape import build_landscape, configure_fonts
from ica.token_helper import is_speaker_token, is_valid_token, token_phase
from rag.tweet_retrieval import bare_names, count_close_tweets


def _step(label, fn):
    t0 = time.time()
    result = fn()
    print(f"{label}: {time.time() - t0:.1f}s", flush=True)
    return result


configure_fonts()
landscape = build_landscape(progress=_step)
print(f"{len(landscape.selected_components)} retained components: "
      f"{landscape.selected_components}", flush=True)

# The interpretable pole of each retained component, resolved once so /token
# is a plain lock-free read — resolved per request it would write Landscape's
# plain-dict caches from whichever thread the request landed on.
COMPONENT_DIRECTIONS = {c: landscape.component(c, "direction")
                        for c in landscape.selected_components}

# The speakers the plot can put a dot on: UMAP was fitted on the speakers with
# more than a handful of tweets, and highlight_positions silently drops the
# rest. The export lists what the plot draws, and filters through this.
DRAWN_SPEAKERS = frozenset(landscape.fitted_speakers)

# Norm and vocabulary warm-ups, here so /token stays the lock-free read
# promised above; the numba warm-up pays the ~3s kernel compile of the first
# reducer.transform rather than letting whoever renders first pay it.
landscape.emb.fill_norms()
landscape.vocabulary(mean_centre=True, min_count=1)
landscape.project(landscape.fitted_speakers[:1])

# matplotlib and the Landscape caches — the nearest-word and candidate ones
# included — are not thread-safe: one render at a time
render_lock = threading.Lock()

# Serializes the corpus similarity scans. It is always taken inside
# render_lock, never the other way round.
search_lock = threading.Lock()

# (word, min_similarity, phase) -> how many tweets are that close to the word.
# The count itself is stored rather than whether it cleared a floor, so moving
# the "at least n tweets" box never costs a scan.
_coverage_cache = {}


def tweet_counts(words_by_phase, min_similarity):
    """
    {phase: {word: count}} — how many tweets of each phase are within
    `min_similarity` of each of its words. Takes every phase at once because
    the corpus product does; scans only what is not cached, and caches every
    phase of whatever it had to scan.
    """
    with search_lock:
        missing = {phase: [w for w in dict.fromkeys(words)
                           if (w, min_similarity, phase) not in _coverage_cache]
                   for phase, words in words_by_phase.items()}
        unknown = list(dict.fromkeys(w for words in missing.values() for w in words))
        if unknown:
            scanned = count_close_tweets(unknown, tuple(missing), min_similarity)
            for phase, counts in scanned.items():
                for word, count in zip(unknown, counts):
                    _coverage_cache[(word, min_similarity, phase)] = count
        return {phase: {w: _coverage_cache[(w, min_similarity, phase)] for w in words}
                for phase, words in words_by_phase.items()}


def covered_words(words, min_tweets, min_similarity, phase):
    """The words at least `min_tweets` tweets of `phase` are within
    `min_similarity` of."""
    counts = tweet_counts({phase: words}, min_similarity)[phase]
    return [w for w in words if counts[w] >= min_tweets]


def speaker_names(source, component, n_extreme, phase):
    """Bare speaker names to restrict retrieval to, or None for every speaker.

    Reads the Landscape's component caches — call under a lock that
    serializes them (the callers take search_lock)."""
    if source == "all":
        return None
    if source == "strong speakers":
        tokens = landscape.component(component, "strong_speakers")
    else:
        tokens = landscape.component(component, "extreme_speakers", n_extreme,
                                     phase=None if phase in ("all", "pooled") else phase)
    return bare_names(tokens)


# The tail sizes behind the auto buttons: the masks pick out the populations
# the extreme rankings run over — speaker tokens, valid non-speaker words —
# built once here, only ever read afterwards.
DIST_TOKENS = landscape.emb.index_to_key
DIST_SPEAKERS = np.array([is_speaker_token(t) for t in DIST_TOKENS])
DIST_WORDS = np.array([is_valid_token(t) and not is_speaker_token(t)
                       for t in DIST_TOKENS])
DIST_PHASES = np.array([token_phase(t) or "" for t in DIST_TOKENS])
