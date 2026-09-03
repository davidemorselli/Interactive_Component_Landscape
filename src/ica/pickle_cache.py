# One params-keyed pickle cache, shared by the ICA and UMAP fits.

import os
import pickle


def load_or_fit(cache_path, params, fit):
    """
    The cached payload dict when its stored params match, else fit() anew.

    `fit` returns the payload dict to store; `params` is stored alongside it
    under a 'params' key, and any mismatch — a changed parameter, a different
    corpus — refits and overwrites. The file holds the payload dict itself,
    so caches written before this helper existed stay readable.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            if cache.get("params") == params:
                return cache
        except Exception:  # foreign, old-format or half-written: fit over it
            pass
    payload = fit() | {"params": params}
    # Through a temporary file: the dump is ~80 MB, and a write interrupted on
    # the real path leaves a half file that raises on every boot after it.
    tmp = f"{cache_path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, cache_path)
    return payload
