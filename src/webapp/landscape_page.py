# The landscape page and its endpoints: the render, the word tables both kinds
# of page read, the drawn-token export, the auto suggestions, and the per-token
# component table.

import base64
import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from flask import jsonify, render_template, request

from ica.centroid_computation import preprocessed_vectors
from ica.closest_words_centroid import average_similarity, cluster_words
from ica.decomposition import tail_count
from ica.token_helper import (bare_name, drawn_phases, phase_prefix,
                              token_phase)

from .app import BadInput, app
from .params import (DEFAULT_PROMPT_SET, RENDER_PARAMS, RUN_PHASES,
                     WORD_PARAMS, csv_response, drawn_kwargs, nearest_kwargs,
                     num, page_context, phase_filter, prompt_sets, read,
                     rounded)
from .state import (COMPONENT_DIRECTIONS, DIST_PHASES, DIST_SPEAKERS,
                    DIST_WORDS, DRAWN_SPEAKERS, landscape, render_lock,
                    tweet_counts)


# The prompt sets this page offers, which is the labels and nothing else: one
# query at a time, written and graded on the spot (webapp.narrative), with no
# runs of its own to file anywhere. The set travels with each call instead.
PROMPT_SETS = prompt_sets()


@app.get("/")
def index():
    return render_template("index.html", prompt_sets=PROMPT_SETS,
                           default_prompt_set=DEFAULT_PROMPT_SET,
                           **page_context(RENDER_PARAMS,
                                          prompt_sets=PROMPT_SETS))


def grouping_args(args):
    """The display grouping as /render and /words read it: the clustering
    threshold (None when grouping is off) and the space the words compare in."""
    threshold = read(args, "group_similarity") if read(args, "group_words") else None
    space = {"mean_centre": read(args, "mean_centre"), "unit_norm": read(args, "unit_norm")}
    return threshold, space


def word_clusters(near, threshold, space):
    """One phase's nearest words as clusters — every word its own when
    threshold is None — plus the whole list's average pairwise similarity.
    Call under render_lock: the embedding caches are not thread-safe."""
    names = list(near["word"])
    whole = average_similarity(landscape.emb, names, **space)
    clusters = (cluster_words(landscape.emb, names, threshold, **space)
                if threshold is not None else [[w] for w in names])
    return clusters, whole


def phase_queries(near, threshold, space):
    """
    One phase's rows as the queries retrieval will run — one per cluster,
    members joined in the order the row shows them — so a count asked for a
    grouped row is the count of exactly that query.
    """
    if near is None:
        return []
    similarity = dict(zip(near["word"], near["similarity"]))
    clusters, _ = word_clusters(near, threshold, space)
    return [" ".join(sorted(c, key=lambda w: -similarity[w])) for c in clusters]


def cluster_rows(clusters, chip, row_count, space, whole):
    """
    One display row per cluster of words — the shape the centroid search and
    the component word lists share. A grouped row is one retrieval query, so
    its tweet count sits on the row — the tweets close to the joined query,
    not to its words one by one — and its chips lose their per-word counts.
    A cluster adds the cohesion pair: how alike its words are next to how
    alike the whole list (`whole`) is; a single word has no pair and shows
    nothing.
    """
    rows = []
    for cluster in clusters:
        chips = [chip(w) for w in cluster]
        grouped = len(cluster) > 1
        if grouped:
            for c in chips:
                c["tweets"] = None
        rows.append({"words": chips,
                     "tweets": row_count(cluster) if grouped else None,
                     "cohesion": rounded(average_similarity(landscape.emb, cluster,
                                                            **space)),
                     "list_cohesion": whole})
    return rows


def nearest_rows(near, counts, threshold, space, component):
    """
    The display rows of one phase's word table: one per cluster when grouping
    is on, one per word otherwise, ordered by the centroid similarity of
    their best word (the order is this function's to promise, not the
    clusterer's to inherit). Each single word keeps its centroid similarity
    and, when the tweet filter asked for a scan, its phase's tweet count;
    `counts` is keyed by the rows' joined queries (phase_queries reads the
    same clusters) and is None when nothing was scanned. Each chip also
    carries the word's value on `component` — coloured as the token table
    colours the same number — and its corpus count.
    """
    if near is None:
        return None
    similarity = dict(zip(near["word"], near["similarity"]))
    clusters, whole = word_clusters(near, threshold, space)
    ordered = [sorted(c, key=lambda w: -similarity[w]) for c in clusters]
    direction = COMPONENT_DIRECTIONS[component]

    def chip(w):
        value = float(landscape.ica_embedding[landscape.emb.key_to_index[w]][component])
        return {"word": w, "similarity": float(similarity[w]),
                "tweets": None if counts is None else counts.get(w),
                "value": round(value, 2), "cls": component_level(value, direction),
                "count": int(landscape.emb.get_vecattr(w, "count"))}

    rows = cluster_rows(ordered, chip,
                        lambda c: None if counts is None else counts.get(" ".join(c)),
                        space, rounded(whole))
    return sorted(rows, key=lambda row: -row["words"][0]["similarity"])


def component_word_tables(args):
    """
    The word tables when a run page takes the component's own words instead of
    searching around a centroid: the same strong/extreme lists the landscape
    page labels, under the same settings, so a list here can be checked
    against the plot's labels. One card, for the one phase the run will scan
    ('all' is a drawn view; the run pages do not offer it). Call under
    render_lock: the Landscape caches are not thread-safe.
    """
    source = read(args, "word_source")  # "strong words" | "extreme words"
    component = read(args, "component")
    family = "strong" if source == "strong words" else "extreme"
    kind = "strong_word_labels" if family == "strong" else "label_words"
    drawn = drawn_kwargs(args)  # same values, and so same cache keys, as /render
    words = landscape.component(component, kind,
                                k=drawn[f"n_{family}_words"],
                                min_count=drawn[f"{family}_min_count"],
                                order=drawn[f"{family}_order"])
    # The rows carry what chose these words — each one's value on the
    # component, coloured as the token table colours its cells — instead of
    # what the centroid search would have measured.
    direction = COMPONENT_DIRECTIONS[component]
    phase = read(args, "phase")
    # The same tweet floor and grouping the centroid search reads — one set of
    # controls, applied to whichever word source is chosen. The floor keeps
    # only the words enough tweets sit close to, closeness read in the
    # sentence-encoder space the tweets are embedded in, not the word2vec
    # space that chose the words; counted for the phase being harvested.
    floor = read(args, "word_min_similarity") if read(args, "tweet_filter") else None
    min_tweets = read(args, "word_min_tweets")
    counts = {} if floor is None else tweet_counts(
        {phase_filter(phase): list(words)}, floor)
    threshold, space = grouping_args(args)

    def phase_table(p):
        phase_counts = counts.get(phase_filter(p))

        kept = (words if phase_counts is None
                else [w for w in words if phase_counts[w] >= min_tweets])

        def chip(w):
            value = float(landscape.ica_embedding[landscape.emb.key_to_index[w]][component])
            return {"word": w, "similarity": None,
                    "tweets": None if phase_counts is None else phase_counts[w],
                    "value": round(value, 2), "cls": component_level(value, direction),
                    "count": int(landscape.emb.get_vecattr(w, "count"))}

        # Grouping off is every word its own cluster, in the order the
        # settings asked for. Group counts are scanned only for the rows that
        # need them (cluster_rows puts them on the row).
        clusters = ([[w] for w in kept] if threshold is None
                    else cluster_words(landscape.emb, kept, threshold, **space))
        joined = [" ".join(c) for c in clusters if len(c) > 1]
        group_counts = ({} if phase_counts is None or not joined else
                        tweet_counts({phase_filter(p): joined},
                                     floor)[phase_filter(p)])
        whole = (rounded(average_similarity(landscape.emb, kept, **space))
                 if threshold is not None else None)
        rows = cluster_rows(clusters, chip,
                            lambda c: group_counts.get(" ".join(c)), space, whole)
        meta = f"{len(kept)} {source} of component {component}"
        return {"phase": p,
                "title": "All phases pooled" if p == "pooled" else f"Phase {p}",
                "meta": meta, "component": component,
                "tweet_similarity": floor, "rows": rows}

    return None, [phase_table(phase)]


def word_tables(args):
    """
    The per-phase word tables both landscape endpoints answer with, identical
    on every page: the phase as title, the centroid and its concentration
    under it, and a row per word or per group of similar words. Call under
    render_lock: the Landscape caches are not thread-safe.
    """
    # The run pages can point their list away from the search; the landscape
    # page never sends word_source and stays on the centroid.
    if read(args, "word_source") != "centroid":
        return component_word_tables(args)
    kwargs = nearest_kwargs(args)
    note, tables = landscape.word_tables(landscape.nearest(**kwargs),
                                         kwargs["component"], kwargs["centroid_source"])
    threshold, space = grouping_args(args)
    # With the tweet filter off nothing is scanned and the rows carry no
    # count; the floor travels with the tables whenever there is one, so a
    # kept list goes on saying what its numbers were counted at.
    floor = read(args, "word_min_similarity") if read(args, "tweet_filter") else None
    # Every phase in one scan (the corpus product is the same for all), per
    # row query rather than per word: a grouped row's query is its words
    # joined, the exact string retrieval will encode.
    counts = {} if floor is None else tweet_counts(
        {phase_filter(t["phase"]): phase_queries(t["nearest"], threshold, space)
         for t in tables if t["nearest"] is not None}, floor)
    return note, [{"phase": t["phase"], "title": t["title"], "meta": t["meta"],
                   "component": kwargs["component"], "tweet_similarity": floor,
                   "rows": nearest_rows(t["nearest"], counts.get(phase_filter(t["phase"])),
                                        threshold, space, kwargs["component"])}
                  for t in tables]


@app.get("/render")
def render_endpoint():
    args = request.args
    params = dict(nearest_kwargs(args), **drawn_kwargs(args))
    with render_lock:
        try:
            fig, note, _ = landscape.render(**params)
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=110)
            plt.close(fig)
        except Exception:
            plt.close("all")  # a half-drawn figure must not outlive the request
            raise
        # The render just cached the nearest() this asks for again.
        _, tables = word_tables(args)
    return jsonify({
        "image": base64.b64encode(buf.getvalue()).decode(),
        "note": note,
        # The page builds the word lists itself; the DataFrame stays on this side.
        "tables": tables,
    })


@app.get("/words")
def words_endpoint():
    """
    The word tables of /render without drawing anything — the run pages'
    endpoint, the same cards the landscape shows. One phase only: a run
    harvests one list for one span of the corpus, so 'all' is refused rather
    than silently answered with four lists a run has no way to be about.
    """
    if read(request.args, "phase") == "all":
        raise BadInput("phase 'all' is a drawn view, not a run: harvest one "
                       f"phase, or 'pooled'. Choose from {list(RUN_PHASES)}.")
    with render_lock:  # the Landscape caches are not thread-safe
        note, tables = word_tables(request.args)
    # Only here: the landscape page has an 'off' state where every table is
    # empty by design, and these pages do not.
    if note is None and not any(t["rows"] for t in tables):
        note = "No words found with these parameters."
    return jsonify({"tables": tables, "note": note})


# The tokens the plot draws, exported for analyses outside the app. A speaker
# token is split into its phase and the bare name the tweet table stores —
# what an external dataset can be joined on.

EXPORT_COLUMNS = ["phase", "type", "token", "count", "value"]

# What the CSV calls the word-search parameters whose page name says too
# little on its own; the rest keep their page names.
EXPORT_PARAM_NAMES = {"unit_norm": "centroid_unit_norm",
                      "mean_centre": "centroid_mean_centre",
                      "n_extreme": "centroid_source_n_extreme",
                      "word_filter": "nearest_word_among",
                      "min_similarity": "centroid_min_similarity"}

# The value-behind-a-checkbox floors: with the box off the CSV writes the
# do-nothing value — 0 for counts, -1 for the similarity floor — so the
# checkbox columns themselves stay out.
EXPORT_FLOOR_FLAGS = {"min_count": "min_count_filter",
                      "min_similarity": "similarity_filter",
                      "word_min_tweets": "tweet_filter",
                      "word_min_similarity": "tweet_filter"}


def token_rows(kind, tokens, component, phase):
    """Export rows for one drawn family: its tokens, their corpus counts and
    their z-scores — each value with the colour class the token table gives
    the same number. Under the pooled view a speaker row keeps the phase its
    token carries, so the export stays joinable per phase."""
    direction = COMPONENT_DIRECTIONS[component]
    rows = []
    for t in tokens:
        value = float(landscape.ica_embedding[landscape.emb.key_to_index[t]][component])
        rows.append({"phase": (token_phase(t) or phase) if phase == "pooled" else phase,
                     "type": kind,
                     "token": bare_name(t),
                     "count": int(landscape.emb.get_vecattr(t, "count")),
                     "value": round(value, 2),
                     "cls": component_level(value, direction)})
    return rows


def drawn_speakers(tokens):
    """The speaker tokens the plot carries a dot for: the families are chosen
    in ICA space but the figure only shows the speakers UMAP was fitted on —
    a row for a dropped one would report a dot that is not on the plot."""
    return [t for t in tokens if t in DRAWN_SPEAKERS]


def displayed_tokens(args):
    """
    One row per token the plot draws under these parameters, families in
    legend order. Call under render_lock: the component lookups write caches.
    """
    component = read(args, "component")
    phase = read(args, "phase")
    phases = drawn_phases(phase)
    rows = []
    if read(args, "show_strong_speakers"):
        strong = drawn_speakers(landscape.component(component, "strong_speakers"))
        for p in phases:
            rows += token_rows("strong speaker",
                               strong if p == "pooled" else
                               [t for t in strong if t.startswith(phase_prefix(p))],
                               component, p)
    if read(args, "show_extreme_speakers"):
        for p in phases:
            rows += token_rows("extreme speaker",
                               drawn_speakers(landscape.component(
                                   component, "extreme_speakers",
                                   read(args, "n_extreme_speakers"),
                                   phase=None if p == "pooled" else p)),
                               component, p)
    # The word families are shared by every panel, so their phase reads "all".
    drawn = drawn_kwargs(args)  # the same values, and so the same cache keys, as the render
    if read(args, "strong_labels"):
        rows += token_rows("strong word",
                           landscape.component(component, "strong_word_labels",
                                               k=drawn["n_strong_words"],
                                               min_count=drawn["strong_min_count"],
                                               order=drawn["strong_order"]),
                           component, "all")
    if read(args, "extreme_labels"):
        rows += token_rows("extreme word",
                           landscape.component(component, "label_words",
                                               drawn["n_extreme_words"],
                                               min_count=drawn["extreme_min_count"],
                                               order=drawn["extreme_order"]),
                           component, "all")
    if read(args, "word_filter") != "off":
        for p, _, _, near, _ in landscape.nearest(**nearest_kwargs(args)):
            if near is not None:
                rows += token_rows("nearest word", near["word"], component, p)
    return rows


@app.get("/export")
def export_endpoint():
    """The drawn tokens as rows, for the table the page shows under the plot."""
    with render_lock:
        rows = displayed_tokens(request.args)
    return jsonify({"rows": rows})


@app.get("/export.csv")
def export_csv():
    """
    The same rows as one CSV, with the word-search parameters repeated on the
    nearest-word rows they produced, so a file keeps saying what made its rows
    after it is concatenated with other exports. The display toggles stay out
    — which families were drawn is readable off the rows themselves.
    """
    args = request.args
    with render_lock:
        rows = displayed_tokens(args)
    component = read(args, "component")
    table = pd.DataFrame(rows, columns=EXPORT_COLUMNS)
    # First the component and how to read `value` on it: a positive z-score
    # points the way the component's strong words do only when direction is +.
    table.insert(0, "component", component)
    table.insert(1, "direction", COMPONENT_DIRECTIONS[component])
    # The parameters only apply to the nearest-word rows; the overlay rows
    # leave them empty rather than implying the tokens depend on them.
    for name in WORD_PARAMS:
        if name in ("component", "phase") or name in EXPORT_FLOOR_FLAGS.values():
            continue
        value = read(args, name)
        if name in EXPORT_FLOOR_FLAGS and not read(args, EXPORT_FLOOR_FLAGS[name]):
            value = -1 if name == "min_similarity" else 0
        # k_extreme and n_extreme are unread unless their setting selects
        # them; unread, their cells stay empty.
        if name == "k_extreme" and read(args, "word_filter") != "extreme words":
            value = ""
        if name == "n_extreme" and read(args, "centroid_source") != "extreme speakers":
            value = ""
        table[EXPORT_PARAM_NAMES.get(name, name)] = [
            value if t == "nearest word" else "" for t in table["type"]]
    return csv_response(table.to_csv(index=False),
                        f"tokens_c{component}_phase-{read(args, 'phase')}.csv")


@app.get("/suggested")
def suggested_endpoint():
    """
    One component's tail size per token family — what the auto >3σ buttons
    set their top-K boxes from. A plain read of the ICA matrix under the
    boot-time masks, so no lock, like /token.
    """
    args = request.args
    component = read(args, "component")
    phase = read(args, "phase")
    direction = COMPONENT_DIRECTIONS[component]
    values = landscape.ica_embedding[:, component]

    def tail(mask):
        return tail_count(values[mask], direction)

    # The speaker count follows how N is consumed: within the drawn phase,
    # across all speakers under 'pooled', and under 'all' — one N per panel —
    # the mean of the per-phase tails rather than their sum.
    if phase == "all":
        speakers = np.mean([tail(DIST_SPEAKERS & (DIST_PHASES == p))
                            for p in "1234"])
    else:
        speakers = tail(DIST_SPEAKERS if phase == "pooled"
                        else DIST_SPEAKERS & (DIST_PHASES == phase))
    # The boxes' floor is 1: an empty tail still has to be a settable value.
    return jsonify({"speakers": max(int(round(speakers)), 1),
                    "words": max(tail(DIST_WORDS), 1)})


# One token across every retained component — the question the plot cannot be
# asked, drawing one component at a time. FastICA whitens its sources to unit
# variance, so a value is a z-score readable on the one scale; the sign is
# arbitrary per component, so the page colours by whether a value points the
# way the component's strong words do.
LEVELS = [0.5, 1, 2, 3]


def component_level(value, direction):
    """
    The colour class of one component value: g1-g4 towards the component's
    direction, r1-r4 away from it, and lvl0 for a value too small to mean
    anything — half a deviation, against a 99th percentile of about three.
    """
    aligned = value if direction == "+" else -value
    level = sum(abs(aligned) >= threshold for threshold in LEVELS)
    if not level:
        return "lvl0"
    return f"{'g' if aligned > 0 else 'r'}{level}"


def component_cells(word):
    """One coloured cell per retained component for one vocabulary word."""
    values = landscape.ica_embedding[landscape.emb.key_to_index[word]]
    return [{"value": round(value, 2), "cls": component_level(value, direction)}
            for c in landscape.selected_components
            for direction in [COMPONENT_DIRECTIONS[c]]
            for value in [float(values[c])]]


def centred_neighbours(token, topn, min_count):
    """
    The topn nearest content words of one token, mean-centred and renormalized
    exactly as the centroid search centres its vocabulary — so a word close to
    a centroid means the same kind of close here. The frequency floor is
    applied straight down the full ranking.
    """
    words, matrix = landscape.vocabulary(mean_centre=True, min_count=1)
    vector = preprocessed_vectors(landscape.emb, [token])[0]
    similarities = matrix @ vector
    kept = []
    for i in np.argsort(similarities)[::-1]:
        word = words[i]
        # The token itself: a content word is its own nearest centred neighbour.
        if word == token or landscape.emb.get_vecattr(word, "count") < min_count:
            continue
        kept.append((word, float(similarities[i])))
        if len(kept) == topn:
            break
    return kept


@app.get("/token")
def token_endpoint():
    """One token and its nearest words, each across every retained component."""
    raw = (request.args.get("token") or "").strip()
    topn = min(max(num(request.args.get("topn"), 10, int), 1), 50)
    # No floor at all is a floor of one: every word the tokenizer kept qualifies.
    min_count = max(num(request.args.get("min_count"), 1, int), 1)
    # An empty box is not an empty token: '' is itself a vocabulary entry.
    if not raw:
        return jsonify({"note": "Type a token.", "components": [], "rows": []})
    # The vocabulary is lower-case and joins multi-word tokens with '_'.
    token = "_".join(raw.lower().split())
    if token not in landscape.emb.key_to_index:
        return jsonify({"note": f"'{token}' is not in the embedding vocabulary.",
                        "components": [], "rows": []})
    neighbours = centred_neighbours(token, topn, min_count)
    # The corpus count is worth a glance next to the values: a word written a
    # handful of times has component values fit on that handful.
    note = (f"'{token}' — {landscape.emb.get_vecattr(token, 'count')} "
            "occurrences in the corpus")
    # A short table is the floor doing its work, not a search that went wrong.
    if len(neighbours) < topn:
        note += (f"; only {len(neighbours)} of its neighbours are written "
                 f"{min_count} times or more")
    return jsonify({
        "note": note,
        "components": [{"component": c, "direction": COMPONENT_DIRECTIONS[c]}
                       for c in landscape.selected_components],
        # The token's own row first: its similarity to itself reads 1, and the
        # rows under it are its nearest content words, nearest first.
        "rows": [{"word": w, "similarity": round(float(s), 2),
                  "cells": component_cells(w)}
                 for w, s in [(token, 1.0), *neighbours]],
    })
