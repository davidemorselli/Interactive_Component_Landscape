# The component landscape, shared by the notebook and the web app.
#
# build_landscape() runs the data pipeline once; the Landscape it returns holds
# that state, caches the expensive lookups, and renders a set of parameters into
# a figure plus the nearest-word tables. Front-ends only display the result.

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from .centroid_computation import compute_centroid
from .closest_words_centroid import content_vocabulary, nearest_words
from .decomposition import (load_or_compute_ica_embeddings, dominant_direction,
                            get_components_with_strong_words, get_extreme_words,
                            get_strong_words, strongest_components)
from .get_speakers import get_extreme_speakers, get_strong_speakers
from .landscape_plot import (component_label_words, draw_phase_landscape,
                             highlight_positions, phase_speaker_positions)
from .load_data import load_embedding
from .stance_labels import COLOURMAP, load_stance_labels, stance_colour
from .token_helper import (bare_name, drawn_phases, is_speaker_token,
                           phase_prefix, token_phase)
from .umap_projection import (compute_umap_projection, load_speakers_tokens,
                              project_tokens, project_vectors)

STRONG_WORD_COLOUR = "#9a3412"


def configure_fonts():
    """
    Fall back to emoji and CJK fonts for non-latin tokens.

    Silences the missing-glyph warnings for any script not covered by these
    fonts either. Call once, before rendering.
    """
    plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Emoji", "Noto Sans Mono CJK JP"]
    warnings.filterwarnings("ignore", message="Glyph .* missing from font")


def build_landscape(strong_word_ratio_threshold=0.007, progress=None):
    """Run the pipeline — embedding, ICA, retained components, 2D projection,
    stance labels — and return the Landscape they make up. `progress` is an
    optional (label, fn) -> fn() to report each step as it runs."""
    step = progress or (lambda label, fn: fn())

    emb = step("load embedding", load_embedding)
    ica_embedding = step("ICA embedding", lambda: load_or_compute_ica_embeddings(emb.vectors))
    selected_components = step("retained components", lambda: get_components_with_strong_words(
        ica_embedding, emb, strong_word_ratio_threshold=strong_word_ratio_threshold))
    reducer, fitted_speakers, coords = step("UMAP projection", lambda: compute_umap_projection(
        emb, load_speakers_tokens()))
    stance = step("stance labels", load_stance_labels)

    return Landscape(emb, ica_embedding, selected_components,
                     reducer, fitted_speakers, coords, stance)


class Landscape:
    """
    The pipeline state, its caches, and the render built on top of them.

    Built by build_landscape() rather than directly. The caches make widget
    and web interaction responsive: every render reuses the vocabularies,
    per-component lookups and token coordinates computed by earlier ones.
    """

    def __init__(self, emb, ica_embedding, selected_components,
                 reducer, fitted_speakers, coords, stance):
        self.emb = emb
        self.ica_embedding = ica_embedding
        self.selected_components = selected_components
        self.reducer = reducer
        self.fitted_speakers = fitted_speakers
        self.coords = coords
        self.stance = stance
        self.phase_positions = {p: phase_speaker_positions(fitted_speakers, coords, p)
                                for p in "1234"}
        # The pooled view: every fitted speaker of every phase at once, each
        # point coloured by its own phase's stance (a stance can change
        # between phases); computed once, stance being fixed at build time.
        bare_names = [bare_name(t) for t in fitted_speakers]
        self.phase_positions["pooled"] = (bare_names, coords)
        self.pooled_colours = [stance_colour(name, stance.get(token_phase(t), {}))
                               for t, name in zip(fitted_speakers, bare_names)]
        # The strong-word assignment, shared by every lookup and every render.
        self.strongest = strongest_components(ica_embedding)

        self._vocab_cache = {}      # (mean_centre, min_count, unit_norm) -> (words, matrix)
        self._component_cache = {}  # (component, kind, k, phase, min_count, order)
        self._coords_cache = {}     # token -> (x, y)
        # These two hold one entry each — see candidates() and nearest().
        self._candidate_cache = {}  # search arguments -> (words, matrix)
        self._nearest_cache = {}    # nearest() arguments -> its per-phase result

    def vocabulary(self, mean_centre, min_count, unit_norm=True):
        """Content-word vocabulary and its vectors, cached per parameter set."""
        key = (mean_centre, min_count, unit_norm)
        if key not in self._vocab_cache:
            self._vocab_cache[key] = content_vocabulary(
                self.emb, mean_centre=mean_centre, min_count=min_count,
                unit_norm=unit_norm)
        return self._vocab_cache[key]

    def candidates(self, component, direction, word_filter, k_extreme,
                   mean_centre, min_count, unit_norm):
        """
        The vocabulary the nearest-word search ranks, resolved once for every
        phase drawn. Only the latest entry is kept: k_extreme comes off a
        slider, so keying on it would grow without bound, while one render
        only ever asks for one set of arguments.
        """
        key = (component, direction, word_filter, k_extreme,
               mean_centre, min_count, unit_norm)
        if key not in self._candidate_cache:
            words, matrix = self.vocabulary(mean_centre, min_count, unit_norm)
            allowed = None
            if word_filter == "strong words":
                allowed = get_strong_words(self.ica_embedding, self.emb, component,
                                           direction, self.strongest)
            elif word_filter == "extreme words":
                allowed = get_extreme_words(self.ica_embedding, self.emb, component,
                                            direction, k=k_extreme)
            if allowed is not None:
                mask = np.isin(words, allowed)
                words, matrix = words[mask], matrix[mask]
            self._candidate_cache = {key: (words, matrix)}
        return self._candidate_cache[key]

    def component(self, component, kind, k=None, phase=None, min_count=None,
                  order=None):
        """A component's direction, speakers or label words, cached per lookup."""
        lookups = {
            "direction": lambda: dominant_direction(
                self.ica_embedding, self.emb, component, self.strongest),
            "strong_speakers": lambda: get_strong_speakers(
                self.ica_embedding, self.emb, component, self.strongest),
            "extreme_speakers": lambda: get_extreme_speakers(
                self.ica_embedding, self.emb, component, k=k, phase=phase,
                strongest=self.strongest),
            "label_words": lambda: self._extreme_word_labels(
                component, k, min_count, order),
            "strong_word_labels": lambda: self._strong_word_labels(
                component, k if k is not None else 20, min_count, order),
        }
        if kind not in lookups:
            raise ValueError(f"unknown component lookup {kind!r}")
        key = (component, kind, k, phase, min_count, order)
        if key not in self._component_cache:
            self._component_cache[key] = lookups[kind]()
        return self._component_cache[key]

    def _strong_word_labels(self, component, n, min_count, order):
        """The n strong words render() labels, non-speakers only — the same
        list /export reports. min_count floors on corpus use before the cap;
        `order` as in _ordered."""
        words = [w for w in get_strong_words(
            self.ica_embedding, self.emb, component,
            self.component(component, "direction"), self.strongest)
            if not is_speaker_token(w)]
        if min_count is not None:
            words = [w for w in words
                     if self.emb.get_vecattr(w, "count") >= min_count]
        return self._ordered(words, component, order)[:n]

    def _extreme_word_labels(self, component, n, min_count, order):
        """The n extreme words render() labels, floored on corpus frequency
        before the cut so a floor costs no labels; `order` as in _ordered
        (the ranking itself leaves the '-' side most-extreme-last)."""
        words = component_label_words(self.ica_embedding, self.emb, component, k=n,
                                      strongest=self.strongest, min_count=min_count)
        return self._ordered(words, component, order)

    def _ordered(self, words, component, order):
        """One label family in the asked order: most written first, or the
        aligned component value descending — not bare |value|, so a list that
        had to reach into the opposite pole puts that pole last instead of
        interleaving the two by magnitude."""
        if order == "component strength":
            sign = 1.0 if self.component(component, "direction") == "+" else -1.0
            return sorted(words, key=lambda w: -sign * float(
                self.ica_embedding[self.emb.key_to_index[w], component]))
        if order == "corpus frequency":
            return sorted(words, key=lambda w: -self.emb.get_vecattr(w, "count"))
        return words

    def project(self, tokens):
        """2D coordinates of tokens, projecting only the ones not seen yet."""
        new = [t for t in tokens if t not in self._coords_cache]
        if new:
            for token, xy in zip(new, project_tokens(self.reducer, self.emb, new)):
                self._coords_cache[token] = xy
        return np.array([self._coords_cache[t] for t in tokens]) if tokens else np.empty((0, 2))

    def nearest(self, component, phase, unit_norm, mean_centre, centroid_source,
                n_extreme, word_filter, k_extreme, min_count, topn,
                min_similarity=None, word_keeper=None):
        """
        Centroid and nearest words for each drawn phase, without any drawing:
        one (phase, tokens, centroid_xy, nearest, concentration) tuple per
        phase, the last three None when the phase has no tokens. `word_keeper`
        — (words, phase) -> the words to keep — takes part in the cache key,
        so a caller that wants the cache to recognise a repeat must pass the
        same OBJECT, not an equivalent one (webapp's word_keeper_for does).
        """
        key = (component, phase, unit_norm, mean_centre, centroid_source, n_extreme,
               word_filter, k_extreme, min_count, topn, min_similarity, word_keeper)
        if key not in self._nearest_cache:
            # One entry only: any parameter change makes the tables stale, and
            # they hold a DataFrame each. It earns its keep within a single
            # interaction — the page asks for the drawn tokens right after the
            # render that drew them, with the same arguments.
            self._nearest_cache = {key: self._nearest(
                component, phase, unit_norm, mean_centre, centroid_source,
                n_extreme, word_filter, k_extreme, min_count, topn,
                min_similarity, word_keeper)}
        return self._nearest_cache[key]

    def word_tables(self, per_phase, component, centroid_source):
        """
        The per-phase word tables of a nearest() result, split out of render()
        so a page that only wants the words shows the same cards as one that
        draws them. Returns (note, tables): a string set only when no centroid
        could be computed, and one dict per drawn phase — 'title' and 'meta'
        are the halves 'heading' joins, 'nearest' a DataFrame or None.
        """
        if not any(tokens for _, tokens, _, _, _ in per_phase):
            return (f"Component {component} has no {centroid_source} — "
                    "no centroid computed."), []
        tables = []
        for p, tokens, _, nearest, concentration in per_phase:
            scope = "All phases pooled" if p == "pooled" else f"Phase {p}"
            meta = f"centroid from {len(tokens)} {centroid_source}"
            if concentration is not None:
                meta += f", concentration {concentration:.2f}"
            tables.append({
                "phase": p, "heading": f"{scope}: {meta}", "title": scope, "meta": meta,
                "n_tokens": len(tokens), "concentration": concentration,
                "nearest": nearest if (nearest is not None and tokens) else None,
            })
        return None, tables

    def _nearest(self, component, phase, unit_norm, mean_centre, centroid_source,
                 n_extreme, word_filter, k_extreme, min_count, topn,
                 min_similarity, word_keeper):
        """The uncached nearest(); call that one instead."""
        direction = self.component(component, "direction")
        strong_speakers = self.component(component, "strong_speakers")
        phases = drawn_phases(phase)

        def centroid_tokens_for(p):
            """Speaker tokens averaged into phase p's centroid. Strong
            speakers are a global assignment, so a phase takes the share
            carrying its prefix; extreme speakers are ranked within the phase,
            so per-phase centroids stay comparable. The pooled pseudo-phase
            takes all strong speakers, or the extremes ranked across every
            phase at once (one phase can then dominate the ranking)."""
            if p == "pooled":
                return (list(strong_speakers) if centroid_source == "strong speakers"
                        else self.component(component, "extreme_speakers",
                                            n_extreme, phase=None))
            if centroid_source == "strong speakers":
                return [t for t in strong_speakers if t.startswith(phase_prefix(p))]
            return self.component(component, "extreme_speakers", n_extreme, phase=p)

        def centroid_and_nearest(tokens, p):
            """Centroid 2D position, nearest-word table and concentration for a set of speaker tokens."""
            if not tokens:
                return None, None, None
            centroid, concentration = compute_centroid(self.emb, tokens, mean_centre=mean_centre,
                                                       unit_norm=unit_norm)
            centroid_xy = project_vectors(self.reducer, centroid[None, :])[0]
            nearest = None
            if word_filter != "off":
                words, matrix = self.candidates(component, direction, word_filter,
                                                k_extreme, mean_centre, min_count,
                                                unit_norm)
                nearest = nearest_words(centroid, words, matrix, topn=topn)
                # A floor on how similar a nearest word must be to the centroid,
                # so a table can come back shorter than topn. Before the
                # word_keeper — a scan per word is what it costs.
                if min_similarity is not None:
                    nearest = nearest[nearest["similarity"] >= min_similarity] \
                        .reset_index(drop=True)
                if word_keeper is not None:
                    # The keeper is told which phase's table it is filtering: a
                    # word kept under "Phase 2" is one phase 2's own tweets
                    # back, not one the corpus at large does.
                    kept = word_keeper(nearest["word"].tolist(), p)
                    nearest = nearest[nearest["word"].isin(kept)].reset_index(drop=True)
            return centroid_xy, nearest, concentration

        # One centroid per drawn phase (mean_centre AND unit_norm drive BOTH the
        # centroid and the vocabulary it is compared against, so they always live
        # in the same space)
        return [(p, tokens, *centroid_and_nearest(tokens, p))
                for p in phases for tokens in [centroid_tokens_for(p)]]

    def render(self, component, phase, show_strong_speakers, strong_labels,
               show_extreme_speakers, extreme_labels, n_extreme_speakers, n_extreme_words,
               unit_norm, mean_centre, centroid_source, n_extreme, word_filter,
               k_extreme, min_count, topn, spread, scale=1.0,
               min_similarity=None, word_keeper=None,
               n_strong_words=20, strong_min_count=None,
               strong_order="corpus frequency",
               extreme_min_count=None, extreme_order="component strength"):
        """
        Draw the landscape for one component: (fig, note, tables) — the figure
        plus what word_tables() returns for the phases drawn.

        `phase` is '1'-'4', 'all' (a 2x2 grid), or 'pooled' — one plot of
        every phase's speakers, a speaker active in several phases appearing
        once per phase by design. The show_*/*_labels flags choose overlays;
        both label families take a count, a corpus-frequency floor (None for
        none) and an order. `spread` is the label separation (0 turns
        adjustText off), `scale` the figure size. The rest configures the
        centroid and the nearest-word search it anchors — see nearest().
        """
        direction = self.component(component, "direction")
        strong_speakers = self.component(component, "strong_speakers")
        phases = drawn_phases(phase)

        per_phase = self.nearest(component, phase, unit_norm, mean_centre, centroid_source,
                                 n_extreme, word_filter, k_extreme, min_count, topn,
                                 min_similarity, word_keeper)
        per_axes = {p: (xy, near) for p, _, xy, near, _ in per_phase}

        shared_groups = []
        if extreme_labels:
            extreme_words = self.component(component, "label_words", n_extreme_words,
                                           min_count=extreme_min_count,
                                           order=extreme_order)
            shared_groups.append((extreme_words, self.project(extreme_words), "deeppink"))
        if strong_labels:
            strong_words = self.component(component, "strong_word_labels",
                                          k=n_strong_words, min_count=strong_min_count,
                                          order=strong_order)
            shared_groups.append((strong_words, self.project(strong_words), STRONG_WORD_COLOUR))

        fig, axes = self._make_axes(phase, scale)
        fig.suptitle(f"Component {component} ({direction})")

        drew_nearest = drew_centroid = False  # track overlays actually drawn, for the legend
        for ax, p in zip(axes, phases):
            centroid_xy, nearest = per_axes[p]
            label_groups = list(shared_groups)
            if nearest is not None and len(nearest):
                nearest_tokens = nearest["word"].tolist()
                label_groups.append((nearest_tokens, self.project(nearest_tokens), "black"))
                drew_nearest = True

            # The pooled pseudo-phase spans every phase: no prefix filter on
            # the highlights, extreme speakers ranked across phases, and each
            # point coloured by its own phase's stance.
            hp = None if p == "pooled" else p
            highlights = []
            if show_strong_speakers:
                highlights.append((highlight_positions(strong_speakers, hp, self.fitted_speakers,
                                                       self.coords), "orange"))
            if show_extreme_speakers:
                extreme_sp = self.component(component, "extreme_speakers",
                                            n_extreme_speakers, phase=hp)
                highlights.append((highlight_positions(extreme_sp, hp, self.fitted_speakers,
                                                       self.coords), "deeppink"))

            names, xy = self.phase_positions[p]
            draw_phase_landscape(ax, p, names, xy, self.stance.get(p, {}),
                                 highlight_groups=highlights, word_label_groups=label_groups,
                                 spread=spread,
                                 colours=self.pooled_colours if p == "pooled" else None,
                                 title="All phases (pooled)" if p == "pooled" else None)
            if centroid_xy is not None:
                ax.scatter(*centroid_xy, marker="x", c="black", s=150, zorder=10)
                drew_centroid = True

        legend = _build_legend(show_strong_speakers, show_extreme_speakers,
                               strong_labels, extreme_labels,
                               drew_nearest, drew_centroid)
        axes[0].legend(handles=legend, loc="upper left", fontsize=8)
        fig.tight_layout()

        # Tables: one per drawn phase, the same ones /words answers with.
        note, tables = self.word_tables(per_phase, component, centroid_source)
        return fig, note, tables

    @staticmethod
    def _make_axes(phase, scale):
        """The figure and its axes list: a 2x2 grid for 'all', one otherwise."""
        if phase == "all":
            fig, axs = plt.subplots(2, 2, figsize=(12 * scale, 10 * scale))
            return fig, axs.ravel()
        fig, ax = plt.subplots(figsize=(8 * scale, 7 * scale))
        return fig, [ax]


def _build_legend(show_strong_speakers, show_extreme_speakers, strong_labels,
                  extreme_labels, drew_nearest, drew_centroid):
    """The legend handles of one render: the stance colours, then one entry
    per overlay actually drawn."""
    legend = [Line2D([0], [0], marker="o", color=c, label=l, markerfacecolor=c,
                     markersize=5, linestyle="") for l, c in COLOURMAP.items()]
    overlays = []  # (label, colour, marker)
    if show_strong_speakers:
        overlays.append(("Strong speakers", "orange", "o"))
    if show_extreme_speakers:
        overlays.append(("Extreme speakers", "deeppink", "o"))
    if strong_labels:
        overlays.append(("Strong words", STRONG_WORD_COLOUR, "1"))
    if extreme_labels:
        overlays.append(("Extreme words", "deeppink", "1"))
    if drew_nearest:
        overlays.append(("Nearest words", "black", "1"))
    legend += [Line2D([0], [0], marker=m, color=c, label=l, markerfacecolor=c,
                      markersize=7, linestyle="") for l, c, m in overlays]
    if drew_centroid:
        legend.append(Line2D([0], [0], marker="X", color="black", label="Centroid",
                             markerfacecolor="black", markeredgecolor="white",
                             markersize=8, linestyle=""))
    return legend
