# Drawing of one phase's speaker landscape onto a matplotlib Axes

import numpy as np
from adjustText import adjust_text

from .decomposition import get_extreme_words, dominant_direction
from .stance_labels import stance_colour
from .token_helper import phase_prefix


def phase_speaker_positions(speaker_tokens, coords, phase):
    """One phase's speakers as aligned (names, xy): the bare names, prefix
    stripped, and their 2D positions in the fitted projection."""
    prefix = phase_prefix(phase)
    indices = [i for i, token in enumerate(speaker_tokens) if token.startswith(prefix)]
    names = [speaker_tokens[i][len(prefix):] for i in indices]
    return names, coords[indices]


def component_label_words(ica_embedding, emb, component_index, k=20, strongest=None,
                          min_count=None):
    """The k extreme words of a component's dominant direction — the side
    holding most of its strong words (the interpretable pole), whatever +/-
    sign FastICA assigned. `min_count` floors them on corpus frequency."""
    direction = dominant_direction(ica_embedding, emb, component_index, strongest)
    return get_extreme_words(ica_embedding, emb, component_index, direction, k=k,
                             min_count=min_count)


def highlight_positions(tokens, phase, fitted_speakers, coords):
    """The 2D positions of the given speaker tokens, restricted to `phase`
    unless it is None (the pooled view). Speakers below the tweet threshold
    were never projected and are silently dropped."""
    wanted = (set(tokens) if phase is None
              else {token for token in tokens if token.startswith(phase_prefix(phase))})
    indices = [i for i, token in enumerate(fitted_speakers) if token in wanted]
    return coords[indices] if indices else np.empty((0, 2))


def draw_phase_landscape(ax, phase, names, xy, phase_labels,
                         highlight_groups=(), word_label_groups=(),
                         title=None, max_adjusted_labels=150, spread=1.0,
                         colours=None):
    """
    Draw one phase's landscape on the given Axes: the stance-coloured speaker
    cloud, then (coords, colour) highlight_groups over it, then
    (words, coords, colour) word_label_groups — each word's true position
    marked with a tri_down marker, the text possibly nudged off it by
    adjustText.
    """
    if colours is None:
        colours = [stance_colour(name, phase_labels) for name in names]
    ax.scatter(xy[:, 0], xy[:, 1], c=colours, s=3, alpha=0.1)

    for coords, colour in highlight_groups:
        if len(coords):
            ax.scatter(coords[:, 0], coords[:, 1], c=colour, s=3, alpha=1)

    texts = []
    for words, coords, colour in word_label_groups:
        if len(coords):
            ax.scatter(coords[:, 0], coords[:, 1], marker="1", c=colour,
                       s=18, alpha=1, zorder=5)
        for word, (x, y) in zip(words, coords):
            texts.append(ax.text(x, y, word, fontsize=8, color=colour))

    if texts and spread > 0 and len(texts) <= max_adjusted_labels:
        adjust_text(texts, ax=ax,
                    force_text=(0.3 * spread, 0.4 * spread),
                    force_static=(0.1 * spread, 0.2 * spread),
                    force_pull=(0.01, 0.01),
                    expand=(1 + 0.1 * spread, 1 + 0.2 * spread),
                    max_move=None, iter_lim=int(20 + 10 * spread))

    ax.axis("off")
    ax.set_title(title if title is not None else f"Phase {phase}")
