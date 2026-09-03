# Stance labels and colours for speakers

import pandas as pd

from config import SPEAKER_LABELS_CSV, LABEL_COLUMN

COLOURMAP = {"anti-vax": "red", "pro-vax": "green", "news": "blue", "unclear": "darkgray"}
DEFAULT_COLOUR = "darkgray"


def load_stance_labels(path=SPEAKER_LABELS_CSV, label_column=LABEL_COLUMN):
    """The per-phase stance labels of every speaker, as phase ('1'..'4') ->
    {bare_name: label}. The CSV stores speakers as 'agent_{name}'; the prefix
    is stripped and the phases are compared as strings."""
    df = pd.read_csv(path, usecols=["speaker", "phase", label_column])
    df["phase"] = df["phase"].astype(str)
    # Two columns zipped into a dict rather than a Series built per row:
    # iterrows takes 6 s on this file against 0.3 s here, and it is the
    # largest boot cost of the app after the sentence encoder.
    return {phase: dict(zip(group["speaker"].str[6:], group[label_column]))
            for phase, group in df.groupby("phase")}


def stance_colour(speaker_name, phase_labels, colourmap=COLOURMAP):
    """One speaker's colour from its stance label; DEFAULT_COLOUR when it is
    unlabelled or its label is not in the colourmap."""
    return colourmap.get(phase_labels.get(speaker_name), DEFAULT_COLOUR)
