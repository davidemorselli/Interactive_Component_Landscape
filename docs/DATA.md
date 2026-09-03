# The data files

All of them sit flat in one folder — `data/` next to the code.  `src/config.py` is where their names are declared; nothing
else in the code hardcodes a path.

Files marked **PONS Drive** are not produced by this project and cannot be
rebuilt from it. Download them from the shared Drive, under
`PONS/EXPERIMENTS/TERM-CORRELATION/Interactive_Component_Landscape/data/`.



## What you must bring

| File | What it is | Where it comes from |
| --- | --- | --- |
| `2021_SA_vaccine-total_phase-annotated.emb` | The word2vec embedding: one vector per word and per speaker of the corpus. This is the object the whole landscape is a picture of. | PONS Drive |
| `…_phase-annotated.emb.vectors.npy` | The vectors of that embedding. gensim splits its model in two files and loads them together by name, so it is really one file in two pieces — keep them side by side. | PONS Drive |
| `2021_SA_vaccine-total_phase-annotated_training-data.txt` | The corpus itself, cleaned, one tweet per line, prefixed by its speaker. It is what the embedding was trained on, and it is also the text the app quotes back when you search for tweets. | PONS Drive |
| `WORD2VEC-SPEAKER-LABELS.csv` | The stance label of each speaker, per phase — the pro/anti colouring you see on the map. Produced by the wider PONS project, by a classifier, not by anything here. | PONS Drive |
| `tweet_vectors.npy` | One vector per tweet, used to find the tweets closest to a word. | Built by `src/rag/encode_corpus/`, but take the Drive copy |
| `tweet_vector_texts.csv` | The text of each of those tweets. | idem |
| `tweet_rows.csv` | The link between the two above: which tweet is which vector. | idem |

The last three are **only meaningful together** — they refer to each other by
position. Never mix a copy of one with an older copy of another; if you rebuild,
rebuild all three. Rebuilding is possible (`Encode_corpus_colab.ipynb`) but
wants a GPU and several hours, which is why the Drive holds a ready-made copy.

## What the app makes by itself

These are not to be downloaded and can be deleted at any time — they are
recomputed, at the cost of one slow start.

| File | What it is |
| --- | --- |
| `ica_embedding.pkl` | The ICA decomposition — the components the app is built around. This is what the long first boot is computing. |
| `umap_projection.pkl` | The 2-D positions of the speakers, i.e. the layout of the map itself. Computed right after the ICA. |
| `tasks.db` | The history of the runs launched from the *bulk* pages. Local bookkeeping, not project data: do not copy it to the Drive or between machines. Losing it loses the list of past runs and nothing else. |

Both `.pkl` files record the settings they were computed with and recompute
themselves if those settings change, so there is nothing to remember to
invalidate.
