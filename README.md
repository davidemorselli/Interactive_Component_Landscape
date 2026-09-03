# Interactive Component Landscape

An interactive map of the ICA components of a tweet corpus: a Flask app that
draws each component's landscape (UMAP projection, strong/extreme words and
speakers), retrieves the tweets closest to any word, and asks LLM agents to
write, grade and refine short narratives about what a word means in the corpus.

## Run

The data files (embedding, training data, labels — see `src/config.py`) sit
flat in one directory, `data/`, see [docs/DATA.md](docs/DATA.md). LLM calls need an `OPENROUTER_API_KEY`, see
`.env.example`.

**It is very easy to run the website locally using the following command:**

```sh
uv sync
cd src && uv run waitress-serve --port=8050 webapp.webapp:app
```

Or with Docker: `docker compose up` and open http://localhost:8050. The first
boot fits ICA and UMAP and takes 15–20 minutes; later boots take ~25 s.

To put it on the internet, with a login in front of it, see
[docs/DEPLOY.md](docs/DEPLOY.md).

The notebooks under `src/` are the other front-end onto the same analysis:
`Component Landscape.ipynb` and `Tweet Narratives.ipynb` drive through
ipywidgets what the web app drives through query parameters. They run locally
against the same `DATA_DIR` — their Colab setup cell skips itself when it does
not find Colab. `Encode_corpus_colab.ipynb` is the exception: it rebuilds the
tweet vectors and wants a GPU.

## Documentation

Three markdown files at the root of the repository, read in place:

- **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — what the app does and how it is put
  together: the four packages, the pages, the write/grade/rewrite pipeline, and
  the handful of things that look arbitrary but are load-bearing.
- **[docs/DATA.md](docs/DATA.md)** — every file in `DATA_DIR`: what it is, where it comes
  from, and which ones the app recreates by itself.
- **[docs/DEPLOY.md](docs/DEPLOY.md)** — renting a server, deploying, and putting it
  behind Cloudflare with access control.
