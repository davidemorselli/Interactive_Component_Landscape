FROM python:3.13-slim

# Fonts the plots fall back to for emoji / CJK tokens (configured in
# ica/landscape.py configure_fonts); without them those glyphs render as
# empty boxes
RUN apt-get update && apt-get install -y --no-install-recommends \
        fonts-dejavu-core fonts-noto-core fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Monochrome "Noto Emoji" (SIL OFL), vendored in the repo: Debian only
# packages the color variant, which matplotlib cannot use, so without this
# file every render logs "Font family 'Noto Emoji' not found".
COPY docker/fonts/NotoEmoji-Regular.ttf /usr/local/share/fonts/

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=1
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

COPY src/ src/

# The data files (embedding, training data, labels, caches) are mounted here
ENV DATA_DIR=/data

WORKDIR /app/src
EXPOSE 8050
# webapp.webapp prints "Ready." once boot completes; the deploy workflow
# (.github/workflows/deploy.yml) greps the compose logs for that exact string
# to declare a deploy healthy. Keep both sides in sync.
CMD ["uv", "run", "--no-sync", "waitress-serve", "--host=0.0.0.0", "--port=8050", "--threads=4", "webapp.webapp:app"]
