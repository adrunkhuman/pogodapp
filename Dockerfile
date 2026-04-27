FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app

ENV PATH="/app/.venv/bin:$PATH" \
    PORT=8000 \
    POGODAPP_DATA_DIR=/app/data \
    POGODAPP_HOST=0.0.0.0 \
    POGODAPP_RELOAD=false \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project

COPY . .
RUN uv sync --locked --no-dev && mkdir -p /app/data

EXPOSE 8000

CMD ["pogodapp", "--no-reload"]
