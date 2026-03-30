# Pogodapp

Scaffold for Climate Matcher: a FastAPI app that serves a Jinja2-rendered page, static assets, and a scoring API backed by DuckDB.

## Stack

- Python 3.14
- FastAPI
- Jinja2
- HTMX
- DuckDB
- Ruff
- ty
- pytest

## Setup

```bash
uv sync
```

## Quality checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run ty check
uv run pytest
```

## Notes

- `PLAN.md` is intentionally local-only and ignored by git.
- App implementation is not started yet; this repo currently contains scaffold and tooling only.
