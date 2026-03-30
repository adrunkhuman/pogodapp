# Project Rules

## Architecture

- Keep one FastAPI app as the only runtime process.
- FastAPI serves page render, static assets, and API routes.
- Use Jinja2 only for the initial page render.
- Use HTMX only for form submission and server-driven interaction.

## API Contract

- `POST /score` accepts standard form fields through FastAPI `Form()` parameters.
- `POST /score` returns raw JSON, not HTML fragments.
- The response contract is a JSON array of objects shaped as `{lat, lon, score}`.
- `score` stays normalized to the `0..1` range.

## Frontend Boundaries

- Keep `frontend/static/map.js` render-only.
- Do not add custom fetch or XHR code to `frontend/static/map.js`.
- Use `htmx:afterRequest` as the handoff from form submission to map rendering.
- Keep the page as a single-screen app with controls and map on one route.

## Backend Boundaries

- Keep route handlers thin.
- Keep scoring logic isolated from FastAPI routing code.
- Keep DuckDB access behind a small adapter or repository boundary.
- Do not leak SQL concerns into templates or frontend code.

## Data Direction

- Start with the `0.5deg` dataset.
- Preserve compatibility with a later swap to `10'` resolution.
- Keep the climate row schema stable unless there is a deliberate migration.
- Treat `climate.duckdb` distribution strategy as unresolved until explicitly decided.

## Delivery

- Prefer small issue-sized changes.
- Preserve compatibility with the existing GitHub issue backlog.
- Do not reintroduce a local-only planning document; keep committed intent in `README.md` and implementation work in GitHub issues.
