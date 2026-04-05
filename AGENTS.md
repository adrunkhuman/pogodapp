# Project Rules

## Architecture

- Keep one FastAPI app as the only runtime process.
- FastAPI serves page render, static assets, and API routes.
- Use Jinja2 only for the initial page render.
- Use HTMX only for form submission and server-driven interaction.

## API Contract

- `POST /score` accepts standard form fields through FastAPI `Form()` parameters.
- `POST /score` returns raw JSON, not HTML fragments.
- The response contract is `{"scores": [{"name", "country_code", "flag", "score"}, ...], "heatmap": "data:image/png;base64,..."}`.
- Empty or all-zero results return `{"scores": [], "heatmap": ""}`.
- `score` stays normalized to the `0..1` range.

## Frontend Boundaries

- Keep `frontend/static/map.js` render-only.
- Do not add custom fetch or XHR code to `frontend/static/map.js`.
- Use `htmx:afterRequest` as the handoff from form submission to map rendering.
- Keep the page as a single-screen app with controls, map, and ranked results on one route.
- Preserve the current delivery model unless deliberately changed: local MapLibre/world assets, HTMX loaded externally.

## Backend Boundaries

- Keep route handlers thin.
- Keep scoring logic isolated from FastAPI routing code.
- Keep DuckDB access behind a small adapter or repository boundary.
- Keep city ranking and heatmap rendering outside the route layer.
- Do not leak SQL concerns into templates or frontend code.

## Data Direction

- Current baseline is the native WorldClim `5m` dataset, not the old `0.5deg` placeholder path.
- Preserve compatibility with possible later moves to finer native resolutions if the product chooses that direction.
- Keep the climate row schema stable unless there is a deliberate migration.
- Treat `climate.duckdb` distribution strategy as unresolved until explicitly decided.

## Delivery

- Prefer small issue-sized changes.
- Preserve compatibility with the existing GitHub issue backlog.
- Keep committed intent in `README.md` and implementation work in GitHub issues.
- When docs drift from code, tests, or open issues, update the docs to match the live system first and leave unresolved strategy questions explicit.
