from __future__ import annotations

import argparse

import uvicorn

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def parse_args() -> argparse.Namespace:
    """Parse local launcher options."""
    parser = argparse.ArgumentParser(description="Run Pogodapp locally.")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface for the local server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port for the local server.")
    parser.add_argument(
        "--no-reload",
        action="store_true",
        help="Disable code reload while developing locally.",
    )
    return parser.parse_args()


def main() -> None:
    """Launch the local app server."""
    args = parse_args()
    uvicorn.run("backend.main:app", host=args.host, port=args.port, reload=not args.no_reload)


if __name__ == "__main__":
    main()
