from __future__ import annotations

import logging

BACKEND_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


def configure_backend_logging() -> None:
    """Attach a stderr handler for `backend.*` logs when the app starts.

    Uvicorn configures its own logger tree but does not automatically route
    arbitrary application loggers like `backend.score_service`. This keeps app
    timing logs visible both with and without reload enabled.
    """
    backend_logger = logging.getLogger("backend")
    backend_logger.setLevel(logging.INFO)

    if backend_logger.handlers:
        return

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(BACKEND_LOG_FORMAT))
    backend_logger.addHandler(handler)
    backend_logger.propagate = False
