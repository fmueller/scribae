from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_LOGGER_NAME = "scribae"


def setup_logging(*, verbose: bool = False, log_file: Path | None = None) -> logging.Logger:
    """Configure Scribae logger handlers and level.

    Calling this function repeatedly is safe; existing handlers are replaced so
    duplicate records are not emitted.
    """

    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s")

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.DEBUG if verbose else logging.WARNING)
    logger.addHandler(stream_handler)

    resolved_log_file = log_file
    if resolved_log_file is None:
        raw_log_file = os.environ.get("SCRIBAE_LOG_FILE")
        if raw_log_file:
            resolved_log_file = Path(raw_log_file).expanduser()

    if resolved_log_file is not None:
        resolved_log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(resolved_log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
        logger.addHandler(file_handler)

    return logger


__all__ = ["setup_logging"]
