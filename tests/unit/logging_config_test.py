from __future__ import annotations

import logging
from pathlib import Path

from scribae.logging_config import setup_logging


def _reset_scribae_logger() -> logging.Logger:
    logger = logging.getLogger("scribae")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    return logger


def test_setup_logging_defaults_to_info_with_warning_stream_handler() -> None:
    _reset_scribae_logger()

    logger = setup_logging()

    assert logger.level == logging.INFO
    assert len(logger.handlers) == 1
    assert isinstance(logger.handlers[0], logging.StreamHandler)
    assert logger.handlers[0].level == logging.WARNING


def test_setup_logging_verbose_enables_debug_level() -> None:
    _reset_scribae_logger()

    logger = setup_logging(verbose=True)

    assert logger.level == logging.DEBUG
    assert any(handler.level == logging.DEBUG for handler in logger.handlers)


def test_setup_logging_can_write_to_file(tmp_path: Path) -> None:
    _reset_scribae_logger()
    log_file = tmp_path / "scribae.log"

    logger = setup_logging(log_file=log_file)
    logger.info("hello logging")
    for handler in logger.handlers:
        handler.flush()

    assert log_file.exists()
    assert "hello logging" in log_file.read_text(encoding="utf-8")


def test_setup_logging_is_idempotent_for_handlers(tmp_path: Path) -> None:
    _reset_scribae_logger()
    log_file = tmp_path / "scribae.log"

    logger = setup_logging(log_file=log_file)
    first = len(logger.handlers)
    logger = setup_logging(log_file=log_file)
    second = len(logger.handlers)

    assert first == second
