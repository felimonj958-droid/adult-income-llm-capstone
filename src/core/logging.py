from __future__ import annotations

import logging
import os

LOGGER_NAME = "adult_income_api"
DEFAULT_LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def _resolve_log_level(level: int | str | None = None) -> int:
    if isinstance(level, int):
        return level

    value = level or os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL)
    if isinstance(value, str):
        resolved = logging.getLevelName(value.upper())
        if isinstance(resolved, int):
            return resolved

    return logging.INFO


def configure_logging(level: int | str | None = None) -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    resolved_level = _resolve_log_level(level)

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(LOG_FORMAT))
        logger.addHandler(handler)

    logger.setLevel(resolved_level)
    logger.propagate = False

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(level=resolved_level, format=LOG_FORMAT)

    return logger


logger = logging.getLogger(LOGGER_NAME)
