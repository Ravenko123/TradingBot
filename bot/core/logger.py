"""Centralized logging helpers for the ICT SMC bot."""

from __future__ import annotations

import logging
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Dict

from config.settings import SETTINGS


_LOGGER_CACHE: Dict[str, Logger] = {}


def _build_handler(path: Path, level: int) -> TimedRotatingFileHandler:
    """Create a rotating file handler for the given log file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    handler = TimedRotatingFileHandler(path, when="midnight", backupCount=7, utc=True)
    handler.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    return handler


def configure_logging() -> None:
    """Configure root logging handlers once per application run."""

    if _LOGGER_CACHE:
        return

    log_files = {
        "system": SETTINGS.logs_dir / "system.log",
        "trades": SETTINGS.logs_dir / "trades.log",
        "errors": SETTINGS.logs_dir / "errors.log",
    }

    for name, path in log_files.items():
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO if name != "errors" else logging.ERROR)
        logger.propagate = False
        # Avoid duplicate handlers when reloading
        logger.handlers.clear()
        handler_level = logging.INFO if name != "errors" else logging.ERROR
        logger.addHandler(_build_handler(path, handler_level))
        _LOGGER_CACHE[name] = logger


def get_logger(name: str) -> Logger:
    """Return a configured logger, ensuring configuration happens once."""

    if not _LOGGER_CACHE:
        configure_logging()
    if name in _LOGGER_CACHE:
        return _LOGGER_CACHE[name]
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers.clear()
    logger.addHandler(_build_handler(SETTINGS.logs_dir / f"{name}.log", logging.INFO))
    _LOGGER_CACHE[name] = logger
    return logger


__all__ = ["configure_logging", "get_logger"]