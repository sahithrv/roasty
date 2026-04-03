from __future__ import annotations

import logging


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure a simple process-wide logger for early development."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)

