"""Logging setup and intermediate statistics helpers for YAAT."""

import logging
import sys
from typing import Optional

import numpy as np

_LOGGER_NAME = "yaat"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the YAAT logger.

    Args:
        level: Logging level (e.g. logging.DEBUG, logging.INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s.%(funcName)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def get_logger() -> logging.Logger:
    """Get the YAAT logger (assumes setup_logging was already called)."""
    return logging.getLogger(_LOGGER_NAME)


def log_array_stats(
    name: str,
    arr: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Log summary statistics for a numpy array.

    Args:
        name: Descriptive name for the array.
        arr: The numpy array to summarize.
        logger: Logger instance. Uses default YAAT logger if None.
    """
    if logger is None:
        logger = get_logger()

    logger.info(
        "%s — shape=%s dtype=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
        name,
        arr.shape,
        arr.dtype,
        float(np.min(arr)),
        float(np.max(arr)),
        float(np.mean(arr)),
        float(np.std(arr)),
    )


def log_stage(stage_name: str, logger: Optional[logging.Logger] = None) -> None:
    """Log entry into a pipeline stage.

    Args:
        stage_name: Name of the pipeline stage.
        logger: Logger instance. Uses default YAAT logger if None.
    """
    if logger is None:
        logger = get_logger()
    logger.info("=" * 60)
    logger.info("STAGE: %s", stage_name)
    logger.info("=" * 60)
