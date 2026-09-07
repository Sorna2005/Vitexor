"""
logger.py

Enterprise logging configuration for the AI Code Reviewer.

This module provides a centralized logging system with support for:

- Console logging
- File logging
- Rotating log files
- Custom formatting
- Singleton logger creation
- Automatic log directory creation

Every module in the project should obtain its logger from this module.
"""

from __future__ import annotations

###############################################################################
# Standard Library Imports
###############################################################################

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

###############################################################################
# Local Imports
###############################################################################

from .config import (
    LOGS_DIR,
    get_log_level,
)

###############################################################################
# Logger Constants
###############################################################################

LOGGER_NAME = "AI_Code_Reviewer"

LOG_FILE_NAME = "application.log"

MAX_LOG_FILE_SIZE = 10 * 1024 * 1024

BACKUP_COUNT = 5

LOG_FORMAT = (
    "%(asctime)s | "
    "%(levelname)-8s | "
    "%(name)s | "
    "%(filename)s:%(lineno)d | "
    "%(message)s"
)

DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

###############################################################################
# Logger Factory
###############################################################################


class LoggerFactory:
    """
    Factory class responsible for creating and configuring loggers.

    This class ensures that every logger is configured only once.
    """

    _configured = False

    @classmethod
    def configure(cls) -> None:
        """
        Configure the root application logger.
        """

        if cls._configured:
            return

        LOGS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        formatter = logging.Formatter(
            fmt=LOG_FORMAT,
            datefmt=DATE_FORMAT,
        )

        console_handler = logging.StreamHandler()

        console_handler.setFormatter(
            formatter
        )

        file_handler = RotatingFileHandler(
            filename=LOGS_DIR / LOG_FILE_NAME,
            maxBytes=MAX_LOG_FILE_SIZE,
            backupCount=BACKUP_COUNT,
            encoding="utf-8",
        )

        file_handler.setFormatter(
            formatter
        )

        logger = logging.getLogger(
            LOGGER_NAME
        )

        logger.setLevel(
            get_log_level()
        )

        logger.handlers.clear()

        logger.addHandler(
            console_handler
        )

        logger.addHandler(
            file_handler
        )

        logger.propagate = False

        cls._configured = True

###############################################################################
# Public Logger
###############################################################################


def get_logger(
    name: str | None = None,
) -> logging.Logger:
    """
    Return a configured logger.

    Parameters
    ----------
    name : str | None

    Returns
    -------
    logging.Logger
    """

    LoggerFactory.configure()

    if not name:
        return logging.getLogger(
            LOGGER_NAME
        )

    return logging.getLogger(
        f"{LOGGER_NAME}.{name}"
    )

###############################################################################
# Logging Helpers
###############################################################################


def log_start(
    logger: logging.Logger,
    task: str,
) -> None:
    """
    Log task start.
    """

    logger.info(
        "=" * 80
    )

    logger.info(
        "START : %s",
        task,
    )


def log_end(
    logger: logging.Logger,
    task: str,
) -> None:
    """
    Log task completion.
    """

    logger.info(
        "END   : %s",
        task,
    )

    logger.info(
        "=" * 80
    )


def log_exception(
    logger: logging.Logger,
    exception: Exception,
) -> None:
    """
    Log an exception.
    """

    logger.exception(
        "%s",
        exception,
    )


###############################################################################
# Initialize Logger
###############################################################################

LoggerFactory.configure()

###############################################################################
# Public Exports
###############################################################################

__all__ = [
    "LoggerFactory",
    "get_logger",
    "log_start",
    "log_end",
    "log_exception",
]