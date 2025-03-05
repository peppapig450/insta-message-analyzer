"""
Centralized logging configuration and utilities.

This module provides functions to configure and retrieve logging instances for the
Instagram Message Analyzer, ensuring consistent log formatting and output to both
console and file.

Functions
---------
setup_logging
    Configures the root logger with console (INFO) and file (DEBUG) handlers.
get_logger
    Returns a configured logger instance for a given module name.
"""

from __future__ import annotations

import logging
import logging.config
from pathlib import Path
from typing import Literal

# Default logging configuration
DEFAULT_LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
DEFAULT_LOGGER_NAME: str = "insta_message_analyzer"
DEFAULT_LOG_FILE: Path = Path("insta_message_analyzer.log")
DEFAULT_CONSOLE_LEVEL: int = logging.INFO
DEFAULT_FILE_LEVEL: int = logging.DEBUG


def setup_logging(
    logger_name: str = DEFAULT_LOGGER_NAME,
    console_level: int = DEFAULT_CONSOLE_LEVEL,
    file_level: int = DEFAULT_FILE_LEVEL,
    log_file: str | Path = DEFAULT_LOG_FILE,
    log_format: str = DEFAULT_LOG_FORMAT,
) -> None:
    """
    Configure the application's logging with console and file handlers.

    Sets up a logger with a console handler at the specified level (default INFO)
    and a file handler at DEBUG level. Ensures the log file's directory exists.

    Parameters
    ----------
    logger_name : str, optional
        Name of the logger. Default is "insta_message_analyzer".
    console_level : int, optional
        Logging level for console output (e.g., logging.INFO). Default is INFO.
    file_level : int, optional
        Logging level for file output (e.g., logging.DEBUG). Default is DEBUG.
    log_file : str | Path, optional
        Path to the log file. Default is "insta_message_analyzer.log".
    log_format : str, optional
        Format string for log messages. Default is "%(asctime)s - %(name)s - %(levelname)s - %(message)s".

    Notes
    -----
    - Handlers are only added if not already present to avoid duplication.
    - The root logger is not modified; a named logger is configured instead.
    """
    # Convert log_file to Path and ensure directory exists
    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the logger (don't modify the root logger)
    logger = logging.getLogger(logger_name)

    # Avoid duplicate handlers by checking if already configured
    if not logger.hasHandlers():
        # Set the logger's level to the most verbose of the handlers
        logger.setLevel(min(console_level, file_level))

        # Formatter
        formatter = logging.Formatter(log_format)

        # Console handler (INFO by default)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(console_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler (DEBUG by default)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        logger.debug(
            "Logging configured with console (%s) and file (%s) handlers",
            logging.getLevelName(console_level),
            logging.getLevelName(file_level),
        )
    else:
        logger.debug("Logger %s already has handlers: %s", logger_name, logger.handlers)


def get_logger(name: str, level: int | Literal["NOTSET"] = logging.NOTSET) -> logging.Logger:
    """
    Get a logger instance for a module.

    Returns a logger configured by the application's centralized logging setup.
    If setup_logging hasn't been called, it will use default settings.

    Parameters
    ----------
    name : str
        Name of the module (e.g., __name__) to associate with the logger.
    level : int | Literal["NOTSET"], optional
        Override the logger's level (e.g., logging.DEBUG). Default is NOTSET,
        which respects the parent logger's level.

    Returns
    -------
    logging.Logger
        A configured logger instance for the specified module name.

    Examples
    --------
    >>> setup_logging(log_level=logging.INFO)
    >>> logger = get_logger(__name__)
    >>> logger.info("This is an info message")

    """
    logger = logging.getLogger(name)
    if level != logging.NOTSET:
        logger.setLevel(level)
    return logger
