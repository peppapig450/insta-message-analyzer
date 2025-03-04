"""
Centralized logging configuration and utilities.

This module provides functions to set up and retrieve logging instances for an
application, ensuring consistent log formatting and output to both console and file.

Attributes
----------
setup_logging : function
    Configures the root logger with console and file handlers.
get_logger : function
    Returns a configured logger instance for a given module name.

"""

import logging
from pathlib import Path


def setup_logging(
    logger_name: str = "insta_analyzer",
    log_level: int = logging.INFO,
    log_file: str | Path = "insta_analyzer.log"
) -> logging.Logger:
    """
    Configure a logger with console and file handlers.

    Sets up a logger with handlers for both console and file output, ensuring the log file's directory exists.
    Existing handlers are cleared to prevent duplication.

    Parameters
    ----------
    logger_name : str, optional
        Name of the logger. Default is "insta_analyzer".
    log_level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
    log_file : str or Path, optional
        Path to the log file. Can be relative or absolute. Default is "insta_analyzer.log".

    Returns
    -------
    logging.Logger
        A configured logger instance.

    Notes
    -----
    The log format is fixed as "%(asctime)s - %(name)s - %(levelname)s - %(message)s".

    """
    # Get or create the logger
    logger = logging.getLogger(logger_name)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Set the logging level
    logger.setLevel(log_level)

    # Convert log_file to a Path object and ensure the directory exists
    log_path = Path(log_file).resolve()
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance for a module.

    Returns a logger configured by the application's centralized logging setup.

    Parameters
    ----------
    name : str
        Name of the module (e.g., __name__) to associate with the logger.

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
    return logging.getLogger(name)
