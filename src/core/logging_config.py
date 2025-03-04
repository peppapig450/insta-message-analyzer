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


def setup_logging(log_level: int = logging.INFO, log_file: str = "insta_analyzer.log") -> None:
    """
    Configure centralized logging for the application.

    Sets up the root logger with handlers for both console and file output, using a
    consistent log format. Existing handlers are cleared to prevent duplication.

    Parameters
    ----------
    log_level : int, optional
        Logging level (e.g., logging.INFO, logging.DEBUG). Default is logging.INFO.
    log_file : str, optional
        Path to the log file. Default is "insta_analyzer.log".

    Notes
    -----
    The log format is fixed as "%(asctime)s - %(name)s - %(levelname)s - %(message)s".
    This function modifies the root logger globally.

    Examples
    --------
    >>> import logging
    >>> setup_logging(log_level=logging.DEBUG, log_file="app.log")
    >>> logger = logging.getLogger("example")
    >>> logger.debug("This is a debug message")

    """
    # Clear any existing handlers to avoid duplicates
    logging.getLogger().handlers = []

    # Define log format
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)

    # Configure root logger
    logging.basicConfig(level=log_level, handlers=[console_handler, file_handler])


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
