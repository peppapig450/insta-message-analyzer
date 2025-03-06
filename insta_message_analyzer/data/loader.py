"""
Module for loading raw Instagram message data from JSON files.

This module provides the `MessageLoader` class, which scans a root directory containing
Instagram message JSON files and loads the raw data into a dictionary for further processing.
"""

import json
import re
from pathlib import Path
from typing import Any

from ..utils.logging import get_logger


class MessageLoader:
    """
    Loads raw Instagram message data from JSON files.

    This class scans a root directory containing Instagram message JSON files and loads
    the raw data into a dictionary, where keys are chat IDs and values are lists of JSON data
    from message_X.json files.

    Attributes
    ----------
    root_dir : Path
        Path to the root directory containing Instagram message data.
    logger : logging.Logger
        Logger instance for logging messages and errors.
    raw_data : dict
        Dictionary containing raw JSON data, keyed by chat ID.

    Methods
    -------
    load_raw_data()
        Loads raw JSON data from the inbox directory into a dictionary.
    get_raw_data()
        Returns the loaded raw JSON data.

    """

    # Regex to split directory name into chat name and numeric ID
    DIR_PATTERN = re.compile(r"^(.*?)(?:_(\d+))?$")

    def __init__(self, root_dir: Path) -> None:
        """
        Initialize the MessageLoader.

        Parameters
        ----------
        root_dir : Path
            Path to the root directory containing Instagram message data.

        """
        self.root_dir = Path(root_dir)
        self.logger = get_logger(__name__)
        self.raw_data: dict[str, Any] = {}
        self.load_raw_data()

    def load_raw_data(self) -> None:
        """
        Load raw Instagram message data from the inbox directory into a dictionary.

        This method scans the inbox directory for chat folders and loads each chat's
        message_X.json files into a dictionary keyed by chat ID.
        """
        inbox_path = self.root_dir / "inbox"
        if not inbox_path.is_dir():
            self.logger.error("No inbox found at %s", inbox_path)
            return

        for chat_dir in inbox_path.iterdir():
            if not chat_dir.is_dir():
                continue

            directory_name = chat_dir.name
            json_files = sorted(chat_dir.glob("message_*.json"))
            chat_data = []

            # Extract chat name and ID from directory name
            if not (match := self.DIR_PATTERN.match(directory_name)):
                self.logger.warning("Could not parse directory name: %s", directory_name)
                continue

            dir_chat_name, dir_chat_id = match.groups()
            chat_id = dir_chat_id if dir_chat_id else directory_name  # Use full name if no ID
            chat_name = dir_chat_name if dir_chat_name else directory_name  # Use full name if no name

            for json_file in json_files:
                try:
                    with json_file.open("r", encoding="utf-8-sig") as file:
                        chat_data.append(json.load(file))
                except (OSError, json.JSONDecodeError, UnicodeDecodeError) as e:
                    self.logger.warning("Error with %s: %s. Trying binary fallback...", json_file, e)
                    try:
                        with json_file.open("rb") as file:
                            text = file.read().decode("utf-8", errors="replace")
                            chat_data.append(json.loads(text))
                    except Exception as e:
                        self.logger.exception("Skipping %s.", json_file)

            if chat_data:
                self.raw_data[chat_id] = {"data": chat_data, "chat_name": chat_name}
                self.logger.info(
                    "Loaded raw data for chat ID %s (preliminary name: '%s')", chat_id, chat_name
                )

        if not self.raw_data:
            self.logger.warning("No valid raw data found")
        else:
            self.logger.info("Loaded raw data for %d chats", len(self.raw_data))

    @property
    def get_raw_data(self) -> dict[str, Any]:
        """
        Return the loaded raw JSON data.

        Returns
        -------
        dict
            Dictionary containing raw JSON data, keyed by chat ID, with values as lists of JSON data.

        """
        return self.raw_data
