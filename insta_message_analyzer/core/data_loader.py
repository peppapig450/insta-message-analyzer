"""
Module for loading and preprocessing Instagram message data from JSON files.

This module provides the `MessageLoader` class, which loads Instagram message data
from JSON files, processes them to determine chat types (group or DM), and stores
the data in a Pandas DataFrame.
"""

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

from .logging_config import get_logger


class MessageLoader:
    """
    Loads and preprocesses Instagram message data from JSON files.

    This class scans a root directory containing Instagram message JSON files,
    processes them to determine chat types (group or DM), and stores the extracted
    message data in a Pandas DataFrame.

    Attributes
    ----------
    root_dir : Path
        Path to the root directory containing Instagram message data.
    logger : logging.Logger
        Logger instance for logging messages and errors.
    messages_df : pd.DataFrame
        DataFrame containing processed message data with columns:
        ['chat_id', 'sender', 'timestamp', 'content', 'chat_type'].

    Methods
    -------
    is_group_chat(chat_data)
        Determines if a chat is a group chat based on JSON data.
    preprocess_chat(chat_dir)
        Loads and processes message JSON files from a chat directory.
    load_data()
        Loads Instagram messages into a Pandas DataFrame.
    get_messages
        Returns the processed messages DataFrame.

    """

    # Precompile regex for group-related keywords
    GROUP_PATTERN = re.compile(
    r"created the group|added to the group|left the group", re.IGNORECASE
)

    # Minimum number of participants/senders for a group chat
    GROUP_THRESHOLD: int = 2

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
        self.messages_df = pd.DataFrame()
        self.load_data()


    def is_group_chat(self, chat_data: list[dict]) -> bool:
        """
        Determine if a chat is a group chat based on consolidated JSON data.

        Parameters
        ----------
        chat_data : list of dict
            List of JSON data from all message_X.json files for a chat.

        Returns
        -------
        bool
            True if the chat is a group chat, False if it's a personal DM.

        """
        unique_senders: set[str] = set()  # Set to track unique senders efficiently

        for data in chat_data:
            # Check for "joinable_mode" - a definitive group chat indicator
            if "joinable_mode" in data:
                self.logger.debug("Chat identified as group due to 'joinable_mode' presence")
                return True

            # Check participant count - more than 10 indicates a sizable group
            if (participants := data.get("participants")) and len(
                participants
            ) >= self.GROUP_THRESHOLD:
                self.logger.debug(
                    "Chat identified as group due to %d participants", len(participants)
                )
                return True

            # Process messages incrementally
            for message in data.get("messages", []):
                # Add sender to set if present
                if sender := message.get("sender_name"):
                    unique_senders.add(sender)
                    if len(unique_senders) >= self.GROUP_THRESHOLD:
                        self.logger.debug(
                            "Chat identified as group due to %d unique senders",
                            len(unique_senders),
                        )
                        return True

                # Check for group-related actions in message content
                if self.GROUP_PATTERN.search(message.get("content", "")):
                    self.logger.debug("Chat identified as group due to group action in content")
                    return True

        # If no group indicators found and ≤self.GROUP_THRESHOLD unique senders, it’s a personal DM
        return False


    def preprocess_chat(self, chat_dir: Path) -> tuple[str, list[dict], str]:
        """
        Load all message_X.json files from a chat directory.

        Parameters
        ----------
        chat_dir : Path
            Path to a chat directory.

        Returns
        -------
        tuple
            A tuple containing:
            - chat_id (str): Chat ID.
            - chat_data (list of dict): List of JSON data.
            - chat_type (str): Type of chat ('group' or 'dm').

        """
        json_files = sorted(chat_dir.glob("message_*.json"))
        chat_data: list[dict] = []

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

        chat_id = chat_dir.name
        chat_type = "group" if self.is_group_chat(chat_data) else "dm"
        self.logger.info("Processed chat %s as %s", chat_id, chat_type)
        return chat_id, chat_data, chat_type


    def load_data(self) -> None:
        """
        Load Instagram messages from the inbox directory into a Pandas DataFrame.

        This method scans the inbox directory for chat folders, processes each chat,
        and stores the extracted messages in a Pandas DataFrame.
        """
        inbox_path = self.root_dir / "inbox"
        messages = []

        if not inbox_path.is_dir():
            self.logger.error("No inbox found at %s", inbox_path)
            return

        for chat_dir in inbox_path.iterdir():
            if not chat_dir.is_dir():
                continue

            chat_id, chat_data, chat_type = self.preprocess_chat(chat_dir)
            for data in chat_data:
                for msg in data.get("messages", []):
                    sender = msg.get("sender_name")
                    timestamp = msg.get("timestamp_ms")
                    if sender and timestamp:  # Basic validation
                        # Decode sender name
                        sender = sender.encode("latin1").decode("utf-8", errors="replace")
                        sender = unicodedata.normalize("NFC", sender)

                        # Decode content
                        content = msg.get("content", "")
                        content = content.encode("latin1").decode("utf-8", errors="replace")
                        content = unicodedata.normalize("NFC", content)

                        messages.append(
                            {
                                "chat_id": chat_id,
                                "sender": sender,
                                "timestamp": pd.to_datetime(timestamp, unit="ms"),
                                "content": content,
                                "chat_type": chat_type,
                            }
                        )
        if not messages:
            self.logger.warning("No valid messages found")

        # NOTE: in future .from_records provides better performance over 100k+ rows
        self.messages_df = pd.DataFrame(messages)
        self.logger.info("Loaded %d messages into DataFrame", len(self.messages_df))

    @property
    def get_messages(self) -> pd.DataFrame:
        """
        Return the loaded messages DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing loaded Instagram messages.

        """
        return self.messages_df
