"""
Module for preprocessing Instagram message data.

This module provides the `MessagePreprocessor` class, which processes raw Instagram message
data to determine chat types (group or DM), decodes message content, and stores the data
in a Pandas DataFrame.
"""

import re
import unicodedata

import pandas as pd

from ..utils.logging import get_logger


class MessagePreprocessor:
    """
    Preprocesses Instagram message data from raw JSON data.

    This class takes raw JSON data, determines chat types (group or DM), decodes sender names
    and message content, and stores the processed data in a Pandas DataFrame.

    Attributes
    ----------
    raw_data : dict
        Dictionary keyed by chat ID (int), with values as {'data': list of JSON, 'chat_name': str}.
    logger : logging.Logger
        Logger instance for logging messages and errors.
    messages_df : pd.DataFrame
        DataFrame containing processed message data with columns:
        ['chat_id', 'chat_name', 'sender', 'timestamp', 'content', 'chat_type', 'reactions_list'].

    Methods
    -------
    is_group_chat(chat_data)
        Determines if a chat is a group chat based on JSON data.
    process_data()
        Processes raw JSON data into a Pandas DataFrame.
    get_processed_data()
        Returns the processed messages DataFrame.

    """

    # Precompile regex for group-related keywords
    GROUP_PATTERN = re.compile(r"created the group|added to the group|left the group", re.IGNORECASE)

    # Minimum number of participants/senders for a group chat
    GROUP_THRESHOLD: int = 5

    def __init__(self, raw_data: dict) -> None:
        """
        Initialize the MessagePreprocessor.

        Parameters
        ----------
        raw_data : dict
            Dictionary keyed by chat ID (int), with values as {'data': list of JSON, 'chat_name': str}.

        """
        self.raw_data = raw_data  # NOTE: TypedDict maybe?
        self.logger = get_logger(__name__)
        self.messages_df = pd.DataFrame()
        self.logger.debug("Initialized MessagePreprocessor with raw_data keys: %s", list(raw_data.keys()))
        self.process_data()

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
        unique_senders: set[str] = set()
        self.logger.debug("Checking if chat is group, data length: %d", len(chat_data))

        for data in chat_data:
            # Check for "joinable_mode" - a definitive group chat indicator
            if "joinable_mode" in data:
                self.logger.debug("Chat identified as group due to 'joinable_mode' presence")
                return True

            # Check participant count
            if (participants := data.get("participants")) and len(participants) >= self.GROUP_THRESHOLD:
                self.logger.debug("Chat identified as group due to %d participants", len(participants))
                return True

            # Process messages incrementally
            for message in data.get("messages", []):
                if sender := message.get("sender_name"):
                    unique_senders.add(sender)
                    if len(unique_senders) >= self.GROUP_THRESHOLD:
                        self.logger.debug(
                            "Chat identified as group due to %d unique senders", len(unique_senders)
                        )
                        return True

                if self.GROUP_PATTERN.search(message.get("content", "")):
                    self.logger.debug("Chat identified as group due to group action in content")
                    return True

        return False

    def process_data(self) -> None:
        """
        Process raw JSON data into a Pandas DataFrame.

        This method processes the raw data to determine chat types, decodes sender names and content,
        and stores the extracted messages in a Pandas DataFrame.
        """
        messages = []

        for chat_id, chat_info in self.raw_data.items():
            chat_data = chat_info["data"]
            preliminary_chat_name = chat_info["chat_name"]
            chat_type = "group" if self.is_group_chat(chat_data) else "dm"

            # Extract title from the first file for this chat, fallback to preliminary name
            chat_name = (
                chat_data[-1].get("title", preliminary_chat_name) if chat_data else preliminary_chat_name
            )

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

                        # Process reactions
                        reactions = msg.get("reactions", [])
                        reaction_list = [
                            (
                                unicodedata.normalize(
                                    "NFC",
                                    reaction["reaction"].encode("latin1").decode("utf-8", errors="replace"),
                                ),
                                unicodedata.normalize(
                                    "NFC",
                                    reaction["actor"].encode("latin1").decode("utf-8", errors="replace"),
                                ),
                            )
                            for reaction in reactions
                        ]

                        messages.append(
                            {
                                "chat_id": int(chat_id),  # Numeric ID from directory
                                "chat_name": chat_name,  # Single name per chat from JSON or directory
                                "sender": sender,
                                "timestamp": pd.to_datetime(timestamp, unit="ms"),
                                "content": content,
                                "chat_type": chat_type,
                                "reactions": reaction_list,
                            }
                        )

        if not messages:
            self.logger.warning("No valid messages found")

        # NOTE: in future .from_records provides better performance over 100k+ rows
        self.messages_df = pd.DataFrame(messages)
        self.logger.info(
            "Processed %d messages into DataFrame, columns: %s",
            len(self.messages_df),
            self.messages_df.columns.tolist(),
        )

    @property
    def get_processed_data(self) -> pd.DataFrame:
        """
        Return the processed messages DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing processed Instagram messages with columns:
            ['chat_id', 'chat_name', 'sender', 'timestamp', 'content', 'chat_type', 'reactions_list'].

        """
        return self.messages_df
