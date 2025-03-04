import json
import re
from pathlib import Path
import unicodedata

from .logging_config import get_logger
import pandas as pd

logger = get_logger(__name__)

# Precompile regex for group-related keywords
GROUP_PATTERN = re.compile(
    r"created the group|added to the group|left the group", re.IGNORECASE
)

# Minimum number of participants/senders for a group chat
GROUP_THRESHOLD: int = 2



def is_group_chat(chat_data: list[dict]) -> bool:
    """
    Determine if a chat is a group chat based on consolidated JSON data.

    Args:
        chat_data (List[Dict]): List of JSON data from all message_X.json files for a chat.

    Returns:
        bool: True if group chat, False if personal DM.
    """
    unique_senders: set[str] = set()  # Set to track unique senders efficiently

    for data in chat_data:
        # Check for "joinable_mode" - a definitive group chat indicator
        if "joinable_mode" in data:
            logger.debug("Chat identified as group due to 'joinable_mode' presence")
            return True

        # Check participant count - more than 10 indicates a sizable group
        if (participants := data.get("participants")) and len(
            participants
        ) >= GROUP_THRESHOLD:
            logger.debug(
                "Chat identified as group due to %d participants", len(participants)
            )
            return True

        # Process messages incrementally
        for message in data.get("messages", []):
            # Add sender to set if present
            if sender := message.get("sender_name"):
                unique_senders.add(sender)
                if len(unique_senders) >= GROUP_THRESHOLD:
                    logger.debug(
                        "Chat identified as group due to %d unique senders",
                        len(unique_senders),
                    )
                    return True

            # Check for group-related actions in message content
            if GROUP_PATTERN.search(message.get("content", "")):
                logger.debug("Chat identified as group due to group action in content")
                return True

    # If no group indicators found and ≤GROUP_THRESHOLD unique senders, it’s a personal DM
    return False


def preprocess_chat(chat_dir: Path) -> tuple[str, list[dict], str]:
    """
    Load all message_X.json files from a chat directory.

    Args:
        chat_dir (Path): Path to a chat directory.

    Returns:
        tuple: (chat_id, list of JSON data, chat_type).
    """
    json_files = sorted(chat_dir.glob("message_*.json"))
    chat_data: list[dict] = []

    for json_file in json_files:
        try:
            with json_file.open("r", encoding="utf-8-sig") as file:
                chat_data.append(json.load(file))
        except (json.JSONDecodeError, UnicodeDecodeError, IOError) as e:
            logger.warning("Error with %s: %s. Trying binary fallback...", json_file, e)
            try:
                with json_file.open("rb") as file:
                    text = file.read().decode("utf-8", errors="replace")
                    chat_data.append(json.loads(text))
            except Exception as e:
                logger.exception("Skipping %s.", json_file)

    chat_id = chat_dir.name
    chat_type = "group" if is_group_chat(chat_data) else "dm"
    logger.info("Processed chat %s as %s", chat_id, chat_type)
    return chat_id, chat_data, chat_type


def load_messages(root_dir: Path) -> pd.DataFrame:
    """
    Load Instagram messages from inbox/ into a DataFrame.

    Args:
        root_dir (Path): Root directory (e.g., "your_instagram_activity/messages").

    Returns:
        pd.DataFrame: Columns [chat_id, sender, timestamp, content, chat_type].
    """
    inbox_path = root_dir / "inbox"
    messages = []

    if not inbox_path.is_dir():
        print(f"No inbox found at {inbox_path}")
        return pd.DataFrame()

    for chat_dir in inbox_path.iterdir():
        if not chat_dir.is_dir():
            continue

        chat_id, chat_data, chat_type = preprocess_chat(chat_dir)
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
        logger.warning("No valid messages found")

    # NOTE: in future .from_records provides better performance over 100k+ rows
    df = pd.DataFrame(messages)
    logger.info("Loaded %d messages into DataFrame", len(df))
    return df
