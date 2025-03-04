import logging
from pathlib import Path

from insta_message_analyzer.core import MessageLoader, get_logger, setup_logging

setup_logging(log_level=logging.INFO, log_file="../output/insta_analyzer.log")
logger = get_logger(__name__)

def main() -> None:
    # Resolve the project root (parent of src/) from main.py's location
    project_root = Path(__file__).parent.parent.resolve()
    root_dir = project_root / "data" / "your_instagram_activity" / "messages"
    output_dir = project_root / "output"
    output_path = output_dir / "messages_raw.csv"

    # Create output directory if it doesn’t exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    loader = MessageLoader(root_dir)
    df = loader.get_messages
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    main()
