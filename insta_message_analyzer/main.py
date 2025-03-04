"""
Main entry point for the Instagram Message Analyzer.

This script orchestrates the loading and preprocessing of Instagram message data,
saving the results to a CSV file.
"""

import logging
from pathlib import Path

from insta_message_analyzer.data import MessageLoader, MessagePreprocessor
from insta_message_analyzer.utils import get_logger, setup_logging

setup_logging(log_level=logging.INFO, log_file="../output/insta_analyzer.log")
logger = get_logger(__name__)

def main() -> None:
    """
    Main function to load, process, and save Instagram message data.

    This function resolves the project root, sets up input and output directories,
    loads raw message data, processes it into a DataFrame, and saves the result to a CSV file.
    """
    # Resolve the project root (parent of insta_message_analyzer/) from main.py's location
    project_root = Path(__file__).parent.parent.resolve()
    root_dir = project_root / "data" / "your_instagram_activity" / "messages"
    output_dir = project_root / "output"
    output_path = output_dir / "messages_raw.csv"

    # Create output directory if it doesn’t exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    loader = MessageLoader(root_dir)
    raw_data = loader.get_raw_data

    # Process data
    preprocessor = MessagePreprocessor(raw_data)
    df = preprocessor.get_processed_data

    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info("Saved processed messages to %s", output_path)

if __name__ == "__main__":
    main()
