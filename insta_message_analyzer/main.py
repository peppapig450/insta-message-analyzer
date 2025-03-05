"""
Main entry point for the Instagram Message Analyzer.

This script orchestrates the loading and preprocessing of Instagram message data,
saving the results to a CSV file.
"""

import logging
from pathlib import Path

from insta_message_analyzer.analysis import ActivityAnalysis, AnalysisPipeline
from insta_message_analyzer.data import MessageLoader, MessagePreprocessor
from insta_message_analyzer.utils import get_logger, setup_logging
from insta_message_analyzer.visualization import TimeSeriesPlotter

# Resolve project root and set up logging
project_root = Path(__file__).parent.parent.resolve()
log_path = project_root / "output" / "logs" / "insta_analyzer.log"
setup_logging(log_level=logging.INFO, log_file=log_path)
logger = get_logger(__name__)

def main() -> None:
    """
    Main function to load, process, analyze, and visualize Instagram message data.

    This function resolves the project root, sets up input and output directories,
    loads raw message data, processes it into a DataFrame, runs analysis strategies,
    and generates visualizations, saving all results to the output directory.
    """
    root_dir = project_root / "data" / "your_instagram_activity" / "messages"
    output_dir = project_root / "output"

    # Create output directory if it doesn’t exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load raw data
    loader = MessageLoader(root_dir)
    raw_data = loader.get_raw_data

    # Process data
    preprocessor = MessagePreprocessor(raw_data)
    df = preprocessor.get_processed_data
    raw_messages_out = output_dir / "messages_raw.csv"
    df.to_csv(raw_messages_out, index=False)
    logger.info("Saved processed messages to %s", raw_messages_out)

    # Run analysis pipeline
    strategies = [ActivityAnalysis()]
    pipeline = AnalysisPipeline(strategies)
    results = pipeline.run_analysis(df)
    pipeline.save_results(results, output_dir)
    logger.info("Saved analysis results to %s", output_dir)

    # Visualize results
    plotter = TimeSeriesPlotter(results, output_dir)
    plotter.plot()
    logger.info("Generated visualizations in %s", output_dir)



if __name__ == "__main__":
    main()
