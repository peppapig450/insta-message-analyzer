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

# Define project root and log path
project_root = Path(__file__).parent.parent.resolve()
log_path = project_root / "output" / "logs" / "insta_analyzer.log"

# Configure logging for the package
setup_logging(
    logger_name="insta_message_analyzer",
    console_level=logging.DEBUG,
    file_level=logging.DEBUG,
    log_file=log_path,
)

# Retrieve logger for this module
logger = get_logger(__name__)


def main() -> None:
    """
    Main function to load, process, analyze, and visualize Instagram message data.

    This function resolves the project root, sets up input and output directories,
    loads raw message data, processes it into a DataFrame, runs analysis strategies,
    and generates visualizations, saving all results to the output directory.
    """
    # Resolve project root and set up logging
    logger.debug("Starting main execution")
    root_dir = project_root / "data" / "your_instagram_activity" / "messages"
    output_dir = project_root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.debug("Loading raw data from %s", root_dir)
    try:
        loader = MessageLoader(root_dir)
        raw_data = loader.get_raw_data
        logger.debug("Raw data loaded, length: %d", len(raw_data))
    except Exception:
        logger.exception("Failed to load raw data: %s")
        raise

    logger.debug("Preprocessing raw data")
    try:
        preprocessor = MessagePreprocessor(raw_data)
        df = preprocessor.get_processed_data
        logger.debug("Processed data shape: %s, columns: %s", df.shape, df.columns.tolist())
        if df.empty:
            logger.warning("Processed DataFrame is empty; analysis will produce no results")
        if "timestamp" not in df.columns:
            logger.error("Processed DataFrame lacks 'timestamp' column; analysis will fail")
    except Exception:
        logger.exception("Failed to preprocess data: %s")
        raise

    raw_messages_out = output_dir / "messages_raw.csv"
    df.to_csv(raw_messages_out, index=False)
    logger.info("Saved processed messages to %s", raw_messages_out)

    logger.debug("Running analysis pipeline")
    strategies = [ActivityAnalysis()]
    pipeline = AnalysisPipeline(strategies)
    try:
        results = pipeline.run_analysis(df)
        logger.debug(
            "Analysis results: %s",
            {
                k: len(v["time_series"]["counts"]) if "time_series" in v else "Empty"
                for k, v in results.items()
            },
        )
        pipeline.save_results(results, output_dir)
        logger.info("Saved analysis results to %s", output_dir)
    except Exception:
        logger.exception("Analysis pipeline failed: %s")
        raise

    logger.debug("Generating visualizations")
    try:
        plotter = TimeSeriesPlotter(results, output_dir)
        plotter.plot()
        logger.info("Generated visualizations in %s", output_dir)
    except Exception:
        logger.exception("Visualization failed: %s")
        raise


if __name__ == "__main__":
    main()
