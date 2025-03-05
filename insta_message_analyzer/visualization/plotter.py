"""
Module for plotting time series visualizations of Instagram message data.

This module provides the `TimeSeriesPlotter` class, which generates plots for temporal
analysis results, including message counts, rolling averages, day-of-week, and hourly distributions.
"""
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import pandas as pd

from ..utils.logging import get_logger

if TYPE_CHECKING:
    from ..analysis.types import ActivityAnalysisResult, TimeSeriesDict

class TimeSeriesPlotter:
    """
    Generates time series visualizations from activity analysis results.

    This class creates plots for temporal analysis metrics from an AnalysisPipeline,
    focusing on ActivityAnalysis results, saving them as PNG files.

    Attributes
    ----------
    pipeline_results : dict[str, dict]
        Results dictionary from AnalysisPipeline, containing strategy results.
    output_dir : Path
        Directory path where plots will be saved.
    logger : logging.Logger
        Logger instance for logging messages and errors.

    Methods
    -------
    plot()
        Generates and saves all time series plots.
    """

    def __init__(self, pipeline_results: dict[str, Mapping], output_dir: Path) -> None:
        """
        Initialize the TimeSeriesPlotter.

        Parameters
        ----------
        pipeline_results : dict[str, Mapping]
            Results dictionary from AnalysisPipeline, keyed by strategy names.
        output_dir : Path
            Directory path where plots will be saved.
        """
        self.pipeline_results = pipeline_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

    def plot(self) -> None:
        """Generate and save all time series plots.

        This method creates plots for message counts, rolling averages, day-of-week,
        and hourly distributions from ActivityAnalysis results, saving them as PNG files.

        Notes
        -----
        Expects 'ActivityAnalysis' key in pipeline_results to conform to ActivityAnalysisResult.
        """
        # TODO: figure out how to properly type this. Type guard?
        # Expect ActivityAnalysisResult, but pipeline might return {} for failure
        activity_results: ActivityAnalysisResult = self.pipeline_results.get("ActivityAnalysis", {}) # type: ignore[assignment]
        if not isinstance(activity_results, dict):
            self.logger.warning("ActivityAnalyis result is not a dict; skipping plotting")
            return

        default_ts: TimeSeriesDict = {
            "counts": pd.Series(dtype="int64"),
            "rolling_avg": pd.Series(dtype="float64"),
            "dow_counts": pd.Series(dtype="int64"),
            "hour_counts": pd.Series(dtype="int64"),
        }
        ts_results: TimeSeriesDict = activity_results.get("time_series", default_ts)
        if not ts_results["counts"].any():
            self.logger.warning("No time series results available for plotting")
            return

        # TODO: plot by user?
        # Plot message counts and rolling average
        if "counts" in ts_results and "rolling_avg" in ts_results:
            plt.figure(figsize=(12, 6))
            ts_results["counts"].plot(label="Message Count")
            ts_results["rolling_avg"].plot(label="7-day Rolling Avg", linestyle="--")
            plt.title("Message Frequency Over Time")
            plt.xlabel("Time")
            plt.ylabel("Messages")
            plt.legend()

            message_freq_plot = self.output_dir / "message_frequency.png"
            plt.savefig(message_freq_plot)
            plt.close()
            self.logger.info("Saved message frequency plot to %s", message_freq_plot)
        else:
            self.logger.warning("Missing 'counts' or 'rolling_avg' in time_series; skipping frequency plot")

        # Plot day-of-week distribution
        if "dow_counts" in ts_results:
            plt.figure(figsize=(8, 5))
            ts_results["dow_counts"].plot(king="bar")
            plt.title("Messages by Day of Week")
            plt.xlabel("Day (0=Mon, 6=Sun)")
            plt.ylabel("Messages")

            dow_count_bar = self.output_dir / "dow_counts.png"
            plt.savefig(dow_count_bar)
            plt.close()
            self.logger.info("Saved day-of-week plot to %s", dow_count_bar)
        else:
            self.logger.warning("Missing 'dow_counts' in time_series; skipping day-of-week plot")

        # Plot hour-of-day distribution
        if "hour_counts" in ts_results:
            plt.figure(figsize=(10, 5))
            ts_results["hour_counts"].plot(kind="bar")
            plt.title("Messages by Hour of Day")
            plt.xlabel("Hour (0-23)")
            plt.ylabel("Messages")

            hour_count_bar = self.output_dir / "hour_counts.png"
            plt.close()
            self.logger.info("Saved hour-of-day plot to %s", hour_count_bar)
        else:
            self.logger.warning("Missing 'hour_counts' in time_series; skipping hour-of-day plot")

        bursts = activity_results.get("bursts", pd.DataFrame())
        if not bursts.empty and "burst_count" in bursts.columns:
            plt.figure(figsize=(12, 6))
            bursts["burst_count"].plot(label="Bursts", color="red")
            plt.title("Message Bursts Over Time")
            plt.xlabel("Time")
            plt.ylabel("Messages")
            plt.legend()

            bursts_plot = self.output_dir / "bursts.png"
            plt.savefig(bursts_plot)
            plt.close()
            self.logger.info("Saved bursts plot to %s", bursts_plot)
        else:
            self.logger.warning("No valid bursts data; skipping bursts plot")
