"""
Module for plotting time series visualizations of Instagram message data.

This module provides the `TimeSeriesPlotter` class, which generates plots for temporal
analysis results, including message counts, rolling averages, day-of-week, and hourly distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from ..analysis.types import ChatId
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

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
        self.logger.debug("Initialized TimeSeriesPlotter, output_dir: %s", output_dir)

    def plot(self) -> None:
        """
        Generate and save all time series plots.

        This method creates plots for message counts, rolling averages, day-of-week,
        and hourly distributions from ActivityAnalysis results, saving them as PNG files.

        Notes
        -----
        Expects 'ActivityAnalysis' key in pipeline_results to conform to ActivityAnalysisResult.
        """
        # Extract and validate ActivityAnalysis results
        # TODO: look into using TypeGuard here too verify type is ActivityAnalysisResult
        activity_results: ActivityAnalysisResult = self.pipeline_results.get("ActivityAnalysis", {})  # type: ignore[assignment]
        if not isinstance(activity_results, dict):
            self.logger.warning("ActivityAnalysis result is not a dict; skipping plotting")
            return

        # Default empty time series data
        default_ts: TimeSeriesDict = {
            "counts": pd.Series(dtype="int64"),
            "rolling_avg": pd.Series(dtype="float64"),
            "dow_counts": pd.Series(dtype="int64"),
            "hour_counts": pd.Series(dtype="int64"),
            "hourly_per_day": pd.DataFrame(),
        }
        ts_results: TimeSeriesDict = activity_results.get("time_series", default_ts)
        if not ts_results["counts"].any():
            self.logger.warning("No time series results available for plotting; skipping plotting")
            return

        # Generate individual plots
        self.logger.debug("Starting plot generation")
        self._plot_message_frequency(ts_results)
        self._plot_day_of_week(ts_results)
        self._plot_hour_of_day(ts_results)
        self._plot_bursts(activity_results)

    def _plot_message_frequency(self, ts_results: TimeSeriesDict) -> None:
        """Plot message counts and rolling average."""
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

    def _plot_day_of_week(self, ts_results: TimeSeriesDict) -> None:
        """Plot day-of-week distribution."""
        if "dow_counts" in ts_results:
            plt.figure(figsize=(8, 5))
            ts_results["dow_counts"].plot(kind="bar")  # Fixed typo: 'king' -> 'kind'
            plt.title("Messages by Day of Week")
            plt.xlabel("Day (0=Mon, 6=Sun)")
            plt.ylabel("Messages")
            dow_count_bar = self.output_dir / "dow_counts.png"
            plt.savefig(dow_count_bar)
            plt.close()
            self.logger.info("Saved day-of-week plot to %s", dow_count_bar)
        else:
            self.logger.warning("Missing 'dow_counts' in time_series; skipping day-of-week plot")

    def _plot_hour_of_day(self, ts_results: TimeSeriesDict) -> None:
        """Plot hour-of-day distribution."""
        if "hour_counts" in ts_results:
            plt.figure(figsize=(10, 5))
            ts_results["hour_counts"].plot(kind="bar")
            plt.title("Messages by Hour of Day")
            plt.xlabel("Hour (0-23)")
            plt.ylabel("Messages")
            hour_count_bar = self.output_dir / "hour_counts.png"
            plt.savefig(hour_count_bar)  # Fixed: Save before close
            plt.close()
            self.logger.info("Saved hour-of-day plot to %s", hour_count_bar)
        else:
            self.logger.warning("Missing 'hour_counts' in time_series; skipping hour-of-day plot")

    def _plot_hourly_per_day(self, ts_results: TimeSeriesDict) -> None:
        """
        Plot heatmap of hourly message counts per day.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics to plot.
        """
        if "hourly_per_day" in ts_results and not ts_results["hourly_per_day"].empty:
            self.logger.debug("Plotting hourly per day, shape: %s", ts_results["hourly_per_day"].shape)
            plt.figure(figsize=(12, 8))
            plt.imshow(ts_results["hourly_per_day"], aspect="auto", cmap="viridis")
            plt.colorbar(label="Message Count")
            plt.title("Hourly Messages Per Day")
            plt.xlabel("Hour (0-23)")
            plt.ylabel("Date")
            plt.xticks(range(24))
            plt.yticks(
                range(len(ts_results["hourly_per_day"])),
                ts_results["hourly_per_day"].index.astype(str).tolist(),
            )
            plt.savefig(self.output_dir / "hourly_per_day.png")
            plt.close()
            self.logger.info("Saved hourly per day heatmap")

    def _plot_bursts(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Plot message frequency with burst periods highlighted.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Full analysis results.
        """
        ts_results = activity_results.get("time_series", {})
        bursts = activity_results.get("bursts", pd.DataFrame())
        if "counts" in ts_results and not bursts.empty:
            self.logger.debug("Plotting bursts, burst rows: %d", len(bursts))
            plt.figure(figsize=(12, 6))
            ax = ts_results["counts"].plot(label="Messages")
            for idx in bursts.index:
                # Note: Wrap date2num with a Typed Wrapper also works
                ax.axvline(
                    x=float(mdates.date2num(idx).item()), # type: ignore[no-untyped-call]
                    color="red",
                    linestyle="--",
                    alpha=0.5,
                    label="Burst" if idx == bursts.index[0] else None,
                )
            plt.title("Message Frequency with Bursts")
            plt.xlabel("Date")
            plt.ylabel("Message Count")
            plt.legend()
            plt.savefig(self.output_dir / "bursts.png")
            plt.close()
            self.logger.info("Saved bursts plot")

    def _plot_top_senders_per_chat(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Plot bar charts of top senders for each chat.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Full analysis results including chat names.
        """
        top_senders = activity_results.get("top_senders_per_chat", {})
        chat_names = activity_results.get("chat_names", {})
        if not top_senders:
            self.logger.warning("No top senders per chat data; skipping plot")
            return
        for chat_id, senders in top_senders.items():
            chat_name = chat_names.get(chat_id, f"Chat {chat_id}")
            self.logger.debug(
                "Plotting top senders for chat %d (%s), senders: %s", chat_id, chat_name, senders.to_dict()
            )
            plt.figure(figsize=(10, 6))
            senders.plot(kind="bar", color="coral")
            plt.title(f"Top Senders in {chat_name}")
            plt.xlabel("Sender")
            plt.ylabel("Message Count")
            plt.xticks(rotation=45, ha="right")
            plt.tight_layout()
            plt.savefig(self.output_dir / f"top_senders_chat_{chat_id}.png")
            plt.close()
            self.logger.info("Saved top senders plot for %s", chat_name)

    def _plot_active_hours_heatmap(self, active_hours: dict[str, pd.Series]) -> None:
        """
        Plot a heatmap of normalized active hours per user.

        Parameters
        ----------
        active_hours : dict[str, pd.Series]
            Normalized hourly message distribution per user.
        """
        if not active_hours:
            self.logger.warning("No active hours data; skipping heatmap")
            return
        df = pd.DataFrame(active_hours).T.fillna(0)
        self.logger.debug("Plotting active hours heatmap, shape: %s", df.shape)
        plt.figure(figsize=(12, max(8, len(df) * 0.3)))
        plt.imshow(df, aspect="auto", cmap="viridis")
        plt.colorbar(label="Message Proportion")
        plt.title("Normalized Active Hours per User")
        plt.xlabel("Hour of Day (0-23)")
        plt.ylabel("User")
        plt.xticks(range(24))
        plt.yticks(range(len(df)), df.index.astype(str).tolist())
        plt.tight_layout()
        plt.savefig(self.output_dir / "active_hours_heatmap.png")
        plt.close()
        self.logger.info("Saved active hours heatmap")

    def _plot_top_senders_day(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Plot heatmap of top senders per day.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Full analysis results.
        """
        top_senders = activity_results.get("top_senders_day", pd.DataFrame())
        if not top_senders.empty:
            self.logger.debug("Plotting top senders day, shape: %s", top_senders.shape)
            plt.figure(figsize=(12, 8))
            plt.imshow(top_senders, aspect="auto", cmap="YlOrRd")
            plt.colorbar(label="Message Count")
            plt.title("Top Senders Per Day")
            plt.xlabel("Sender")
            plt.ylabel("Date")
            plt.xticks(
                range(len(top_senders.columns)),
                top_senders.columns.astype(str).tolist(),
                rotation=45,
                ha="right",
            )
            date_labels = top_senders.index.astype(str).tolist()
            plt.yticks(range(len(date_labels)), date_labels)
            plt.tight_layout()
            plt.savefig(self.output_dir / "top_senders_day.png")
            plt.close()
            self.logger.info("Saved top senders day plot")

    def _plot_top_senders_week(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Plot heatmap of top senders per week.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Full analysis results.
        """
        top_senders = activity_results.get("top_senders_week", pd.DataFrame())
        if not top_senders.empty:
            self.logger.debug("Plotting top senders week, shape: %s", top_senders.shape)
            plt.figure(figsize=(12, 8))
            plt.imshow(top_senders, aspect="auto", cmap="YlOrRd")
            plt.colorbar(label="Message Count")
            plt.title("Top Senders Per Week")
            plt.xlabel("Sender")
            plt.ylabel("Week Start")
            plt.xticks(
                range(len(top_senders.columns)),
                top_senders.columns.astype(str).tolist(),
                rotation=45,
                ha="right",
            )
            week_labels = top_senders.index.astype(str).tolist()
            plt.yticks(range(len(week_labels)), week_labels)
            plt.tight_layout()
            plt.savefig(self.output_dir / "top_senders_week.png")
            plt.close()
            self.logger.info("Saved top senders week plot")

    def _plot_chat_lifecycles(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Plot chat lifecycle durations with chat names.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Full analysis results including chat names.
        """
        lifecycles = activity_results.get("chat_lifecycles", {})
        chat_names = activity_results.get("chat_names", {})
        if not lifecycles:
            self.logger.warning("No chat lifecycles data; skipping plot")
            return
        df = pd.DataFrame(lifecycles).T
        durations = (df["last_message"] - df["first_message"]).dt.total_seconds() / (24 * 3600)  # Days
        labels = [f"{chat_names.get(ChatId(i), i)}" for i in df.index]
        self.logger.debug("Plotting chat lifecycles, chats: %d", len(df))
        plt.figure(figsize=(12, 6))
        durations.plot(kind="bar")
        plt.title("Chat Lifecycle Durations")
        plt.xlabel("Chat")
        plt.ylabel("Duration (Days)")
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "chat_lifecycles.png")
        plt.close()
        self.logger.info("Saved chat lifecycles plot")
