"""Temporal activity analysis strategy for the Instagram Message Analyzer."""

import logging
from pathlib import Path

import pandas as pd

from ...utils.logging import get_logger
from ..protocol import AnalysisStrategy
from ..types import ActivityAnalysisResult, TimeSeriesKey


class ActivityAnalysis(AnalysisStrategy[ActivityAnalysisResult]):
    """Concrete strategy for analyzing temporal activity patterns in Instagram message data.

    This strategy computes time-series metrics such as message counts, rolling averages,
    day-of-week and hour-of-day distributions, and detects message bursts.

    Attributes
    ----------
    name : str
        Unique name identifier for this strategy instance.
    rolling_window : int
        Window size (in days) for computing the rolling average, by default 7.
    burst_threshold : float
        Threshold (z-score) for detecting message bursts, by default 2.0.
    """

    def __init__(
        self, name: str = "ActivityAnalysis", rolling_window: int = 7, burst_threshold: float = 0.2
    ) -> None:
        """Initialize the ActivityAnalysis strategy.

        Parameters
        ----------
        name : str, optional
            Unique name for this strategy instance, by default "ActivityAnalysis".
        rolling_window : int, optional
            Window size (in days) for the rolling average, by default 7.
        burst_threshold : float, optional
            Z-score threshold for detecting message bursts, by default 2.0.
        """
        self._name = name
        self.rolling_window = rolling_window
        self.burst_threshold = burst_threshold
        self.logger = get_logger(__name__)
        self.logger.debug("Initialized ActivityAnalysis: name=%s, rolling_window=%d, burst_threshold=%.2f",
                         name, rolling_window, burst_threshold)
        self.logger.info("ActivityAnalysis initialized")

    @property
    def name(self) -> str:
        """Get the unique name of the strategy.

        Returns
        -------
        str
            The name of the strategy instance.
        """
        return self._name

    def analyze(self, data: pd.DataFrame) -> ActivityAnalysisResult:
        """Analyze temporal activity patterns in the provided Instagram message data.

        Computes daily message counts, rolling averages, day-of-week and hour-of-day
        distributions, and detects bursts of high messaging activity.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame with Instagram message data, expected to have a 'timestamp' column.

        Returns
        -------
        dict
            Dictionary containing:
            - 'time_series': dict with 'counts' (Series), 'rolling_avg' (Series),
              'dow_counts' (Series), 'hour_counts' (Series).
            - 'bursts': DataFrame with burst periods (timestamp, burst_count).
            - 'total_messages': int, total number of messages.

        Raises
        ------
        KeyError
            If 'timestamp' column is missing from the input DataFrame.
        """
        self.logger.debug("Starting analyze with data shape: %s", data.shape)
        if "timestamp" not in data.columns:
            error_msg = "DataFrame must contain a 'timestamp' column"
            self.logger.error("Missing 'timestamp' column in data: %s", error_msg)
            raise KeyError(error_msg)

        df = data.copy()
        self.logger.debug("Copied input data, initial rows: %d", len(df))

        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        self.logger.debug("Converted timestamps, sample: %s", df["timestamp"].head().tolist())

        dropped_rows = len(data) - len(df.dropna(subset=["timestamp"]))
        df = df.dropna(subset=["timestamp"])
        self.logger.debug(
            "Dropped %d rows with invalid timestamps, remaining rows: %d", dropped_rows, len(df)
        )

        counts = df["timestamp"].dt.floor("D").value_counts().sort_index()
        counts.name = "message_count"
        self.logger.debug(
            "Computed daily counts, length: %d, sample: %s", len(counts), counts.head().to_dict()
        )

        # TODO: Add configurable granularity (e.g., "H" for hourly counts)
        rolling_avg = counts.rolling(window=f"{self.rolling_window}D", min_periods=1, closed="both").mean()
        rolling_avg.name = "rolling_avg"
        self.logger.debug(
            "Computed rolling average, length: %d, sample: %s",
            len(rolling_avg),
            rolling_avg.head().to_dict(),
        )

        # TODO: Consider adaptive or exponential moving average for recent trends
        dow_counts = df["timestamp"].dt.dayofweek.value_counts().sort_index()
        dow_counts.name = "dow_counts"
        self.logger.debug("Computed day-of-week counts: %s", dow_counts.to_dict())

        # TODO: Normalize by number of days per weekday for fair comparison
        hour_counts = df["timestamp"].dt.hour.value_counts().sort_index()
        hour_counts.name = "hour_counts"
        self.logger.debug("Computed hour-of-day counts: %s", hour_counts.to_dict())

        # TODO: Add hourly counts per day for heatmap potential
        z_scores = (counts - counts.mean()) / counts.std()
        self.logger.debug(
            "Computed z-scores, mean: %.2f, std: %.2f, sample: %s",
            counts.mean(),
            counts.std(),
            z_scores.head().to_dict(),
        )

        burst_mask = z_scores > self.burst_threshold  # TODO: Fix default threshold (0.2 -> 2.0)
        bursts = counts[burst_mask].to_frame(name="burst_count")
        self.logger.debug(
            "Detected %d bursts with threshold %.2f, sample: %s",
            len(bursts),
            self.burst_threshold,
            bursts.head().to_dict(),
        )

        # TODO: Explore percentile-based or hourly burst detection

        results: ActivityAnalysisResult = {
            "time_series": {
                "counts": counts,
                "rolling_avg": rolling_avg,
                "dow_counts": dow_counts,
                "hour_counts": hour_counts,
            },
            "bursts": bursts,
            "total_messages": len(df),
        }
        self.logger.debug(
            "Analysis results prepared: total_messages=%d, time_series_keys=%s, bursts_rows=%d",
            results["total_messages"],
            list(results["time_series"].keys()),
            len(results["bursts"]),
        )
        self.logger.info("Completed temporal analysis for %s messages", results["total_messages"])
        return results

    def save_results(self, results: ActivityAnalysisResult, output_dir: Path) -> None:
        """Save temporal activity analysis results to the specified directory.

        Saves time-series metrics and burst data as separate CSV files.

        Parameters
        ----------
        results : ActivityAnalysisResult
            Results of the analysis, expected to match the structure returned by analyze().
        output_dir : pathlib.Path
            Directory path where results will be saved.

        Notes
        -----
        Files saved include: message_counts.csv, rolling_avg.csv, dow_counts.csv,
        hour_counts.csv, and bursts.csv.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        ts_results = results["time_series"]

        # Map time-series keys to their output filenames
        time_series_files: dict[TimeSeriesKey, str] = {
            "counts": "message_counts.csv",
            "rolling_avg": "rolling_avg.csv",
            "dow_counts": "dow_counts.csv",
            "hour_counts": "hour_counts.csv",
        }

        # Save each time-series result if present
        for key, filename in time_series_files.items():
            if key in ts_results:
                out_file = output_dir / filename
                ts_results[key].to_csv(out_file)
                self.logger.info("Saved %s to %s", key.replace("_", " "), out_file)
            else:
                self.logger.warning("time_series data incomplete or invalid; skipping save")

        # Save bursts if data exists
        bursts = results.get("bursts", pd.DataFrame())
        if not bursts.empty:
            bursts_out_file = output_dir / "bursts.csv"
            bursts.to_csv(bursts_out_file)
            self.logger.info("Saved bursts to %s", bursts_out_file)

        # Save summary
        summary = {"total_messages": results.get("total_messages", 0)}
        summary_out_file = output_dir / "activity_summary.csv"
        pd.Series(summary).to_csv(summary_out_file)
        self.logger.info("Saved activity summary to %s", summary_out_file)
