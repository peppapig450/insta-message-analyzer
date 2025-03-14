"""Temporal activity analysis strategy for the Instagram Message Analyzer."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import pandas as pd

from ...utils.setup_logging import get_logger
from ..analysis_types import ActivityAnalysisResult, ChatId, ChatLifecycle, TimeSeriesDict, TimeSeriesKey
from ..protocol import AnalysisStrategy

if TYPE_CHECKING:
    from pathlib import Path


class ActivityAnalysis(AnalysisStrategy[ActivityAnalysisResult]):
    """Concrete strategy for analyzing temporal activity patterns in Instagram message data.

    Computes metrics such as time-series data, burst periods, top senders, and normalized active hours
    per user, both overall and per chat, using chat IDs and names for identification.

    Attributes
    ----------
    name : str
        Unique name identifier for this strategy instance.
    rolling_window : int
        Window size (in days) for computing the rolling average, by default 7.
    burst_percentile : float
        Percentile threshold for detecting message bursts, by default 95.0.
    granularity : str
        Time granularity for counts (e.g., "D" for day, "H" for hour).
    top_n_senders : int
        Number of top senders to rank.
    analyze_overall : bool
        Whether to compute overall metrics.
    logger : logging.Logger
        Logger for debugging and info messages.
    """

    def __init__(  # noqa: PLR0913
        self,
        name: str = "ActivityAnalysis",
        rolling_window: int = 7,
        burst_percentile: float = 95.0,
        granularity: str = "D",
        top_n_senders: int = 5,
        *,
        analyze_overall: bool = True,
    ) -> None:
        """Initialize the ActivityAnalysis strategy.

        Parameters
        ----------
        name : str, optional
            Unique name for this strategy instance (default: "ActivityAnalysis").
        rolling_window : int, optional
            Window size in days for rolling average (default: 7).
        burst_percentile : float, optional
            Percentile threshold for detecting message bursts, by default 95.0.
        granularity : str, optional
            Time granularity for counts (default: "D").
        top_n_senders : int, optional
            Number of top senders to identify (default: 5).
        analyze_overall : bool, optional
            Toggle overall analysis (default: True).
        """
        self._name = name
        self.rolling_window = rolling_window
        self.burst_percentile = burst_percentile
        self.granularity = granularity
        self.top_n_senders = top_n_senders
        self.analyze_overall = analyze_overall
        self.logger = get_logger(__name__)
        self.logger.debug(
            "Initialized %s: rolling_window=%d, burst_percentile=%.2f, granularity=%s, top_n_senders=%d, analyze_overall=%s",
            name,
            rolling_window,
            burst_percentile,
            granularity,
            top_n_senders,
            analyze_overall,
        )

    @property
    def name(self) -> str:
        """Get the unique name of the strategy.

        Returns
        -------
        str
            The name of the strategy instance.
        """
        return self._name

    def analyze(self, data: pd.DataFrame) -> ActivityAnalysisResult:  # noqa: PLR0915 #TODO: refactor this function to be simpler (53 branches is too many)
        """Analyze temporal activity patterns in the provided Instagram message data.

        Computes daily message counts, rolling averages, day-of-week and hour-of-day
        distributions, and detects bursts of high messaging activity.

        Parameters
        ----------
        data : pd.DataFrame
            DataFrame with 'timestamp', 'chat_id' (int), 'chat_name', 'reactions' and 'sender' columns.

        Returns
        -------
        ActivityAnalysisResult
            Dictionary containing computed metrics and chat names.

        Raises
        ------
        KeyError
            If required columns are missing from the input DataFrame.
        """
        self.logger.debug("Starting analyze data with shape %s", data.shape)
        required_cols: set[str] = {"timestamp", "chat_id", "chat_name", "sender"}
        if missing := required_cols - set(data.columns):
            self.logger.error("Missing required columns: %s", missing)
            error_msg = f"DataFrame missing required columns: {missing}"
            raise KeyError(error_msg)

        if not pd.api.types.is_integer_dtype(data["chat_id"]):
            self.logger.error("chat_id column is not of integer type: %s", data["chat_id"].dtype)
            error_msg = "chat_id must be an integer type"
            raise TypeError(error_msg)

        message_data = data.copy()
        message_data["timestamp"] = pd.to_datetime(message_data["timestamp"], utc=True, errors="coerce")
        dropped_rows = len(message_data) - len(message_data.dropna(subset=["timestamp"]))
        message_data = message_data.dropna(subset=["timestamp"])
        self.logger.debug(
            "Dropped %d rows with invalid timestamps, remaining: %d", dropped_rows, len(message_data)
        )
        if message_data.empty:
            self.logger.warning("DataFrame is empty after processing")
            return self._empty_result()

        # Overall analysis
        overall_time_series = (
            self._compute_time_series(message_data) if self.analyze_overall else self._empty_time_series()
        )
        overall_bursts = (
            self._compute_bursts(overall_time_series["counts"], self.granularity)
            if self.analyze_overall
            else pd.DataFrame()
        )
        total_messages = len(message_data)

        # Per chat analysis
        per_chat: dict[ChatId, TimeSeriesDict] = {}
        per_chat_bursts: dict[ChatId, pd.DataFrame] = {}
        chat_lifecycles: dict[ChatId, ChatLifecycle] = {}
        top_senders_per_chat: dict[ChatId, pd.Series] = {}
        chat_names: dict[ChatId, str] = {}

        for chat_id, group in message_data.groupby("chat_id"):
            chat_id = cast(int, chat_id)
            chat_id_int: ChatId = ChatId(chat_id)
            chat_names[chat_id_int] = group["chat_name"].iloc[0]  # Assume consistent chat_name within group
            chat_ts = self._compute_time_series(group)
            per_chat[chat_id_int] = chat_ts
            per_chat_bursts[chat_id_int] = self._compute_bursts(chat_ts["counts"], self.granularity)

            # Chat lifecycle
            timestamps = group["timestamp"].sort_values()
            response_times = timestamps.diff().dt.total_seconds().dropna()
            chat_lifecycles[chat_id_int] = {
                "first_message": timestamps.iloc[0],
                "peak_date": pd.Timestamp(chat_ts["counts"].idxmax()).strftime("%Y-%m-%d")
                if not chat_ts["counts"].empty
                else "",
                "last_message": timestamps.iloc[-1],
                "avg_response_time": response_times.mean() if not response_times.empty else 0.0,
                "message_count": len(group),
            }

            # Top senders per chat
            top_senders = group["sender"].value_counts().head(self.top_n_senders)
            top_senders_per_chat[chat_id_int] = top_senders
            self.logger.debug(
                "Analyzed chat %d (%s): messages=%d, avg_response_time=%.2f s, top_senders=%s",
                chat_id_int,
                chat_names[chat_id_int],
                len(group),
                chat_lifecycles[chat_id_int]["avg_response_time"],
                top_senders.index.tolist(),
            )

        # Top senders overall (day and week)
        message_data["date"] = message_data["timestamp"].dt.floor("D")
        message_data["week"] = message_data["timestamp"].dt.to_period("W").dt.start_time
        top_senders_day = (
            message_data.groupby("date")["sender"]  # noqa: PD010
            .value_counts()
            .groupby("date")
            .head(self.top_n_senders)
            .unstack(fill_value=0)
        )
        top_senders_week = (
            message_data.groupby("week")["sender"]  # noqa: PD010
            .value_counts()
            .groupby("week")
            .head(self.top_n_senders)
            .unstack(fill_value=0)
        )
        self.logger.debug(
            "Top senders day shape: %s, week shape: %s", top_senders_day.shape, top_senders_week.shape
        )

        # Active hours per user
        active_hours_per_user: dict[str, pd.Series] = {}
        message_count_per_user: dict[str, int] = {}
        hours = range(24)
        for sender, group in message_data.groupby("sender"):
            hour_counts = (
                group["timestamp"].dt.hour.value_counts().reindex(hours, fill_value=0).sort_index()
            )

            # Store raw hour counts for later use
            sender_str = str(sender)
            active_hours_per_user[f"{sender_str}_raw"] = hour_counts.copy()

            # Store total message count for this user
            message_count_per_user[sender_str] = int(hour_counts.sum())

            # Calculate and store normalized values (percentage of activity by hour)
            active_hours_per_user[str(sender)] = hour_counts / hour_counts.sum()  # Normalized

            self.logger.debug(
                "Computed active hours for user %s, total messages: %d, sample: %s",
                sender,
                message_count_per_user[sender_str],
                hour_counts.head().to_dict(),
            )

        results: ActivityAnalysisResult = {
            "time_series": overall_time_series,
            "per_chat": per_chat,
            "bursts": overall_bursts,
            "per_chat_bursts": per_chat_bursts,
            "total_messages": total_messages,
            "top_senders_day": top_senders_day,
            "top_senders_week": top_senders_week,
            "chat_lifecycles": chat_lifecycles,
            "top_senders_per_chat": top_senders_per_chat,
            "active_hours_per_user": active_hours_per_user,
            "message_count_per_user": message_count_per_user,
            "chat_names": chat_names,
        }
        self.logger.info(
            "Analysis complete: %d messages, %d chats, %d users",
            total_messages,
            len(per_chat),
            len(active_hours_per_user),
        )

        return results

    def _compute_time_series(self, df: pd.DataFrame) -> TimeSeriesDict:
        """
        Compute time-series metrics for a DataFrame.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame with a 'timestamp' column.

        Returns
        -------
        TimeSeriesDict
            Dictionary with daily counts, rolling average, day-of-week, and hour distributions.
        """
        counts = df["timestamp"].dt.floor(self.granularity).value_counts().sort_index()
        counts.name = "message_count"
        self.logger.debug("Computed counts, length: %d, sample: %s", len(counts), counts.head().to_dict())
        rolling_avg = counts.rolling(window=self.rolling_window, min_periods=1).mean()
        rolling_avg.name = "rolling_avg"

        dow_counts = df["timestamp"].dt.dayofweek.value_counts().sort_index()
        num_days = (df["timestamp"].max() - df["timestamp"].min()).days + 1
        dow_counts = (dow_counts / (num_days / 7)).round(0).astype(int)
        dow_counts.name = "dow_counts_avg"

        hour_counts = df["timestamp"].dt.hour.value_counts().sort_index()
        hour_counts.name = "hour_counts"

        hourly_per_day = df.assign(date=df["timestamp"].dt.date, hour=df["timestamp"].dt.hour).pivot_table(
            index="date", columns="hour", values="timestamp", aggfunc="count", fill_value=0
        )
        hourly_per_day.index = pd.to_datetime(hourly_per_day.index, errors="raise")
        hourly_per_day.index.name = "date"
        hourly_per_day.columns.name = "hour"
        self.logger.debug("Hourly per day shape: %s", hourly_per_day.shape)

        result: TimeSeriesDict = {
            "counts": counts,
            "rolling_avg": rolling_avg,
            "dow_counts": dow_counts,
            "hour_counts": hour_counts,
            "hourly_per_day": hourly_per_day,
        }
        return result

    def _compute_bursts(self, counts: pd.Series[int], granularity: str) -> pd.DataFrame:
        """
        Compute message burst periods from a counts Series using percentile thresholds.

        Identifies continuous periods where message counts exceed a percentile threshold,
        grouping consecutive periods into bursts. For hourly granularity, uses hour-specific
        percentiles; otherwise, uses an overall percentile. Returns a DataFrame with start/end
        times and total message counts for each burst period.

        Parameters
        ----------
        counts : pd.Series
            Series of message counts with a datetime index (e.g., daily or hourly counts).
        granularity : str
            Time granularity of the counts (e.g., "D" for day, "H" for hour).

        Returns
        -------
        pd.DataFrame
            DataFrame with columns 'start' (datetime), 'end' (datetime), and 'message_count' (int),
            where each row represents a burst period. Empty if no bursts are detected.

        Notes
        -----
        - Uses self.burst_percentile (e.g., 95.0) to determine burst significance.
        - For granularity "H", applies hour-specific percentiles to account for daily patterns.
        - Gaps greater than the specified granularity (e.g., >1 day for "D", >1 hour for "H")
        separate bursts.
        """
        if counts.empty:
            self.logger.debug("Empty counts, returning empty bursts")
            return pd.DataFrame(columns=["start", "end", "message_count"])

        # Coerce index to DatetimeIndex
        counts.index = pd.to_datetime(counts.index, errors="coerce")
        if counts.index.hasnans:
            self.logger.warning("Index contains NaT values after coercion; dropping them")
            counts = counts.dropna()  # Ensure no NaT entries

        if not isinstance(counts.index, pd.DatetimeIndex):
            self.logger.warning("counts.index is not a DatetimeIndex; returning empty bursts")
            return pd.DataFrame(columns=["start", "end", "message_count"])

        if granularity == "H":
            # Hour-specific percentiles
            hour = counts.index.hour
            percentiles = counts.groupby("hour").quantile(self.burst_percentile / 100)
            # NOTE: percentiles.reindex(hour).values could replace the list comprehension for larger datasets
            threshold_series = pd.Series([percentiles[h] for h in hour], index=counts.index)
        else:
            # Overall percentile
            threshold = counts.quantile(self.burst_percentile / 100)
            threshold_series = pd.Series(threshold, index=counts.index)

        burst_mask = counts > threshold_series

        # Group consecutive burst days into periods
        burst_periods: list[dict[str, pd.Timestamp | int | None]] = []
        start_idx: pd.Timestamp | None = None
        prev_idx: pd.Timestamp | None = None
        total_count: int = 0

        for idx, is_burst in burst_mask.items():
            timestamp_idx = cast(pd.Timestamp, idx)  # Cast Hashable to Timestamp
            if is_burst:
                if start_idx is None:
                    start_idx = timestamp_idx  # Start of a new burst
                total_count += counts[timestamp_idx]
                prev_idx = timestamp_idx
            elif start_idx is not None:
                burst_periods.append({"start": start_idx, "end": prev_idx, "message_count": total_count})
                start_idx = None
                total_count = 0

        # Handle case where burst ends at the last index
        if start_idx is not None:
            burst_periods.append({"start": start_idx, "end": prev_idx, "message_count": total_count})

        bursts_df = pd.DataFrame(burst_periods)
        self.logger.debug("Detected %d burst periods", len(bursts_df))
        return bursts_df

    def _empty_time_series(self) -> TimeSeriesDict:
        """
        Return an empty time series structure.

        Returns
        -------
        TimeSeriesDict
            Dictionary with empty time series metrics.
        """
        empty_series = pd.Series(dtype="int64")
        empty_df = pd.DataFrame()
        self.logger.debug("Returning empty time series")
        return {
            "counts": empty_series,
            "rolling_avg": empty_series,
            "dow_counts": empty_series,
            "hour_counts": empty_series,
            "hourly_per_day": empty_df,
        }

    def _empty_result(self) -> ActivityAnalysisResult:
        """
        Return an empty ActivityAnalysisResult for edge cases.

        Returns
        -------
        ActivityAnalysisResult
            Dictionary with all fields initialized to empty or zero values.
        """
        self.logger.debug("Returning empty result")
        return {
            "time_series": self._empty_time_series(),
            "per_chat": {},
            "bursts": pd.DataFrame(),
            "per_chat_bursts": {},
            "total_messages": 0,
            "top_senders_day": pd.DataFrame(),
            "top_senders_week": pd.DataFrame(),
            "chat_lifecycles": {},
            "top_senders_per_chat": {},
            "active_hours_per_user": {},
            "message_count_per_user": {},
            "chat_names": {},
        }

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
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.logger.exception("Failed to create output directory %s", output_dir)
            raise
        self.logger.debug("Saving results to %s", output_dir)

        self._save_overall_results(results, output_dir)
        self._save_per_chat_results(results, output_dir)
        self._save_additional_results(results, output_dir)

        summary = {"total_messages": results["total_messages"], "total_chats": len(results["per_chat"])}

        try:
            pd.Series(summary).to_csv(output_dir / "activity_summary.csv")
        except OSError:
            self.logger.exception("Failed to save activity_summary.csv to %s", output_dir)
            raise

    def _save_overall_results(self, results: ActivityAnalysisResult, output_dir: Path) -> None:
        """Save overall time series and burst resutls to the main output directory."""
        if not self.analyze_overall:
            return

        # Note: Could use dynamic mapping like {k: f"overall_{k}.csv" for k in TimeSeriesDict.__annotations__}
        # for future maintainability if TimeSeriesDict keys change, but keeping hardcoded for simplicity now.
        time_series_files: dict[TimeSeriesKey, str] = {
            "counts": "overall_message_counts.csv",
            "rolling_avg": "overall_rolling_avg.csv",
            "dow_counts": "overall_dow_counts.csv",
            "hour_counts": "overall_hour_counts.csv",
            "hourly_per_day": "overall_hourly_per_day.csv",
        }
        for key, filename in time_series_files.items():
            if key in results["time_series"] and not results["time_series"][key].empty:
                try:
                    results["time_series"][key].to_csv(output_dir / filename)
                    self.logger.debug("Saved overall %s", key)
                except OSError:
                    self.logger.exception("Failed to save %s to %s", filename, output_dir)
                    raise
        if not results["bursts"].empty:
            try:
                results["bursts"].to_csv(output_dir / "overall_bursts.csv")
                self.logger.debug("Saved overall bursts, rows: %d", len(results["bursts"]))
            except OSError:
                self.logger.exception("Failed to save overall_bursts.csv to %s", output_dir)
                raise

    def _save_per_chat_results(self, results: ActivityAnalysisResult, output_dir: Path) -> None:
        """Save per-chat time series, bursts, and top senders to chat-specific subdirectories."""
        time_series_files: dict[TimeSeriesKey, str] = {
            "counts": "overall_message_counts.csv",
            "rolling_avg": "overall_rolling_avg.csv",
            "dow_counts": "overall_dow_counts.csv",
            "hour_counts": "overall_hour_counts.csv",
            "hourly_per_day": "overall_hourly_per_day.csv",
        }
        for chat_id, chat_ts in results["per_chat"].items():
            chat_dir = output_dir / f"chat_{chat_id}"
            try:
                chat_dir.mkdir(exist_ok=True)
            except OSError:
                self.logger.exception("Failed to create chat directory %s", chat_dir)
                raise
            for key, filename in time_series_files.items():
                if key in chat_ts and not chat_ts[key].empty:
                    try:
                        chat_ts[key].to_csv(chat_dir / filename.replace("overall_", ""))
                        self.logger.debug("Saved %s for chat %d", key, chat_id)
                    except OSError:
                        self.logger.exception(
                            "Failed to save %s for chat %d", filename.replace("overall_", ""), chat_id
                        )
                        raise
            if chat_id in results["per_chat_bursts"] and not results["per_chat_bursts"][chat_id].empty:
                try:
                    results["per_chat_bursts"][chat_id].to_csv(chat_dir / "bursts.csv")
                    self.logger.debug("Saved bursts for chat %d", chat_id)
                except OSError:
                    self.logger.exception("Failed to save bursts.csv for chat %d", chat_id)
                    raise
            if chat_id in results["top_senders_per_chat"]:
                try:
                    results["top_senders_per_chat"][chat_id].to_csv(chat_dir / "top_senders.csv")
                    self.logger.debug("Saved top senders for chat %d", chat_id)
                except OSError:
                    self.logger.exception("Failed to save top_senders.csv for chat %d", chat_id)
                    raise

    def _save_additional_results(self, results: ActivityAnalysisResult, output_dir: Path) -> None:
        """Save additional aggregated results to the main output directory."""
        if not results["top_senders_day"].empty:
            try:
                results["top_senders_day"].to_csv(output_dir / "top_senders_day.csv")
                self.logger.debug("Saved top senders day")
            except OSError:
                self.logger.exception("Failed to save top_senders_day.csv to %s", output_dir)
                raise
        if not results["top_senders_week"].empty:
            try:
                results["top_senders_week"].to_csv(output_dir / "top_senders_week.csv")
                self.logger.debug("Saved top senders week")
            except OSError:
                self.logger.exception("Failed to save top_senders_week.csv to %s", output_dir)
                raise
        if results["chat_lifecycles"]:
            try:
                pd.DataFrame(results["chat_lifecycles"]).T.to_csv(output_dir / "chat_lifecycles.csv")
                self.logger.debug("Saved chat lifecycles for %d chats", len(results["chat_lifecycles"]))
            except OSError:
                self.logger.exception(
                    "Failed to save chat_lifecycles.csv to %s",
                    output_dir,
                )
                raise
        if results["active_hours_per_user"]:
            try:
                pd.DataFrame(results["active_hours_per_user"]).T.to_csv(
                    output_dir / "active_hours_per_user.csv"
                )
                self.logger.debug("Saved active hours for %d users", len(results["active_hours_per_user"]))
            except OSError:
                self.logger.exception("Failed to save active_hours_per_user.csv to %s", output_dir)
                raise
        if results["chat_names"]:
            try:
                pd.Series(results["chat_names"]).to_csv(output_dir / "chat_names.csv")
                self.logger.debug("Saved chat names for %d chats", len(results["chat_names"]))
            except OSError:
                self.logger.exception("Failed to save chat_names.csv to %s", output_dir)
                raise
