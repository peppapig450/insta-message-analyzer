"""Type definitions for the Instagram Message Analyzer analysis module."""

from typing import Literal, TypedDict

import pandas as pd


class TimeSeriesDict(TypedDict):
    """Directory structure for time-series analysis results."""

    counts: pd.Series
    rolling_avg: pd.Series
    dow_counts: pd.Series
    hour_counts: pd.Series


class ActivityAnalysisResult(TypedDict):
    """Result structure for ActivityAnalysis strategy."""

    time_series: TimeSeriesDict
    bursts: pd.DataFrame
    total_messages: int


type TimeSeriesKey = Literal["counts", "rolling_avg", "dow_counts", "hour_counts"]
