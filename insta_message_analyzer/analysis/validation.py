"""Validation functions for the Instagram Message Analyzer analysis module."""

from typing import TypeGuard

import pandas as pd

from .analysis_types import (
    ActivityAnalysisResult,
    ChatLifecycle,
    TimeSeriesDict,
)


def is_activity_analysis_result(obj: object) -> TypeGuard[ActivityAnalysisResult]:  # noqa: PLR0911, PLR0912
    """
    Type guard to verify if an object conforms to the ActivityAnalysisResult structure.

    This function performs a detailed structural validation of the input object to ensure it matches
    the expected schema of an ActivityAnalysisResult, including required keys, their types, and
    nested structures. It is designed to be used in type narrowing contexts to assist static type
    checkers and runtime validation.

    Parameters
    ----------
    obj : object
        The object to validate against the ActivityAnalysisResult type.

    Returns
    -------
    bool
        True if the object fully conforms to the ActivityAnalysisResult structure, False otherwise.

    Notes
    -----
    - Uses ActivityAnalysisResult.__annotations__ to dynamically retrieve required keys.
    - Performs shallow type checking on nested structures (e.g., dictionaries and pandas objects).
    - Does not validate the internal consistency of data (e.g., whether timestamps are valid).
    """
    # Initial check: must be a dictionary
    if not isinstance(obj, dict):
        return False

    # Get required keys from ActivityAnalysisResult annotations
    required_keys = set(ActivityAnalysisResult.__annotations__.keys())

    # Check if all required keys are present in the object
    if not required_keys.issubset(obj.keys()):
        return False

    # Validate types of each field based on annotations
    # time_series: must be a TimeSeriesDict
    if not isinstance(obj["time_series"], dict):
        return False
    time_series_keys = set(TimeSeriesDict.__annotations__.keys())
    if not time_series_keys.issubset(obj["time_series"].keys()):
        return False

    # per_chat: dict with ChatId keys and TimeSeriesDict values
    if not isinstance(obj["per_chat"], dict) or not all(isinstance(k, int) for k in obj["per_chat"]):
        return False
    if not all(
        isinstance(v, dict) and time_series_keys.issubset(v.keys()) for v in obj["per_chat"].values()
    ):
        return False

    # bursts: must be a DataFrame
    if not isinstance(obj["bursts"], pd.DataFrame):
        return False
    if not obj["bursts"].empty and not {"start", "end", "message_count"}.issubset(obj["bursts"].columns):
        return False  # Ensure required columns if not empty

    # per_chat_bursts: dict with ChatId keys and DataFrame values
    if not isinstance(obj["per_chat_bursts"], dict) or not all(
        isinstance(k, int) for k in obj["per_chat_bursts"]
    ):
        return False
    if not all(isinstance(v, pd.DataFrame) for v in obj["per_chat_bursts"].values()):
        return False

    # active_hours_per_user: dict with str keys and Series values
    if not isinstance(obj["active_hours_per_user"], dict) or not all(
        isinstance(k, str) for k in obj["active_hours_per_user"]
    ):
        return False
    if not all(isinstance(v, pd.Series) for v in obj["active_hours_per_user"].values()):
        return False

    # top_senders_day and top_senders_week: must be DataFrames
    if not isinstance(obj["top_senders_day"], pd.DataFrame) or not isinstance(
        obj["top_senders_week"], pd.DataFrame
    ):
        return False

    # chat_lifecycles: dict with ChatId keys and ChatLifecycle values
    if not isinstance(obj["chat_lifecycles"], dict) or not all(
        isinstance(k, int) for k in obj["chat_lifecycles"]
    ):
        return False
    lifecycle_keys = set(ChatLifecycle.__annotations__.keys())
    for lifecycle in obj["chat_lifecycles"].values():
        if not isinstance(lifecycle, dict) or not lifecycle_keys.issubset(lifecycle.keys()):
            return False
        if not (
            isinstance(lifecycle["first_message"], pd.Timestamp)
            and isinstance(lifecycle["peak_date"], str)
            and isinstance(lifecycle["last_message"], pd.Timestamp)
            and isinstance(lifecycle["avg_response_time"], float)
            and isinstance(lifecycle["message_count"], int)
        ):
            return False

    # chat_names: dict with ChatId keys and str values
    if not isinstance(obj["chat_names"], dict) or not all(isinstance(k, int) for k in obj["chat_names"]):
        return False
    return all(isinstance(v, str) for v in obj["chat_names"].values())


def is_time_series_dict(obj: object) -> TypeGuard[TimeSeriesDict]:
    """
    Type guard to verify if an object conforms to the TimeSeriesDict structure.

    Validates that the input object is a dictionary with the required keys and correct
    types as specified in TimeSeriesDict. This is useful for type narrowing in static
    type checkers and runtime validation within plotting or analysis code.

    Parameters
    ----------
    obj : object
        The object to validate against the TimeSeriesDict type.

    Returns
    -------
    bool
        True if the object fully conforms to the TimeSeriesDict structure, False otherwise.

    Notes
    -----
    - Checks for presence of all required keys: 'counts', 'rolling_avg', 'dow_counts',
      'hour_counts', and 'hourly_per_day'.
    - Validates that values are of the correct type (Series or DataFrame).
    - Does not check data content (e.g., emptiness or index validity), focusing on structure.
    """
    if not isinstance(obj, dict):
        return False

    # Define required keys from TimeSeriesDict
    required_keys = set(TimeSeriesDict.__annotations__.keys())
    if not required_keys.issubset(obj.keys()):
        return False

    # Validate types for each key
    return (
        isinstance(obj["counts"], pd.Series)
        and isinstance(obj["rolling_avg"], pd.Series)
        and isinstance(obj["dow_counts"], pd.Series)
        and isinstance(obj["hour_counts"], pd.Series)
        and isinstance(obj["hourly_per_day"], pd.DataFrame)
    )
