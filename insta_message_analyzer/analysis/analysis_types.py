"""Type definitions for the Instagram Message Analyzer analysis module."""

from typing import Any, Literal, NewType, TypedDict

import networkx as nx
import pandas as pd

# Base types and type aliases
ChatId = NewType("ChatId", int)
"""Unique numeric identifier for a chat, distinct from other integer types."""

type CountSeries = pd.Series
"""Series containing message counts or related metrics."""

type BurstDataFrame = pd.DataFrame
"""DataFrame storing burst analysis results."""

type TopSendersDataFrame = pd.DataFrame
"""DataFrame storing top senders aggregated by day or week."""

# Valid keys for time series metrics
type TimeSeriesKey = Literal["counts", "rolling_avg", "dow_counts", "hour_counts", "hourly_per_day"]
"""Allowed keys for accessing time series metrics."""


# Time series metrics structure
class TimeSeriesDict(TypedDict):
    """
    Structure for storing time-series analysis results.

    Attributes
    ----------
    counts : CountSeries
        Total message counts over time.
    rolling_avg : CountSeries
        Rolling average of message counts.
    dow_counts : CountSeries
        Message counts by day of week.
    hour_counts : CountSeries
        Message counts by hour of day.
    hourly_per_day : HourlyDataFrame
        Hourly message counts per day.
    """

    counts: CountSeries
    rolling_avg: CountSeries
    dow_counts: CountSeries
    hour_counts: CountSeries
    hourly_per_day: TopSendersDataFrame


class ChatLifecycle(TypedDict):
    """
    Structure for chat lifecycle metrics.

    Attributes
    ----------
    first_message : pd.Timestamp
        Timestamp of the first message in the chat.
    peak_date : str
        Date with the highest message count (YYYY-MM-DD).
    last_message : pd.Timestamp
        Timestamp of the last message in the chat.
    avg_response_time : float
        Average response time between messages in seconds.
    message_count : int
        Total number of messages in a chat.
    """

    first_message: pd.Timestamp
    peak_date: str
    last_message: pd.Timestamp
    avg_response_time: float
    message_count: int


# Per-chat and per-user analysis types
type PerChatTimeSeries = dict[ChatId, TimeSeriesDict]
"""Time series metrics grouped by chat."""

type PerChatBursts = dict[ChatId, BurstDataFrame]
"""Burst analysis results grouped by chat."""

type TopSendersPerChat = dict[ChatId, CountSeries]
"""Top message senders grouped by chat."""

type ActiveHoursPerUser = dict[str, CountSeries]
"""Hourly message distribution grouped by user."""


# Full analysis result structure
class ActivityAnalysisResult(TypedDict):
    """
    Structure for the complete activity analysis output.

    All fields are required and populated by ActivityAnalysis, even in edge cases.
    Derived from a DataFrame with 'chat_id' (int), 'chat_name', 'sender', 'timestamp', etc.

    Attributes
    ----------
    time_series : TimeSeriesDict
        Aggregated time series metrics across all chats.
    per_chat : PerChatTimeSeries
        Time series metrics for each chat.
    bursts : BurstDataFrame
        Aggregated burst analysis across all chats.
    per_chat_bursts : PerChatBursts
        Burst analysis for each chat.
    total_messages : int
        Total number of messages analyzed.
    top_senders_per_chat : TopSendersPerChat
        Top senders for each chat, with chat IDs as keys and sender counts as values.
    active_hours_per_user : ActiveHoursPerUser
        Normalized hourly activity distribution for each user, with usernames as keys.
    message_count_per_user : dict[str, int]
        Total message count for each user, with usernames as keys.
    top_senders_day : TopSendersDataFrame
        Top senders aggregated by day, with dates as rows and senders as columns.
    top_senders_week : TopSendersDataFrame
        Top senders aggregated by week, with week starts as rows and senders as columns.
    chat_lifecycles : dict[ChatId, ChatLifecycle]
        Lifecycle metrics for each chat.
    chat_names : dict[ChatId, str]
        Mapping of chat_id to chat_name.
    """

    time_series: TimeSeriesDict
    per_chat: PerChatTimeSeries
    bursts: BurstDataFrame
    per_chat_bursts: PerChatBursts
    total_messages: int
    top_senders_per_chat: TopSendersPerChat
    active_hours_per_user: ActiveHoursPerUser
    message_count_per_user: dict[str, int]
    top_senders_day: TopSendersDataFrame
    top_senders_week: TopSendersDataFrame
    chat_lifecycles: dict[ChatId, ChatLifecycle]
    chat_names: dict[ChatId, str]

# TODO: add better docstrings, use type aliases, try to get rid of Any
class NetworkAnalysisResult(TypedDict):
    """Structure for network analysis output."""

    bipartite_graph: nx.Graph
    sender_centrality: dict[str, dict[str, float]]
    chat_centrality: dict[str, dict[str, float]]
    communities: dict[str, int]
    community_metrics: dict[str, Any]
    sender_projection: nx.Graph
    influence_metrics: dict[str, Any]
    cross_chat_metrics: dict[str, Any]
