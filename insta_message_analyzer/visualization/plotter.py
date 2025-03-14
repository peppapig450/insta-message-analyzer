"""
Module for plotting time series visualizations of Instagram message data.

This module provides the `TimeSeriesPlotter` class, which generates plots for temporal
analysis results, including message counts, rolling averages, day-of-week, and hourly distributions.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..analysis.validation import is_activity_analysis_result, is_time_series_dict
from ..utils.setup_logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..analysis.analysis_types import ActivityAnalysisResult, TimeSeriesDict


class TimeSeriesPlotter:
    """
    Generates interactive time series visualizations from activity analysis results.

    This class creates interactive Plotly plots for temporal analysis metrics
    from an AnalysisPipeline, focusing on ActivityAnalysis results, saving them as HTML files.

    Attributes
    ----------
    pipeline_results : dict[str, dict]
        Results dictionary from AnalysisPipeline, containing strategy results.
    output_dir : Path
        Directory path where plots will be saved.
    logger : logging.Logger
        Logger instance for logging messages and errors.
    theme : str
        Visual theme for plots (either 'light' or 'dark').
    width : int
        Default width for plots in pixels.
    height : int
        Default height for plots in pixels.
    plotly_template : str
        Plotly template based on theme ('plotly_white' for light, 'plotly_dark' for dark).
    color_scheme : dict
        Color palette for visualization elements, dependent on the selected theme.
    """

    def __init__(
        self,
        pipeline_results: dict[str, Mapping],
        output_dir: Path,
        theme: str = "dark",
        width: int = 900,
        height: int = 600,
    ) -> None:
        """
        Initialize the TimeSeriesPlotter.

        Sets up the plotter with pipeline results, output directory, and visualization preferences.

        Parameters
        ----------
        pipeline_results : dict[str, Mapping]
            Results dictionary from `AnalysisPipeline`, keyed by strategy names.
        output_dir : Path
            Directory path where plots will be saved.
        theme : str, optional
            Visual theme for plots, either 'light' or 'dark'. Default is 'dark'.
        width : int, optional
            Default width for plots in pixels. Default is 900.
        height : int, optional
            Default height for plots in pixels. Default is 600.
        """
        self.pipeline_results = pipeline_results
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(__name__)

        # Configuration for plots
        self.theme = theme
        self.width = width
        self.height = height
        self.plotly_template = "plotly_white" if theme == "light" else "plotly_dark"

        # Define color schemes based on theme
        if theme == "dark":
            self.color_scheme = {
                "background": "#1e1e1e",
                "paper_bgcolor": "#2d2d2d",
                "font_color": "#ffffff",
                "grid_color": "rgba(255, 255, 255, 0.1)",
                "primary": "#5dadec",  # Blue
                "secondary": "#ff6b6b",  # Red
                "accent": "#ffd700",  # Gold
                "colorscale_sequential": "Viridis",
                "colorscale_diverging": "RdBu",
            }
        else:  # Light theme
            self.color_scheme = {
                "background": "#ffffff",
                "paper_bgcolor": "#f8f9fa",
                "font_color": "#333333",
                "grid_color": "rgba(0, 0, 0, 0.1)",
                "primary": "#1f77b4",  # Blue
                "secondary": "#d62728",  # Red
                "accent": "#ff7f0e",  # Orange
                "colorscale_sequential": "Blues",
                "colorscale_diverging": "RdBu",
            }

        self.logger.debug("Initialized TimeSeriesPlotter, output_dir: %s, theme: %s", output_dir, theme)

    def plot(self) -> None:
        """
        Generate and save all time series plots as interactive HTML files.

        Validates the 'ActivityAnalysis' results using `is_activity_analysis_result` and generates
        plots if data is valid. Skips plotting with an error log if validation fails.

        Notes
        -----
        - Requires 'ActivityAnalysis' key in `pipeline_results` with valid data.
        - Saves all plots as HTML files in `output_dir` for interactivity.
        - Logs warnings if data validation fails and exceptions if plotting errors occur
        """
        # Extract and validate ActivityAnalysis results
        activity_results = self.pipeline_results.get("ActivityAnalysis", {})

        if not is_activity_analysis_result(activity_results):
            self.logger.warning("ActivityAnalysis result is not a dict; skipping plotting")
            return

        # Extract and validate time series data
        ts_results = activity_results.get("time_series", {})
        if not is_time_series_dict(ts_results):
            self.logger.warning("Invalid TimeSeriesDict structure in time_series; skipping plotting")
            return

        # Generate individual plots
        self.logger.debug("Starting plot generation")
        try:
            self._plot_message_frequency(ts_results)
            self._plot_day_of_week(ts_results)
            self._plot_hour_of_day(ts_results)
            self._plot_hourly_per_day(ts_results)
            self._plot_bursts(activity_results)
            self._plot_top_senders_per_chat(activity_results)
            self._plot_top_senders_per_day(activity_results)
            self._plot_top_senders_per_week(activity_results)
            self._plot_active_hours_heatmap(activity_results)
            self._plot_chat_lifecycles(activity_results)

            self.logger.info("Generated visualizations in %s", self.output_dir)
        except Exception:
            self.logger.exception("Error generating plots")

    def _apply_common_layout(self, fig: go.Figure, title: str) -> None:
        """
        Apply common layout settings to a Plotly figure.

        Updates the given Plotly figure with standardized layout settings, including title,
        background colors, font colors, grid lines, and a watermark, based on the class's theme.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to modify in place.
        title : str
            Title to set for the plot.

        Notes
        -----
        - Modifies the figure in place; does not return a new figure.
        - Adds a subtle watermark with the text "Generated with TimeSeriesPlotter".
        """
        fig.update_layout(
            title={
                "text": title,
                "font": {"size": 24, "color": self.color_scheme["font_color"]},
                "x": 0.5,  # Center the title
                "xanchor": "center",
            },
            template=self.plotly_template,
            paper_bgcolor=self.color_scheme["paper_bgcolor"],
            plot_bgcolor=self.color_scheme["background"],
            font={"color": self.color_scheme["font_color"]},
            width=self.width,
            height=self.height,
            margin={"l": 80, "r": 30, "t": 100, "b": 80},
            xaxis={
                "gridcolor": self.color_scheme["grid_color"],
                "linecolor": self.color_scheme["grid_color"],
            },
            yaxis={
                "gridcolor": self.color_scheme["grid_color"],
                "linecolor": self.color_scheme["grid_color"],
            },
            hovermode="closest",
        )

        # Add subtle watermark
        value = int(0.5 * 255)  # 127
        color = f"rgba({value},{value},{value},{value})"
        fig.add_annotation(
            text="Generated with TimeSeriesPlotter",
            xref="paper",
            yref="paper",
            x=0.98,
            y=0.01,
            showarrow=False,
            font={"size": 8, "color": color},
        )

    def _save_figure(self, fig: go.Figure, filename: str, title: str = "Plot") -> Path | None:
        """
        Save a Plotly figure as an interactive HTML file and handle exceptions.

        Writes the figure to the specified filename in the output directory with interactive
        features like zoom and export options, logging the outcome.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to save.
        filename : str
            Filename (without path) to save the plot as (e.g., 'plot.html').
        title : str, optional
            Description of the plot for logging purposes. Default is "Plot".

        Returns
        -------
        Path | None
            Path to the saved HTML file if successful, None if saving fails.

        Notes
        -----
        - Saves the file in `output_dir` with Plotly.js included via CDN.
        - Logs success or failure with exceptions if they occur.
        """
        output_path = self.output_dir / filename
        try:
            # Add configuration options for the interactive plot
            config = {
                "scrollZoom": True,
                "displayModeBar": True,
                "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                "displaylogo": False,
                "toImageButtonOptions": {
                    "format": "png",
                    "filename": filename.replace(".html", ""),
                    "height": self.height,
                    "width": self.width,
                    "scale": 2,  # Higher resolution for exports
                },
            }

            fig.write_html(
                output_path,
                include_plotlyjs="cdn",  # Use CDN to reduce file size
                config=config,
                include_mathjax="cdn",
                full_html=True,
            )

            self.logger.info("Saved %s to %s", title, output_path)
        except Exception:
            self.logger.exception("Faild to save %s", title)
            return None
        else:
            return output_path

    def _plot_message_frequency(self, ts_results: TimeSeriesDict) -> None:
        """
        Generate and save an interactive plot of daily message counts and 7-day rolling average.

        Creates a dual-axis Plotly figure with daily message counts as bars and a 7-day rolling
        average as a line, including a range slider for time navigation.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics containing 'counts' (daily message counts) and 'rolling_avg'
            (7-day rolling average).

        Notes
        -----
        - Skips plotting if 'counts' or 'rolling_avg' is empty, logging a warning.
        - Saves the plot to 'message_frequency.html' in `output_dir`.
        - Includes hover templates and range selector buttons for interactivity.
        """
        if ts_results["counts"].empty or ts_results["rolling_avg"].empty:
            self.logger.warning("Empty 'counts' or 'rolling_avg'; skipping frequency plot")
            return

        try:
            # Create figure with two y-axes for better comparison
            fig = make_subplots(specs=[[{"secondary_y": True}]])

            # Add bar chart for daily message counts
            fig.add_trace(
                go.Bar(
                    x=ts_results["counts"].index,
                    y=ts_results["counts"].to_numpy(),
                    name="Daily Messages",
                    marker_color=self.color_scheme["primary"],
                    opacity=0.7,
                    hovertemplate=(
                        "<b>Date</b>: %{x|%Y-%m-%d}<br><b>Messages</b>: %{y:,}<br><extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

            # Add line for rolling average
            fig.add_trace(
                go.Scatter(
                    x=ts_results["rolling_avg"].index,
                    y=ts_results["rolling_avg"].to_numpy(),
                    name="7-Day Average",
                    line={"color": self.color_scheme["secondary"], "width": 3},
                    hovertemplate=(
                        "<b>Date</b>: %{x|%Y-%m-%d}<br><b>7-Day Avg</b>: %{y:.1f}<br><extra></extra>"
                    ),
                ),
                secondary_y=True,
            )

            # Add range slider and selector buttons for time navigation
            fig.update_layout(
                xaxis={
                    # Range selector buttons configuration
                    "rangeselector": {
                        "buttons": [
                            # 1 week backward
                            {"count": 7, "label": "1w", "step": "day", "stepmode": "backward"},
                            # 1 month backward
                            {"count": 1, "label": "1m", "step": "month", "stepmode": "backward"},
                            # 3 months backward
                            {"count": 3, "label": "3m", "step": "month", "stepmode": "backward"},
                            # 6 months backward
                            {"count": 6, "label": "6m", "step": "month", "stepmode": "backward"},
                            # 1 year backward
                            {"count": 1, "label": "1y", "step": "year", "stepmode": "backward"},
                            # Show all data
                            {"step": "all"},
                        ],
                        # Styling
                        "bgcolor": self.color_scheme["paper_bgcolor"],
                        "activecolor": self.color_scheme["primary"],
                    },
                    # Range slider configuration
                    "rangeslider": {"visible": True},
                    # Set axis type
                    "type": "date",
                }
            )

            # Set axis titles
            fig.update_yaxes(title_text="Daily Message Count", secondary_y=False, showgrid=True)
            fig.update_yaxes(title_text="7-day Rolling Average", secondary_y=True, showgrid=False)
            fig.update_xaxes(title_text="Date")

            # Apply common layout settings
            self._apply_common_layout(fig, "Message Frequency Over Time")

            # Save the interactive plot
            self._save_figure(fig, "message_frequency.html", "interactive message frequency plot")
        except Exception:
            self.logger.exception("Error plotting interactive message frequency")

    def _plot_day_of_week(self, ts_results: TimeSeriesDict) -> None:
        """
        Generate and save an interactive bar chart of message counts by day of the week.

        Creates a Plotly bar chart showing total message counts for each day of the week,
        with hover information and text labels on bars.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics containing 'dow_counts' (message counts per day of week).

        Notes
        -----
        - Skips plotting if 'dow_counts' is empty, logging a warning.
        - Saves the plot to 'dow_counts.html' in `output_dir`.
        - Maps day indices to names (e.g., 0 to 'Monday') if exactly 7 days are present.
        """
        if ts_results["dow_counts"].empty:
            self.logger.warning("Empty 'dow_counts'; skipping day-of-week plot")
            return

        try:
            # Map numerical day values to day names
            day_names: tuple[str, ...] = (
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday",
            )
            dow_counts = ts_results["dow_counts"].copy()

            if len(dow_counts) == 7:  # Only rename if we have exactly 7 days  # noqa: PLR2004
                dow_counts.index = pd.Index(day_names)
            else:
                self.logger.warning("'dow_counts' does not have an index of length 7; skipping renaming.")

            # Convert to DataFrame for Plotly Express
            dow_df = dow_counts.to_frame(name="Messages").reset_index(names=["Day"])

            # Create bar chart
            fig = px.bar(
                data_frame=dow_df,
                x="Day",
                y="Messages",
                color="Messages",
                color_continuous_scale=self.color_scheme["colorscale_sequential"],
                text="Messages",
                labels={"Messages": "Message Count"},
                template=self.plotly_template,
            )

            # Customize text display format
            fig.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                hovertemplate=("<b>%{x}</b><br>Messages: %{y:,}<br><extra></extra>"),
            )

            # Enchanced layout
            fig.update_layout(
                coloraxis_showscale=False,  # Hide the color scale
                xaxis_title="Day of Week",
                yaxis_title="Message Count",
            )

            # Apply common layout
            self._apply_common_layout(fig, "Messages by Day of Week")

            # Save the figure
            self._save_figure(fig, "dow_counts.html", "interactive day-of-week plot")
        except Exception:
            self.logger.exception("Error plotting interactive day of week")

    def _plot_hour_of_day(self, ts_results: TimeSeriesDict) -> None:
        """
        Generate and save an interactive bar chart of message counts by hour of the day.

        Creates a Plotly bar chart displaying total message counts for each hour (0-23),
        with formatted hour labels (e.g., '00:00') and hover information.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics containing 'hour_counts' (message counts per hour).

        Notes
        -----
        - Skips plotting if 'hour_counts' is empty, logging a warning.
        - Saves the plot to 'hour_counts.html' in `output_dir`.
        - Rotates x-axis labels 45 degrees for readability.
        """
        if ts_results["hour_counts"].empty:
            self.logger.warning("Empty 'hour_counts'; skipping hour-of-day plot")
            return

        try:
            # Convert hour counts to DataFrame for Plotly Express
            hour_df = ts_results["hour_counts"].to_frame(name="Messages").reset_index(names=["Hour"])

            # NOTE: Later when optimizing, chain above with .assign(HourLabel=lambda x: x["Hour"].map("{:02d}:00"))
            # Add formatted hour labels
            def _format_hour(hour: int) -> str:
                """Format an hour as a two-digit string with ':00' appended."""
                return f"{hour:02d}:00"

            hour_df["HourLabel"] = hour_df["Hour"].astype(int).apply(_format_hour)

            # Create bar chart
            fig = px.bar(
                data_frame=hour_df,
                x="HourLabel",
                y="Messages",
                text="Messages",
                labels={"HourLabel": "Hour of Day", "Messages": "Message Count"},
                template=self.plotly_template,
            )

            # Customize text display format
            fig.update_traces(
                texttemplate="%{text:,}",
                textposition="outside",
                hovertemplate=("<b>%{x}</b><br>Messages: %{y:,}<br><extra></extra>"),
            )

            # Enhanced layout
            fig.update_layout(
                xaxis_title="Hour of Day",
                yaxis_title="Message Count",
                xaxis={
                    "tickmode": "array",
                    "tickvals": hour_df["HourLabel"].tolist(),
                    "ticktext": hour_df["HourLabel"].tolist(),
                    "tickangle": 45,
                },
            )

            # Apply common layout settings
            self._apply_common_layout(fig, "Messages by Hour of Day")

            # Save the figure
            self._save_figure(fig, "hour_counts.html", "interactive hour-of-day plot")

        except Exception:
            self.logger.exception("Error plotting interactive hour of day")

    def _plot_hourly_per_day(self, ts_results: TimeSeriesDict) -> None:
        """
        Generate and save an interactive heatmap of message counts per hour for each day.

        Creates a Plotly density heatmap with dates on the y-axis, hours (00:00-23:00) on
        the x-axis, and message counts as color intensity.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics containing 'hourly_per_day' (DataFrame of hourly counts per day).

        Notes
        -----
        - Skips plotting if 'hourly_per_day' is empty, logging a debug message.
        - Saves the plot to 'hourly_per_day.html' in `output_dir`.
        - Displays most recent dates at the top (reversed y-axis).
        """
        if ts_results["hourly_per_day"].empty:
            self.logger.debug("Empty 'hourly_per_day'; skipping hourly per day plot")
            return

        try:
            hourly_df = ts_results["hourly_per_day"]

            # Convert DataFrame to format suitable for Plotly
            hourly_data: list[dict[str, str | int]] = []
            for date_idx, row in hourly_df.iterrows():
                if isinstance(date_idx, pd.Timestamp) and pd.notna(date_idx):
                    date_str = date_idx.strftime("%Y-%m-%d")
                else:
                    date_str = "Unknown"  # Fallback for NaT or non-Timestamp
                    self.logger.debug(
                        "Non-datetime index after coercion: %s", date_idx
                    )  # NOTE: maybe try pd.Timestamp in future versions?

                for hour, count in enumerate(row):
                    hourly_data.append(
                        {
                            "Date": date_str,
                            "Hour": f"{hour:02d}:00",
                            "Count": int(count) if pd.notna(count) else 0,
                            "HourNum": hour,
                        }
                    )

            hourly_plot_df = pd.DataFrame(hourly_data)

            # Create heatmap
            fig = px.density_heatmap(
                data_frame=hourly_plot_df,
                x="Hour",
                y="Date",
                z="Count",
                labels={"Count": "Message Count"},
                color_continuous_scale=self.color_scheme["colorscale_sequential"],
                template=self.plotly_template,
            )

            # Customize hover information
            fig.update_traces(
                hovertemplate=(
                    "<b>Date</b>: %{y}<br><b>Hour</b>: %{x}<br><b>Messages</b>: %{z}<br><extra></extra>"
                ),
                # NOTE: customdata if we still want HourNum explicitly
                # customdata=hourly_plot_df["HourNum"].values.reshape(-1, 1),  # noqa: ERA001
            )

            # Set custom tick labels for hours
            fig.update_layout(
                xaxis={
                    "tickmode": "array",
                    "tickvals": list(range(24)),
                    "ticktext": [f"{h:02d}:00" for h in range(24)],
                    "tickangle": 45,
                    "title": "Hour of Day",
                },
                yaxis={
                    "title": "Date",
                    "autorange": "reversed",  # Most recent dates at the top
                },
            )

            # Apply common layout
            self._apply_common_layout(fig, "Hourly Messages Per Day")

            # Save the figure
            self._save_figure(fig, "hourly_per_day.html", "interactive hourly per day heatmap")
        except Exception:
            self.logger.exception("Error plotting interactive hourly per day")

    def _plot_bursts(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Create an interactive Gantt chart of message burst periods.

        This method visualizes continuous burst periods as horizontal bars, where each bar represents
        a period of high message activity identified by message counts exceeding a percentile threshold.
        The x-axis shows the timeline with start dates as the base and durations extending to end dates,
        while the y-axis lists burst periods. Hover tooltips provide start/end dates and total message counts.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Analysis results containing 'bursts' (DataFrame with 'start', 'end', 'message_count' columns).

        Notes
        -----
        - Skips plotting if 'bursts' is empty
        - Assumes 'bursts' is generated by _compute_bursts, with 'start' and 'end' as pd.Timestamp
        and 'message_count' as an integer sum of messages in the period.
        - Uses Plotly's horizontal bar chart with a range slider and date-based x-axis for interactivity.
        - Burst periods are labeled sequentially (e.g., 'Burst 1', 'Burst 2') on the y-axis.
        """
        bursts = activity_results.get("bursts", pd.DataFrame())

        if bursts.empty:
            self.logger.warning("Missing 'bursts' data; skipping plot")
            return

        try:
            fig = px.timeline(
                bursts,
                x_start="start",
                x_end="end",
                y=bursts.index,
                hover_data={"message_count": True},
                color_discrete_sequence=[self.color_scheme["primary"]],
            )

            fig.update_yaxes(
                title_text="Burst Period",
                tickmode="array",
                tickvals=bursts.index,
                ticktext=[f"Burst {i + 1}" for i in bursts.index],
            )

            fig.update_layout(xaxis={"rangeslider": {"visible": True}, "type": "date", "title": "Time"})

            # Apply common layout and save
            self._apply_common_layout(fig, "Message Burst Periods")
            self._save_figure(fig, "bursts.html", "interactive bursts Gantt chart")
        except Exception:
            self.logger.exception("Error plotting bursts")

    def _plot_top_senders_per_chat(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Create an interactive grouped bar chart of top senders per chat.

        This method visualizes the top message senders for each chat, grouping bars by chat name
        (mapped from chat IDs) and coloring by sender. Chat names are sourced from the 'chat_names'
        dictionary for improved readability.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Analysis results containing 'top_senders_per_chat' (dict of chat IDs to sender counts)
            and 'chat_names' (dict mapping chat IDs to names).

        Notes
        -----
        - Skips plotting if 'top_senders_per_chat' is empty or missing, logging a warning.
        - Uses chat names from 'chat_names'; falls back to 'Chat {chat_id}' if a name is missing.
        - Rotates x-axis labels 45 degrees for readability with potentially long sender names.
        """
        top_senders = activity_results.get("top_senders_per_chat", {})
        chat_names = activity_results.get("chat_names", {})

        if not top_senders:
            self.logger.warning("No top senders data available; skipping top senders per chat plot")
            return

        try:
            # NOTE: When optimizing the pandas operations can be chained

            # Create DataFrame with chat IDs as columns and senders as index
            chat_sender_counts = pd.DataFrame(
                {str(chat_id): series for chat_id, series in top_senders.items()}
            )

            # Melt the DataFrame
            sender_df = chat_sender_counts.reset_index().melt(
                id_vars=["sender"],
                value_vars=chat_sender_counts.columns.tolist(),
                var_name="ChatID",
                value_name="Messages",
            )
            # Drop rows with NaN messages and map ChatID to Chat names
            sender_df = sender_df.dropna(subset=["Messages"])

            # Map ChatID to Chat names
            chat_names_str = {str(k): v for k, v in chat_names.items()}
            sender_df["Chat"] = sender_df["ChatID"].map(lambda x: chat_names_str.get(x, f"Chat {x}"))
            sender_df = sender_df.drop(columns=["ChatID"])[["Chat", "sender", "Messages"]]

            # Sort by Messages in descending order to order bars greatest to least within each chat
            sender_df = (
                sender_df.groupby("Chat")
                .apply(lambda x: x.sort_values("Messages", ascending=False))
                .reset_index(drop=True)
            )

            fig = px.bar(
                data_frame=sender_df,
                x="sender",
                y="Messages",
                color="Chat",
                barmode="group",
                text="Messages",
                labels={"Messages": "Message Count", "Chat": "Chat Name", "sender": "Sender"},
                template=self.plotly_template,
            )

            # Update traces for better readability
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")

            # Perform type narrowing with a runtime check
            if not isinstance(fig.data, Sequence):
                error_msg = "fig.data must be a sequence to support len()"
                self.logger.exception(
                    "%s, got type: %s, value: %r", error_msg, type(fig.data).__name__, fig.data
                )
                raise TypeError(error_msg)  # noqa: TRY301 #NOTE: add function for this later

            # Create dropdown buttons for filtering by chat
            all_chats = sender_df["Chat"].unique().tolist()
            buttons = [
                {
                    "label": "All Chats",
                    "method": "update",
                    "args": [
                        {"visible": [True] * len(fig.data)},
                        {"title": "Top Senders Per Chat - All Chats"},
                    ],
                }
            ]

            # Add a button for each chat
            for chat in all_chats:
                # Get senders for this chat only
                chat_df = sender_df[sender_df["Chat"] == chat]
                chat_senders = chat_df["sender"].tolist()
                visibility = [trace.name == chat for trace in fig.data]
                buttons.append(
                    {
                        "label": chat,
                        "method": "update",
                        "args": [
                            {"visible": visibility},
                            {
                                "title": f"Top Senders Per Chat - {chat}",
                                "xaxis.categoryorder": "array",
                                "xaxis.categoryarray": chat_senders,  # Only show relevant senders
                            },
                        ],
                    }
                )

            # Update layout with dropdown menu
            # NOTE: for multiple selections we need to use Dash or custom JavaScript and write custom callbacks
            fig.update_layout(
                xaxis_title="Sender",
                yaxis_title="Message Count",
                xaxis={
                    "tickangle": 45,
                },
                legend_title="Chat Name",
                yaxis={"range": [0, sender_df["Messages"].max() * 1.1]},
                updatemenus=[
                    {
                        "buttons": buttons,
                        "direction": "down",
                        "pad": {"r": 10, "t": 10},
                        "showactive": True,
                        "x": 0.1,
                        "xanchor": "left",
                        "y": 1.1,
                        "yanchor": "top",
                    }
                ],
            )

            self._apply_common_layout(fig, "Top Senders Per Chat")
            self._save_figure(fig, "top_senders_per_chat.html", "interactive top senders per chat plot")
        except Exception:
            self.logger.exception("Error plotting top senders per chat")

    def _plot_active_hours_heatmap(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Create an interactive heatmap of active hours per user with enhanced features.

        Visualizes hourly message activity with hours (0-23) on x-axis and users on y-axis,
        including hover details, zoom functionality, and interactive controls.
        Supports both normalized percentage view and raw count view.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Analysis results containing 'active_hours_per_user' (dict of user hourly activity)
            and 'message_count_per_user' (dict of user message counts).

        Notes
        -----
        - Skips plotting if 'active_hours_per_user' is empty, logging a warning.
        - Saves the plot to 'active_hours_heatmap.html' in `output_dir`.
        - Includes buttons for toggling views and sorting, plus a peak activity annotation.
        """
        active_hours = activity_results.get("active_hours_per_user", {})
        message_counts = activity_results.get("message_count_per_user", {})

        if not active_hours:
            self.logger.warning("No active hours data available; skipping active hours heatmap")
            return

        try:
            # Extract normalized data (keys without _raw suffix)
            normalized_keys = [k for k in active_hours if not k.endswith("_raw")]
            normalized_data = pd.DataFrame({k: active_hours[k] for k in normalized_keys}).T.fillna(0)

            # Extract raw data (keys with _raw suffix)
            raw_keys = [k for k in active_hours if k.endswith("_raw")]

            # If raw data is available in the format we stored
            if raw_keys:
                # Strip _raw suffix for display
                raw_data = pd.DataFrame(
                    {k.replace("_raw", ""): active_hours[k] for k in raw_keys}
                ).T.fillna(0)
            else:
                # Fallback to reconstructing from normalized data and message counts
                raw_data = normalized_data.copy()
                for user in raw_data.index:
                    if user in message_counts:
                        raw_data.loc[user] = normalized_data.loc[user] * message_counts[user]

            if normalized_data.empty:
                self.logger.warning("Active hours data is empty after conversion; skipping plot")
                return

            # Cap the maximum height to prevent overly tall plots with many users
            max_height = min(800, max(400, len(normalized_data) * 20))

            # Initial view with normalized data
            fig = px.imshow(
                normalized_data,
                labels={"x": "Hour", "y": "User", "color": "Activity %"},
                x=[f"{h:02d}:00" for h in range(24)],
                color_continuous_scale=self.color_scheme["colorscale_sequential"],
                template=self.plotly_template,
                aspect="auto",  # Allow flexible aspect ratio
                height=max_height,  # Capped dynamic height
            )

            # Enhanced layout configuration
            fig.update_layout(
                title={
                    "text": "Active Hours Heatmap Per User (Normalized)",
                    "y": 0.95,
                    "x": 0.5,
                    "xanchor": "center",
                    "yanchor": "top",
                },
                xaxis_title="Hour of Day",
                yaxis_title="User",
                xaxis={
                    "tickangle": 45,
                    "tickmode": "array",
                    "tickvals": list(range(24)),
                    "ticktext": [f"{h:02d}:00" for h in range(24)],
                    "showgrid": True,
                    "gridcolor": "rgba(128,128,128,0.2)",
                },
                yaxis={
                    "tickmode": "linear",
                    "automargin": True,  # Prevents label cutoff
                    # Sort users by most active to least active
                    "categoryorder": "total descending",
                },
                coloraxis_colorbar={
                    "title": "Activity %",
                    "tickformat": ".1%",  # Format as percentage for normalized data
                    "thickness": 20,
                },
                margin={"l": 100, "r": 50, "t": 100, "b": 50},
                hovermode="closest",
            )

            # Add interactive hover template based on normalized view
            fig.update_traces(
                hovertemplate="<b>User</b>: %{y}<br><b>Hour</b>: %{x}<br><b>Messages</b> %{z:.1%}<br>"
            )

            # Find peak activity hours for annotation
            peak_hour = int(normalized_data.mean().idxmax())
            fig.add_annotation(
                x=peak_hour,
                y=-0.5,  # Just below the chart
                text=f"Peak Activity: {peak_hour:02d}:00",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="red",
                font={"size": 12, "color": "darkred"},
            )

            # Build toggle buttons for different views
            normalized_args = [
                {
                    "z": [normalized_data.to_numpy()],
                    "zmin": [0],
                    "zmax": [normalized_data.to_numpy().max()],
                },
                {
                    "title": "Active Hours Heatmap Per User (Normalized)",
                    "coloraxis.colorbar.title": "Activity %",
                    "coloraxis.colorbar.tickformat": ".1%",
                },
            ]

            raw_args = [
                {"z": [raw_data.to_numpy()], "zmin": [0], "zmax": [raw_data.to_numpy().max()]},
                {
                    "title": "Active Hours Heatmap Per User (Raw Counts)",
                    "coloraxis.colorbar.title": "Messages",
                    "coloraxis.colorbar.tickformat": ",.0f",
                },
            ]

            # Add buttons for interactivity
            fig.update_layout(
                updatemenus=[
                    {
                        "buttons": [
                            {
                                "args": normalized_args,
                                "label": "Show Normalized",
                                "method": "update",
                            },
                            {
                                "args": raw_args,
                                "label": "Show Raw Counts",
                                "method": "update",
                            },
                            {
                                "args": [{"zmax": [0.3], "zmin": [0]}, {"title": "Zoomed View (0-30%)"}],
                                "label": "Zoom 0-30%",
                                "method": "update",
                            },
                        ],
                        "direction": "down",
                        "showactive": True,
                        "x": 0.1,
                        "xanchor": "left",
                        "y": 1.1,
                        "yanchor": "top",
                    },
                    {
                        "buttons": [
                            {
                                "args": [
                                    {"yaxis.categoryorder": "total descending"},
                                    {"title": "Sorted by Total Activity (Descending)"},
                                ],
                                "label": "Sort by Activity ↓",
                                "method": "update",
                            },
                            {
                                "args": [
                                    {"yaxis.categoryorder": "alphabet"},
                                    {"title": "Sorted Alphabetically"},
                                ],
                                "label": "Sort Alphabetically",
                                "method": "update",
                            },
                        ],
                        "direction": "down",
                        "showactive": True,
                        "x": 0.3,
                        "xanchor": "left",
                        "y": 1.1,
                        "yanchor": "top",
                    },
                ]
            )

            # Apply common layout and save
            self._apply_common_layout(fig, "Active Hours Heatmap Per User")
            self._save_figure(fig, "active_hours_heatmap.html", "interactive active hours heatmap")
        except Exception:
            self.logger.exception("Error plotting active hours heatmap.")

    def _plot_top_senders_per_day(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Create an interactive grouped bar chart of top senders per day.

        This method visualizes the top message senders for each day, grouping bars by sender
        and coloring by day, with dates formatted as 'Feb 12, 2025' for readability.
        The y-axis scale adjusts dynamically based on the maximum message count of the currently
        displayed data (all days or a specific day).

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Analysis results containing 'top_senders_day' (DataFrame with dates as index and senders as columns).

        Notes
        -----
        - Skips plotting if 'top_senders_day' is empty or missing, logging a warning.
        - Bars are grouped by sender, with colors representing days.
        - Rotates x-axis labels 45 degrees for readability with potentially long sender names.
        - Y-axis range adjusts dynamically to the maximum message count of the visible data.
        """
        top_senders_df = activity_results.get("top_senders_day", pd.DataFrame())
        if top_senders_df.empty:
            self.logger.warning("No top senders by day data available; skipping top senders per day plot")
            return

        try:
            # Melt the DataFrame to long format: columns 'date', 'sender', 'Messages'
            sender_df = top_senders_df.reset_index().melt(
                id_vars=["date"], var_name="sender", value_name="Messages"
            )
            # Filter out zero message counts (senders not in top N for a day)
            sender_df = sender_df[sender_df["Messages"] > 0]
            # Rename 'date' to 'Day' and convert to string for trace consistency
            sender_df = sender_df.rename(columns={"date": "Day"})
            # Format the Day column to a readable string
            sender_df["Day"] = sender_df["Day"].dt.strftime("%b %d, %Y")  # e.g., "Feb 12, 2025"

            # Sort senders by message count within each day (descending)
            sender_df = (
                sender_df.groupby("Day")
                .apply(lambda x: x.sort_values("Messages", ascending=False))
                .reset_index(drop=True)
            )

            # Create the interactive bar chart
            fig = px.bar(
                data_frame=sender_df,
                x="sender",
                y="Messages",
                color="Day",
                barmode="group",
                text="Messages",
                labels={"Messages": "Message Count", "Day": "Day", "sender": "Sender"},
                template=self.plotly_template,
            )

            # Enhance readability of bar labels
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")

            # Verify fig.data is a sequence for length operations
            if not isinstance(fig.data, Sequence):
                error_msg = "fig.data must be a sequence to support len()"
                self.logger.exception(
                    "%s, got type: %s, value: %r", error_msg, type(fig.data).__name__, fig.data
                )
                raise TypeError(error_msg)  # noqa: TRY301

            # Calculate maximum message counts for scaling
            # Overall max for "All Days"
            overall_max = sender_df["Messages"].max() * 1.2  # Add 20% buffer
            # Per-day max for individual day filters
            day_maxes = sender_df.groupby("Day")["Messages"].max() * 1.2  # Add 20% buffer per day

            # Create dropdown buttons
            all_days = sender_df["Day"].unique().tolist()
            buttons = [
                {
                    "label": "All Days",
                    "method": "update",
                    "args": [
                        {"visible": [True] * len(fig.data)},
                        {"title": "Top Senders Per Day - All Days", "yaxis.range": [0, overall_max]},
                    ],
                }
            ]

            # Add a button for each day
            for day in all_days:
                # Filter senders for this day
                day_df = sender_df[sender_df["Day"] == day]
                day_senders = day_df["sender"].tolist()
                # Get the max for this day
                day_max = day_maxes[day]
                # Set visibility: show only the trace for this day
                visibility = [trace.name == day for trace in fig.data]
                buttons.append(
                    {
                        "label": day,
                        "method": "update",
                        "args": [
                            {"visible": visibility},
                            {
                                "title": f"Top Senders Per Day - {day}",
                                "xaxis.categoryorder": "array",
                                "xaxis.categoryarray": day_senders,  # Only show relevant senders
                                "yaxis.range": [0, day_max],  # Scale to this day's max
                            },
                        ],
                    }
                )

            # Update layout with axes titles, legend, and dropdown
            fig.update_layout(
                xaxis_title="Sender",
                yaxis_title="Message Count",
                xaxis={"tickangle": 45},
                legend_title="Day",
                yaxis={"range": [0, overall_max]},
                updatemenus=[
                    {
                        "buttons": buttons,
                        "direction": "down",
                        "pad": {"r": 10, "t": 10},
                        "showactive": True,
                        "x": 0.1,
                        "xanchor": "left",
                        "y": 1.1,
                        "yanchor": "top",
                    }
                ],
            )

            # Apply common layout and save the figure
            self._apply_common_layout(fig, "Top Senders Per Day")
            self._save_figure(fig, "top_senders_per_day.html", "interactive top senders per day plot")

        except Exception:
            self.logger.exception("Error plotting top senders per day")

    def _plot_top_senders_per_week(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Create an interactive grouped bar chart of top senders per week.

        This method visualizes the top message senders for each week, grouping bars by sender
        and coloring by week, with week start dates formatted as 'Jan 6, 2025' for readability.
        It includes a dropdown menu to filter the plot by specific weeks or view all weeks, with
        the y-axis scale adjusting dynamically to the maximum message count of the visible data.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Analysis results containing 'top_senders_week' (DataFrame with week start dates as index
            and senders as columns).

        Notes
        -----
        - Skips plotting if 'top_senders_week' is empty or missing, logging a warning.
        - Bars are grouped by sender, with colors representing weeks.
        - Rotates x-axis labels 45 degrees for readability with potentially long sender names.
        - Y-axis range adjusts dynamically to the maximum message count of the visible data.
        """
        top_senders_df = activity_results.get("top_senders_week", pd.DataFrame())
        if top_senders_df.empty:
            self.logger.warning("No top senders by week data available; skipping top senders per week plot")
            return

        try:
            # Melt the DataFrame to long format: 'week', 'sender', 'Messages'
            sender_df = top_senders_df.reset_index().melt(
                id_vars=["week"], var_name="sender", value_name="Messages"
            )

            # Filter out zero message counts
            sender_df = sender_df[sender_df["Messages"] > 0]
            # Format week start dates as 'Jan 6, 2025' and rename to 'Week'
            sender_df = sender_df.rename(columns={"week": "Week", "sender": "Sender"})
            sender_df["Week"] = sender_df["Week"].dt.strftime("%b %d, %Y")  # e.g., "Jan 6, 2025"

            # Sort senders by message count within each week (descending)
            sender_df = (
                sender_df.groupby("Week")
                .apply(lambda x: x.sort_values("Messages", ascending=False))
                .reset_index(drop=True)
            )

            # Create the interactive bar chart
            fig = px.bar(
                data_frame=sender_df,
                x="Sender",
                y="Messages",
                color="Week",
                barmode="group",
                text="Messages",
                labels={"Messages": "Message Count", "Week": "Week Start", "Sender": "Sender"},
                template=self.plotly_template,
            )

            # Enhance readability of bar labels
            fig.update_traces(texttemplate="%{text:,}", textposition="outside")

            # Verify fig.data is a sequence for length operations
            if not isinstance(fig.data, Sequence):
                error_msg = "fig.data must be a sequence to support len()"
                self.logger.exception(
                    "%s, got type: %s, value: %r", error_msg, type(fig.data).__name__, fig.data
                )
                raise TypeError(error_msg)  # noqa: TRY301 #TODO: properly resolve this

            # Calculate maximum message counts for scaling
            # Overall max for "All Weeks"
            overall_max = sender_df["Messages"].max() * 1.2  # Add 20% buffer
            # Per-week max for individual week filters
            week_maxes = sender_df.groupby("Week")["Messages"].max() * 1.2  # Add a 20% buffer per week

            # Create dropdown buttons
            all_weeks = sender_df["Week"].unique().tolist()
            buttons = [
                {
                    "label": "All Weeks",
                    "method": "update",
                    "args": [
                        {"visible": [True] * len(fig.data)},
                        {
                            "title": "Top Senders Per Week - All Weeks",
                            "yaxis.range": [0, overall_max],  # Scale to overall max
                        },
                    ],
                }
            ]

            # Add a button for ea ch week
            for week in all_weeks:
                # Filter senders for this week
                week_df = sender_df[sender_df["Week"] == week]
                week_senders = week_df["Sender"].tolist()
                # Get the max for this week
                week_max = week_maxes[week]
                # Set visibility: show only the trace for this week
                visibility = [trace.name == week for trace in fig.data]
                buttons.append(
                    {
                        "label": week,  # Formatted as "Jan 6, 2025"
                        "method": "update",
                        "args": [
                            {"visible": visibility},
                            {
                                "title": f"Top Senders Per Week - {week}",
                                "xaxis.categoryorder": "array",
                                "xaxis.categoryarray": week_senders,
                                "yaxis.range": [0, week_max],
                            },
                        ],
                    }
                )

            # Update layout with axes titles, legend, and dropdown
            fig.update_layout(
                xaxis_title="Sender",
                yaxis_title="Message Count",
                xaxis={"tickangle": 45},
                legend_title="Week Start",
                yaxis={"range": [0, overall_max]},
                updatemenus=[
                    {
                        "buttons": buttons,
                        "direction": "down",
                        "pad": {"r": 10, "t": 10},
                        "showactive": True,
                        "x": 0.1,
                        "xanchor": "left",
                        "y": 1.1,
                        "yanchor": "top",
                    }
                ],
            )

            # Apply common layout and save the figure
            self._apply_common_layout(fig, "Top Senders Per Week")
            self._save_figure(fig, "top_senders_week.html", "interactive top senders per week plot")
        except Exception:
            self.logger.exception("Error plotting top senders per week")

    def _plot_chat_lifecycles(self, activity_results: ActivityAnalysisResult) -> None:
        """
        Generate and save an interactive bar chart of chat lifecycle durations.

        Visualizes each chat's duration as horizontal bars along a timeline, with start dates
        on the x-axis, chat names on the y-axis, and optional peak activity markers.

        Parameters
        ----------
        activity_results : ActivityAnalysisResult
            Analysis results containing 'chat_lifecycles' (dict of chat lifecycle data)
            and 'chat_names' (dict mapping chat IDs to names).

        Notes
        -----
        - Skips plotting if 'chat_lifecycles' is empty, logging a warning.
        - Saves the plot to 'chat_lifecycles.html' in `output_dir`.
        - Includes hover details (e.g., duration, response time) and a range slider.
        """
        lifecycles = activity_results.get("chat_lifecycles", {})
        chat_names = activity_results.get("chat_names", {})

        if not lifecycles:
            self.logger.warning("No chat lifecycles data; skipping plot")
            return

        try:
            lifecycle_data = []
            for chat_id, lifecycle in lifecycles.items():
                # Get display name if available
                chat_name = chat_names.get(chat_id, f"Chat {chat_id}")

                duration_hours = (
                    lifecycle["last_message"] - lifecycle["first_message"]
                ).total_seconds() / 3600

                # NOTE: TypedDict?
                lifecycle_data.append(
                    {
                        "ChatID": chat_id,
                        "ChatName": chat_name,
                        "Start": lifecycle["first_message"],
                        "End": lifecycle["last_message"],
                        "PeakDate": pd.Timestamp(lifecycle["peak_date"])
                        if lifecycle["peak_date"]
                        else None,
                        "AvgResponseTime": lifecycle["avg_response_time"],
                        "DurationHours": duration_hours,
                        "MessageCount": lifecycle["message_count"],
                    }
                )

            if not lifecycle_data:
                self.logger.warning("No lifecycle data to plot")
                return

            lifecycle_df = pd.DataFrame(lifecycle_data)
            # Sort by start date for better visualization
            lifecycle_df = lifecycle_df.sort_values(by="Start")

            # Create figure with timeline display
            fig = go.Figure()

            # Add the main chat timeline bars
            fig.add_trace(
                go.Bar(
                    x=lifecycle_df["Start"],
                    y=lifecycle_df["ChatName"],
                    width=(lifecycle_df["DurationHours"] * pd.Timedelta(hours=1)).dt.total_seconds()
                    * 1000,  # Convert hours to milliseconds for plotly
                    orientation="h",
                    marker_color=self.color_scheme["primary"],
                    hovertemplate=(
                        "<b>Chat</b>: %{y}<br>"
                        "<b>Start</b>: %{x|%Y-%m-%d %H:%M}<br>"
                        "<b>End</b>: %{customdata[0]|%Y-%m-%d %H:%M}<br>"
                        "<b>Duration</b>: %{customdata[1]:.1f} hours<br>"
                        "<b>Avg Response<b>: %{customdata[2]:.1f} seconds<br>"
                        "<b>Messages</b>: %{customdata[3]}<br>"
                        "<extra></extra>"
                    ),
                    customdata=list(
                        zip(
                            lifecycle_df["End"],
                            lifecycle_df["DurationHours"],
                            lifecycle_df["AvgResponseTime"],
                            lifecycle_df["MessageCount"],
                            strict=False,
                        )
                    ),
                )
            )

            # Add peak activity markers if available
            peak_dates = lifecycle_df[lifecycle_df["PeakDate"].notna()]
            if not peak_dates.empty:
                fig.add_trace(
                    go.Scatter(
                        x=peak_dates["PeakDate"],
                        y=peak_dates["ChatName"],
                        mode="markers",
                        marker={
                            "symbol": "star",
                            "size": 10,
                            "color": self.color_scheme.get("accent", "yellow"),
                        },
                        name="Peak Activity",
                        hovertemplate="<b>%{y}</b><br>Peak Activity: %{x|%Y-%m-%d}<extra></extra>",
                    )
                )

            fig.update_layout(
                xaxis_title="Timeline",
                yaxis_title="Chat",
                xaxis={"type": "date", "rangeslider_visible": True},
                barmode="overlay",
                height=max(300, len(lifecycle_df) * 30),
            )

            self._apply_common_layout(fig, "Chat Lifecycles")
            self._save_figure(fig, "chat_lifecycles.html", "interactive chat lifecycles plot")
        except Exception:
            self.logger.exception("Error plotting chat lifecycles")
