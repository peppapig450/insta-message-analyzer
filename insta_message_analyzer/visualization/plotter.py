"""
Module for plotting time series visualizations of Instagram message data.

This module provides the `TimeSeriesPlotter` class, which generates plots for temporal
analysis results, including message counts, rolling averages, day-of-week, and hourly distributions.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import emoji
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from matplotlib import rcParams
from plotly.subplots import make_subplots

from ..analysis.types import ChatId
from ..analysis.validation import is_activity_analysis_result, is_time_series_dict
from ..utils.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Mapping

    from ..analysis.types import ActivityAnalysisResult, TimeSeriesDict


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

        Parameters
        ----------
        pipeline_results : dict[str, Mapping]
            Results dictionary from AnalysisPipeline, keyed by strategy names.
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
            - Requires 'ActivityAnalysis' key in pipeline_results.
            - Saves plots as HTML files in output_dir for interactivity.
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
           #self._plot_message_frequency(ts_results)
           #self._plot_day_of_week(ts_results)
           #self._plot_hour_of_day(ts_results)
           #self._plot_hourly_per_day(ts_results)
           self._plot_bursts(activity_results)

           self.logger.info("Generated visualizations in %s", self.output_dir)
        except Exception:
            self.logger.exception("Error generating plots")

    def _apply_common_layout(self, fig: go.Figure, title: str) -> None:
        """
        Apply common layout settings to a Plotly figure.

        Parameters
        ----------
        fig : go.Figure
            The Plotly figure to modify.
        title : str
            Title for the plot.
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
        Save the plotly figure as an HTML file and handle exceptions.

        Parameters
        ----------
        fig : go.Figure
            The plotly figure to save
        filename : str
            Filename to save the plot to
        title : str
            Description of the plot for logging

        Returns
        -------
        Path | None
            Path to the saved file or None if saving failed
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
        Plot interactive message counts and rolling average.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics to plot
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
        Plot interactive day-of-week distribution with hover information.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics to plot.
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
        Plot interactive hour-of-day distribution.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics to plot.
        """
        if ts_results["hour_counts"].empty:
            self.logger.warning("Empty 'hour_counts'; skipping hour-of-day plot")
            return

        try:
            # Convert hour counts to DataFrame for Plotly Express
            hour_df = ts_results["hour_counts"].to_frame(name="Messages").reset_index(names=["Hour"])

            # Note: Later when optimizing, chain above with .assign(HourLabel=lambda x: x["Hour"].map("{:02d}:00"))
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
        Plot interactive heatmap of hourly message counts per day.

        Parameters
        ----------
        ts_results : TimeSeriesDict
            Time series metrics to plot.
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
                ticktext=[f"Burst {i + 1}" for i in bursts.index]
            )

            fig.update_layout(
                xaxis={
                    "rangeslider": {"visible": True},
                    "type": "date",
                    "title": "Time"
                }
            )

            # Apply common layout and save
            self._apply_common_layout(fig, "Message Burst Periods")
            self._save_figure(fig, "bursts.html", "interactive bursts Gantt chart")
        except Exception:
            self.logger.exception("Error plotting bursts")

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
        # Store original setting and disable math parsing
        original_parse_math = rcParams["text.parse_math"]
        rcParams["text.parse_math"] = False

        # Convert to DataFrame and transpose for heatmap
        active_hours_df = pd.DataFrame(active_hours).T
        active_hours_df.index.name = "User"
        active_hours_df.columns.name = "Hour"

        # Sanitize labels: remove emojis and escape '$'
        def sanitize_label(text: str) -> str:
            return emoji.replace_emoji(text, replace="").replace("$", "")

        sanitized_labels = [sanitize_label(label) for label in active_hours_df.index]

        # Set figure size dynamically
        n_users = len(active_hours_df)
        plt.figure(figsize=(12, max(8, n_users * 0.2)))

        # Create heatmap
        ax = sns.heatmap(active_hours_df, cmap="YlGnBu", cbar_kws={"label": "Normalized Activity"})

        # Set escaped labels on the y-axis
        ax.set_yticks(range(n_users))
        ax.set_yticklabels(sanitized_labels, rotation=0, fontsize=8)

        # Add titles and labels
        plt.title("Active Hours per User")
        plt.xlabel("Hour of Day")
        plt.ylabel("User")
        plt.tight_layout()

        # Save and close
        plt.savefig(self.output_dir / "active_hours_heatmap.png")
        plt.close()

        # Restore original setting
        rcParams["text.parse_math"] = original_parse_math
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

        # Ensure datetime type
        df["first_message"] = pd.to_datetime(df["first_message"], errors="coerce")
        df["last_message"] = pd.to_datetime(df["last_message"], errors="coerce")
        df = df.dropna(subset=["first_message", "last_message"])
        if df.empty:
            self.logger.warning("No valid chat lifecycle data after datetime conversion; skipping plot")
            return

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
