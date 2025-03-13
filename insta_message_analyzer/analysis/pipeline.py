"""
Pipeline orchestration for analysis strategies in the Instagram Message Analyzer.

Classes
-------
AnalysisPipeline
    Orchestrates multiple analysis strategies on Instagram message data.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from ..utils.setup_logging import get_logger
from .protocol import AnalysisStrategy


class AnalysisPipeline:
    """
    Orchestrates multiple analysis strategies on Instagram message data.

    This class runs a list of analysis strategies on the provided data and aggregates
    their results, handling errors gracefully.

    Attributes
    ----------
    strategies : Sequence[AnalysisStrategy[Any]]
        List of analysis strategy instances to execute.
    logger : logging.Logger
        Logger instance for logging messages and errors.

    """

    def __init__(self, strategies: Sequence[AnalysisStrategy[Any]]) -> None:
        """
        Initialize the AnalysisPipeline.

        Parameters
        ----------
        strategies : Sequence[AnalysisStrategy[Any]]
            Sequence of analysis strategy instances to execute.

        """
        self.strategies = strategies
        self.logger = get_logger(__name__)

    def run_analysis(self, data: pd.DataFrame) -> dict[str, Mapping]:
        """
        Run all strategies on the provided data and return aggregated results.

        Parameters
        ----------
        data : pandas.DataFrame
        Input DataFrame with Instagram message data.

        Returns
        -------
        dict[str, Mapping]
            Dictionary with strategy names as keys and their analysis results as values.
            Failed strategies return None.

        Notes
        -----
        Errors are logged, and the pipeline continues with remaining strategies.

        """
        results: dict[str, Mapping] = {}
        for strategy in self.strategies:
            self.logger.info("Running analysis for %s", strategy.name)
            try:
                results[strategy.name] = strategy.analyze(data)
            except Exception:
                self.logger.exception("Error in %s", strategy.name)
                results[strategy.name] = {}
        self.logger.info("Completed analysis pipeline with %d strategies", len(self.strategies))
        return results

    def save_results(self, results: dict[str, Mapping], output_dir: Path) -> None:
        """
        Save all analysis results to the specified directory.

        Parameters
        ----------
        results : dict[str, Mapping]
            Dictionary of analysis results keyed by strategy names.
        output_dir : pathlib.Path
            Directory path where results will be saved.

        Notes
        -----
        Errors are logged, and the pipeline continues with remaining strategies.

        """
        output_dir.mkdir(parents=True, exist_ok=True)
        for strategy in self.strategies:
            strategy_results = results.get(strategy.name, {})
            try:
                strategy.save_results(strategy_results, output_dir)
            except Exception:
                self.logger.exception("Error saving results for %s", strategy.name)
