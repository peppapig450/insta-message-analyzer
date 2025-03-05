"""
Analysis strategies for the Instagram Message Analyzer.

Classes
-------
AnalysisStrategy
    Protocol defining analysis strategies for Instagram message data.

Notes
-----
Concrete strategy implementations are imported from submodules (e.g., activity.py).
"""

from pathlib import Path
from typing import Protocol

import pandas as pd


class AnalysisStrategy(Protocol):
    """
    Protocol for analysis strategies on Instagram message data.

    Attributes
    ----------
    name : str
        Unique name identifier for the strategy instance.

    Methods
    -------
    analyze(data)
        Performs analysis on the provided DataFrame and returns results.
    save_results(results, output_dir)
        Saves analysis results to the specified directory.

    """

    @property
    def name(self) -> str:
        """Unique name for the strategy instance."""
        ...

    def analyze(self, data: pd.DataFrame) -> dict:
        """
        Perform analysis on the provided DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame with Instagram message data.

        Returns
        -------
        dict
            Results of the analysis, format depends on the strategy.

        """
        ...

    def save_results(self, results: dict, output_dir: Path) -> None:
        """
        Save analysis results to the specified directory.

        Parameters
        ----------
        results : dict
            Results of the analysis to be saved.
        output_dir : pathlib.Path
            Directory path where results will be saved.

        """
        ...
