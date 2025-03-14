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

from collections.abc import Mapping
from pathlib import Path
from typing import Generic, Protocol, TypeVar

import pandas as pd

# Define a generic type variable for the result
R = TypeVar("R", bound=Mapping)  # Constrain to dict-like types (e.g., TypedDict or dict)


class AnalysisStrategy(Protocol, Generic[R]):
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

    def analyze(self, data: pd.DataFrame) -> R:
        """
        Perform analysis on the provided DataFrame.

        Parameters
        ----------
        data : pandas.DataFrame
            Input DataFrame with Instagram message data.

        Returns
        -------
        R
            Results of the analysis, format depends on the strategy.

        """
        ...

    def save_results(self, results: R, output_dir: Path) -> None:
        """
        Save analysis results to the specified directory.

        Parameters
        ----------
        results : R
            Results of the analysis to be saved.
        output_dir : pathlib.Path
            Directory path where results will be saved.

        """
        ...
