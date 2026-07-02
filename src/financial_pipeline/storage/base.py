from abc import ABC, abstractmethod

import pandas as pd


class BaseStorage(ABC):
    """Abstract base for all storage backends."""

    @abstractmethod
    def save(self, df: pd.DataFrame, table: str) -> int:
        """Persist a DataFrame. Returns the number of rows written."""
        ...

    @abstractmethod
    def load(self, table: str) -> pd.DataFrame:
        """Load a full table into a DataFrame."""
        ...
