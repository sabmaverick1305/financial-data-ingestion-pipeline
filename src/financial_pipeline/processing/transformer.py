from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger()


class DataTransformer:
    """Applies transformations to raw ingested data."""

    def __init__(self, source: str) -> None:
        self._log = log.bind(source=source)

    def to_dataframe(self, data: list[dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(data)
        self._log.debug("transformer.to_dataframe", rows=len(df), columns=list(df.columns))
        return df

    def normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        return df

    def cast_numeric(self, df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        for col in columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    def drop_duplicates(self, df: pd.DataFrame, subset: list[str] | None = None) -> pd.DataFrame:
        before = len(df)
        df = df.drop_duplicates(subset=subset)
        dropped = before - len(df)
        if dropped:
            self._log.info("transformer.dedup", dropped=dropped)
        return df
