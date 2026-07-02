from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger()


class ValidationError(Exception):
    pass


class DataValidator:
    """Validates a DataFrame before storage."""

    def __init__(self, source: str) -> None:
        self._log = log.bind(source=source)

    def require_columns(self, df: pd.DataFrame, columns: list[str]) -> None:
        missing = [c for c in columns if c not in df.columns]
        if missing:
            raise ValidationError(f"Missing required columns: {missing}")

    def check_nulls(self, df: pd.DataFrame, columns: list[str], max_null_pct: float = 0.0) -> None:
        for col in columns:
            null_pct = df[col].isna().mean()
            if null_pct > max_null_pct:
                raise ValidationError(f"Column '{col}' has {null_pct:.1%} nulls (limit: {max_null_pct:.1%})")

    def validate(self, df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
        """Run all validations defined in schema dict."""
        required = schema.get("required_columns", [])
        non_null = schema.get("non_null_columns", [])
        numeric = schema.get("numeric_columns", [])

        self.require_columns(df, required)
        self.check_nulls(df, non_null)

        for col in numeric:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                raise ValidationError(f"Column '{col}' must be numeric")

        self._log.info("validator.passed", rows=len(df))
        return df
