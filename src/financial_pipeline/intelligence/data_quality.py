"""Production data-quality gate for mutual-fund candidate evidence."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import date, timedelta
import re

@dataclass(frozen=True)
class DataQualityIssue:
    code: str
    reason: str

@dataclass(frozen=True)
class DataQualityResult:
    valid: bool
    issues: tuple[DataQualityIssue, ...]

class FundDataQualityGate:
    def validate_candidate(self, row: dict, *, as_of: date | None = None) -> DataQualityResult:
        issues: list[DataQualityIssue] = []
        today = as_of or date.today()
        latest_nav_date = row.get("latest_nav_date")
        if latest_nav_date is None or latest_nav_date < today - timedelta(days=14):
            issues.append(DataQualityIssue("stale_nav", "latest NAV is older than 14 days"))

        return_1y = self._num(row.get("return_1y"))
        return_3y = self._num(row.get("return_3y_cagr"))
        volatility = self._num(row.get("rolling_volatility"))
        if return_1y is not None and not (-100.0 <= return_1y <= 300.0):
            issues.append(DataQualityIssue("implausible_return_1y", f"1Y return out of plausible bounds: {return_1y}"))
        if return_3y is not None and not (-100.0 <= return_3y <= 150.0):
            issues.append(DataQualityIssue("implausible_return_3y", f"3Y CAGR out of plausible bounds: {return_3y}"))
        if volatility is not None and not (0.0 <= volatility <= 150.0):
            issues.append(DataQualityIssue("implausible_volatility", f"volatility out of plausible bounds: {volatility}"))

        name = str(row.get("scheme_name") or "").lower()
        category = str(row.get("category") or "").lower()
        if any(token in name for token in ("liquid fund", "ultra short", "overnight fund", "money market")) and any(
            token in category for token in ("large cap", "mid cap", "small cap", "multi cap", "flexi cap")
        ):
            issues.append(DataQualityIssue("category_name_mismatch", "debt/liquid-style scheme is classified as equity category"))

        return DataQualityResult(valid=not issues, issues=tuple(issues))

    @staticmethod
    def _num(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None
