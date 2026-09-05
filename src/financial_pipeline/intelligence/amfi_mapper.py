"""Map structured AMFI category rows into canonical FIES financial facts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Any

from financial_pipeline.intelligence.models import DataCoverage, FinancialFact, SourceProvenance


_METRIC_UNITS: dict[str, str] = {
    "no_of_schemes": "COUNT",
    "no_of_folios": "COUNT",
    "funds_mobilized": "INR_CRORE",
    "redemption": "INR_CRORE",
    "net_inflow": "INR_CRORE",
    "aum": "INR_CRORE",
    "avg_aum": "INR_CRORE",
}


def _to_decimal(value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value
    return Decimal(str(value))


def map_amfi_fund_stats_row(row: Mapping[str, Any]) -> list[FinancialFact]:
    """Expand one amfi_fund_stats row into canonical metric facts."""
    year = int(row["period_year"])
    month = int(row["period_month"])
    category = str(row["fund_category"])

    provenance = SourceProvenance(
        source_name="AMFI",
        source_document_id=str(row["source_document_id"]) if row.get("source_document_id") else None,
        source_ref=row.get("original_url"),
        s3_raw_key=row.get("s3_raw_key"),
        s3_processed_key=row.get("s3_processed_key"),
        extraction_method=row.get("extraction_method") or "amfi_fund_stats",
    )

    facts: list[FinancialFact] = []
    for metric, unit in _METRIC_UNITS.items():
        value = _to_decimal(row.get(metric))
        if value is None:
            continue
        facts.append(
            FinancialFact(
                entity_type="fund_category",
                entity=category,
                metric=metric,
                value=value,
                unit=unit,
                period_year=year,
                period_month=month,
                provenance=provenance,
            )
        )
    return facts


def build_monthly_coverage(
    rows: Sequence[Mapping[str, Any]],
    *,
    year: int,
    expected_through_month: int = 12,
) -> DataCoverage:
    """Build coverage from rows already filtered to one logical series."""
    observed_months = [
        int(row["period_month"])
        for row in rows
        if int(row["period_year"]) == year and row.get("period_month") is not None
    ]
    return DataCoverage.monthly(
        year=year,
        observed_months=observed_months,
        expected_through_month=expected_through_month,
    )
