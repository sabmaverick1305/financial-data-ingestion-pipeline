from decimal import Decimal

import pytest

from financial_pipeline.intelligence.models import (
    CoverageStatus,
    DataCoverage,
    FactQuality,
    FinancialFact,
    SourceProvenance,
)


def test_full_year_coverage_is_complete() -> None:
    coverage = DataCoverage.monthly(year=2025, observed_months=list(range(1, 13)))
    assert coverage.status is CoverageStatus.COMPLETE
    assert coverage.is_complete
    assert coverage.missing_periods == ()
    assert coverage.completeness_ratio == 1.0


def test_missing_months_are_reported() -> None:
    coverage = DataCoverage.monthly(year=2025, observed_months=[1, 2, 3, 5, 6, 8, 9, 10, 11, 12])
    assert coverage.status is CoverageStatus.PARTIAL
    assert coverage.missing_periods == (4, 7)
    assert coverage.completeness_ratio == pytest.approx(10 / 12)


def test_duplicate_month_makes_coverage_invalid() -> None:
    coverage = DataCoverage.monthly(year=2025, observed_months=[1, 2, 2, 3])
    assert coverage.status is CoverageStatus.INVALID
    assert coverage.duplicate_periods == (2,)


def test_partial_reporting_year_uses_explicit_expected_month() -> None:
    coverage = DataCoverage.monthly(
        year=2026,
        observed_months=[1, 2, 3, 4, 5, 6, 7, 8],
        expected_through_month=8,
    )
    assert coverage.status is CoverageStatus.COMPLETE
    assert coverage.expected_periods == (1, 2, 3, 4, 5, 6, 7, 8)


def test_unexpected_month_makes_coverage_invalid() -> None:
    coverage = DataCoverage.monthly(year=2026, observed_months=[1, 2, 3, 4], expected_through_month=3)
    assert coverage.status is CoverageStatus.INVALID
    assert coverage.unexpected_periods == (4,)


def test_financial_fact_keeps_decimal_and_fact_level_provenance() -> None:
    provenance = SourceProvenance(
        source_name="AMFI",
        source_document_id="doc-123",
        s3_raw_key="bronze/amfi/research/2026/08/report.pdf",
        s3_processed_key="silver/amfi/research/2026/08/report",
        table_asset_id="table-7",
        page_number=17,
        row_reference="Large Cap Fund",
        extraction_method="structured_table",
    )
    fact = FinancialFact(
        entity_type="fund_category",
        entity="Large Cap Fund",
        metric="net_inflow",
        value=Decimal("6200.25"),
        unit="INR_CRORE",
        period_year=2026,
        period_month=8,
        provenance=provenance,
        quality=FactQuality.VALIDATED,
    )
    assert fact.value == Decimal("6200.25")
    assert fact.provenance.page_number == 17
    assert fact.quality is FactQuality.VALIDATED


def test_invalid_fact_month_is_rejected() -> None:
    provenance = SourceProvenance(source_name="AMFI")
    with pytest.raises(ValueError, match="period_month"):
        FinancialFact(
            entity_type="fund_category",
            entity="Large Cap Fund",
            metric="aum",
            value=Decimal("1"),
            unit="INR_CRORE",
            period_year=2026,
            period_month=13,
            provenance=provenance,
        )
