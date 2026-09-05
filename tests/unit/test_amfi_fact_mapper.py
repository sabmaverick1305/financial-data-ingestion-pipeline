from decimal import Decimal

from financial_pipeline.intelligence.amfi_mapper import build_monthly_coverage, map_amfi_fund_stats_row
from financial_pipeline.intelligence.models import CoverageStatus


def test_map_amfi_row_expands_metrics_with_correct_units() -> None:
    row = {
        "period_year": 2026,
        "period_month": 8,
        "fund_category": "Large Cap Fund",
        "no_of_schemes": 31,
        "no_of_folios": 12345678,
        "funds_mobilized": Decimal("9200.50"),
        "redemption": Decimal("5100.25"),
        "net_inflow": Decimal("4100.25"),
        "aum": Decimal("345678.90"),
        "avg_aum": Decimal("340000.10"),
        "source_document_id": "doc-123",
        "original_url": "https://example.test/amfi.pdf",
        "s3_raw_key": "bronze/amfi/report.pdf",
        "s3_processed_key": "processed/v1/amfi/report",
    }

    facts = map_amfi_fund_stats_row(row)
    by_metric = {fact.metric: fact for fact in facts}

    assert len(facts) == 7
    assert by_metric["net_inflow"].value == Decimal("4100.25")
    assert by_metric["net_inflow"].unit == "INR_CRORE"
    assert by_metric["no_of_folios"].unit == "COUNT"
    assert by_metric["aum"].provenance.source_document_id == "doc-123"
    assert by_metric["aum"].provenance.s3_raw_key == "bronze/amfi/report.pdf"


def test_map_amfi_row_skips_null_metrics() -> None:
    row = {
        "period_year": 2026,
        "period_month": 8,
        "fund_category": "Large Cap Fund",
        "no_of_schemes": None,
        "no_of_folios": None,
        "funds_mobilized": None,
        "redemption": None,
        "net_inflow": Decimal("-120.50"),
        "aum": Decimal("1000"),
        "avg_aum": None,
        "source_document_id": None,
    }

    facts = map_amfi_fund_stats_row(row)

    assert {fact.metric for fact in facts} == {"net_inflow", "aum"}
    assert next(fact for fact in facts if fact.metric == "net_inflow").value == Decimal("-120.50")


def test_build_monthly_coverage_exposes_missing_months() -> None:
    rows = [
        {"period_year": 2025, "period_month": month}
        for month in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)
    ]

    coverage = build_monthly_coverage(rows, year=2025)

    assert coverage.status is CoverageStatus.PARTIAL
    assert coverage.missing_periods == (4,)


def test_build_monthly_coverage_preserves_duplicates_as_invalid() -> None:
    rows = [
        {"period_year": 2025, "period_month": 1},
        {"period_year": 2025, "period_month": 2},
        {"period_year": 2025, "period_month": 2},
    ]

    coverage = build_monthly_coverage(rows, year=2025, expected_through_month=2)

    assert coverage.status is CoverageStatus.INVALID
    assert coverage.duplicate_periods == (2,)
