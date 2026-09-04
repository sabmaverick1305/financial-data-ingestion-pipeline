"""Canonical domain models for the FIES intelligence layer."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum


class FactQuality(StrEnum):
    VALIDATED = "validated"
    PARTIAL = "partial"
    REJECTED = "rejected"


class CoverageStatus(StrEnum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    EMPTY = "empty"
    INVALID = "invalid"


@dataclass(frozen=True)
class SourceProvenance:
    """Fact-level lineage. Pipeline-run lineage stays in services.lineage."""

    source_name: str
    source_document_id: str | None = None
    source_ref: str | None = None
    s3_raw_key: str | None = None
    s3_processed_key: str | None = None
    table_asset_id: str | None = None
    page_number: int | None = None
    row_reference: str | None = None
    extraction_method: str | None = None

    def __post_init__(self) -> None:
        if not self.source_name.strip():
            raise ValueError("source_name must not be blank")
        if self.page_number is not None and self.page_number < 1:
            raise ValueError("page_number must be >= 1")


@dataclass(frozen=True)
class FinancialFact:
    """One normalized financial observation."""

    entity_type: str
    entity: str
    metric: str
    value: Decimal
    unit: str
    period_year: int
    provenance: SourceProvenance
    period_month: int | None = None
    canonical_entity_id: str | None = None
    quality: FactQuality = FactQuality.VALIDATED

    def __post_init__(self) -> None:
        for field_name, field_value in (
            ("entity_type", self.entity_type),
            ("entity", self.entity),
            ("metric", self.metric),
            ("unit", self.unit),
        ):
            if not field_value.strip():
                raise ValueError(f"{field_name} must not be blank")
        if self.period_year < 1900:
            raise ValueError("period_year must be >= 1900")
        if self.period_month is not None and not 1 <= self.period_month <= 12:
            raise ValueError("period_month must be between 1 and 12")


@dataclass(frozen=True)
class DataCoverage:
    """Expected-versus-observed monthly coverage for one reporting year."""

    year: int
    expected_periods: tuple[int, ...]
    observed_periods: tuple[int, ...]
    duplicate_periods: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.year < 1900:
            raise ValueError("year must be >= 1900")
        for name, periods in (
            ("expected_periods", self.expected_periods),
            ("observed_periods", self.observed_periods),
            ("duplicate_periods", self.duplicate_periods),
        ):
            invalid = [period for period in periods if not 1 <= period <= 12]
            if invalid:
                raise ValueError(f"{name} contains invalid month(s): {invalid}")
        if len(set(self.expected_periods)) != len(self.expected_periods):
            raise ValueError("expected_periods must not contain duplicates")

    @classmethod
    def monthly(
        cls,
        *,
        year: int,
        observed_months: list[int] | tuple[int, ...],
        expected_through_month: int = 12,
    ) -> DataCoverage:
        """Build monthly coverage without assuming the current reporting cutoff."""

        if not 1 <= expected_through_month <= 12:
            raise ValueError("expected_through_month must be between 1 and 12")

        seen: set[int] = set()
        duplicates: set[int] = set()
        for month in observed_months:
            if month in seen:
                duplicates.add(month)
            seen.add(month)

        return cls(
            year=year,
            expected_periods=tuple(range(1, expected_through_month + 1)),
            observed_periods=tuple(sorted(seen)),
            duplicate_periods=tuple(sorted(duplicates)),
        )

    @property
    def missing_periods(self) -> tuple[int, ...]:
        observed = set(self.observed_periods)
        return tuple(period for period in self.expected_periods if period not in observed)

    @property
    def unexpected_periods(self) -> tuple[int, ...]:
        expected = set(self.expected_periods)
        return tuple(period for period in self.observed_periods if period not in expected)

    @property
    def completeness_ratio(self) -> float:
        if not self.expected_periods:
            return 1.0
        observed_expected = set(self.observed_periods).intersection(self.expected_periods)
        return len(observed_expected) / len(self.expected_periods)

    @property
    def status(self) -> CoverageStatus:
        if self.duplicate_periods or self.unexpected_periods:
            return CoverageStatus.INVALID
        if not self.observed_periods:
            return CoverageStatus.EMPTY
        if not self.missing_periods:
            return CoverageStatus.COMPLETE
        return CoverageStatus.PARTIAL

    @property
    def is_complete(self) -> bool:
        return self.status is CoverageStatus.COMPLETE
