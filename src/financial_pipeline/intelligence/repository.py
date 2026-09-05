"""Repository access for canonical FIES financial facts."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.engine import Engine

from financial_pipeline.intelligence.amfi_mapper import build_monthly_coverage, map_amfi_fund_stats_row
from financial_pipeline.intelligence.models import DataCoverage, FinancialFact


_ALLOWED_METRICS = {
    "no_of_schemes",
    "no_of_folios",
    "funds_mobilized",
    "redemption",
    "net_inflow",
    "aum",
    "avg_aum",
}


@dataclass(frozen=True)
class FinancialTimeSeries:
    entity_type: str
    entity: str
    metric: str
    facts: tuple[FinancialFact, ...]
    coverage_by_year: dict[int, DataCoverage]

    @property
    def is_complete(self) -> bool:
        return all(coverage.is_complete for coverage in self.coverage_by_year.values())


class FinancialFactRepository:
    """Read structured AMFI facts without exposing table details downstream."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_amfi_series(
        self,
        *,
        fund_category: str,
        metric: str,
        year_from: int,
        year_to: int,
        expected_through_month_by_year: dict[int, int] | None = None,
    ) -> FinancialTimeSeries:
        if metric not in _ALLOWED_METRICS:
            raise ValueError(f"unsupported AMFI metric: {metric}")
        if year_from > year_to:
            raise ValueError("year_from must be <= year_to")

        sql = f"""
            SELECT
                afs.period_year,
                afs.period_month,
                afs.fund_category,
                afs.{metric},
                afs.source_document_id,
                dm.original_url,
                dm.s3_raw_key,
                dm.s3_processed_key
            FROM amfi_fund_stats afs
            LEFT JOIN document_metadata dm
              ON dm.document_id = afs.source_document_id
            WHERE afs.fund_category = :fund_category
              AND afs.period_year BETWEEN :year_from AND :year_to
              AND afs.{metric} IS NOT NULL
            ORDER BY afs.period_year ASC, afs.period_month ASC
        """

        params = {
            "fund_category": fund_category,
            "year_from": year_from,
            "year_to": year_to,
        }
        with self._engine.connect() as conn:
            rows = [dict(row) for row in conn.execute(text(sql), params).mappings().all()]

        facts: list[FinancialFact] = []
        for row in rows:
            facts.extend(map_amfi_fund_stats_row(row))
        facts = [fact for fact in facts if fact.metric == metric]

        cutoffs = expected_through_month_by_year or {}
        coverage_by_year: dict[int, DataCoverage] = {}
        for year in range(year_from, year_to + 1):
            year_rows = [row for row in rows if int(row["period_year"]) == year]
            coverage_by_year[year] = build_monthly_coverage(
                year_rows,
                year=year,
                expected_through_month=cutoffs.get(year, 12),
            )

        return FinancialTimeSeries(
            entity_type="fund_category",
            entity=fund_category,
            metric=metric,
            facts=tuple(facts),
            coverage_by_year=coverage_by_year,
        )
