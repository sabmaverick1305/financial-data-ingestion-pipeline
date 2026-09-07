"""Data-readiness contracts for intelligence scenarios."""

from __future__ import annotations

from dataclasses import dataclass

from financial_pipeline.intelligence.models import DataCoverage, FinancialFact
from financial_pipeline.intelligence.scenario import GoldenScenario


@dataclass(frozen=True)
class ScenarioDataContract:
    scenario: GoldenScenario
    current_fact: FinancialFact | None
    previous_fact: FinancialFact | None
    coverage: DataCoverage

    @property
    def has_current_fact(self) -> bool:
        return self.current_fact is not None

    @property
    def has_previous_fact(self) -> bool:
        return self.previous_fact is not None

    @property
    def has_valid_coverage(self) -> bool:
        return self.coverage.is_complete

    @property
    def is_ready(self) -> bool:
        return self.has_current_fact and self.has_previous_fact and self.has_valid_coverage
