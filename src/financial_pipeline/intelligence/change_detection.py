"""Deterministic change detection for validated financial facts."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from financial_pipeline.intelligence.computation import ComputationEngine
from financial_pipeline.intelligence.contracts import ScenarioDataContract


class ChangeDirection(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


@dataclass(frozen=True)
class MetricChange:
    previous_value: Decimal
    current_value: Decimal
    absolute_change: Decimal
    percentage_change: Decimal | None
    direction: ChangeDirection


class ChangeDetector:
    def __init__(self, computation: ComputationEngine | None = None) -> None:
        self._computation = computation or ComputationEngine()

    def detect(self, contract: ScenarioDataContract) -> MetricChange:
        if not contract.is_ready:
            raise ValueError("scenario data contract must be ready before change detection")
        assert contract.current_fact is not None
        assert contract.previous_fact is not None

        result = self._computation.percentage_change(
            contract.previous_fact.value,
            contract.current_fact.value,
        )
        direction = (
            ChangeDirection.UP if result.absolute_change > 0
            else ChangeDirection.DOWN if result.absolute_change < 0
            else ChangeDirection.FLAT
        )
        return MetricChange(
            previous_value=result.previous,
            current_value=result.current,
            absolute_change=result.absolute_change,
            percentage_change=result.percentage_change,
            direction=direction,
        )
