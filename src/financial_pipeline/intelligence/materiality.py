"""Policy-driven materiality assessment."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum

from financial_pipeline.intelligence.change_detection import MetricChange


class MaterialityLevel(StrEnum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass(frozen=True)
class MaterialityResult:
    is_material: bool
    level: MaterialityLevel
    reason: str


class MaterialityPolicy:
    def __init__(
        self,
        *,
        pct_threshold: Decimal = Decimal("20"),
        absolute_threshold: Decimal = Decimal("1000"),
    ) -> None:
        self._pct_threshold = pct_threshold
        self._absolute_threshold = absolute_threshold

    def evaluate(self, change: MetricChange) -> MaterialityResult:
        abs_change = abs(change.absolute_change)
        pct_material = (
            change.percentage_change is not None
            and abs(change.percentage_change) >= self._pct_threshold
        )
        absolute_material = abs_change >= self._absolute_threshold

        if pct_material and absolute_material:
            return MaterialityResult(True, MaterialityLevel.HIGH, "percentage and absolute thresholds exceeded")
        if pct_material or absolute_material:
            return MaterialityResult(True, MaterialityLevel.MEDIUM, "one materiality threshold exceeded")
        if abs_change > 0:
            return MaterialityResult(False, MaterialityLevel.LOW, "change below materiality thresholds")
        return MaterialityResult(False, MaterialityLevel.NONE, "no change detected")
