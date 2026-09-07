"""Golden business scenario contracts for FIES intelligence."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ScenarioStatus(StrEnum):
    READY = "ready"
    INSUFFICIENT_DATA = "insufficient_data"
    NO_CHANGE = "no_change"
    INVESTIGATION_REQUIRED = "investigation_required"


@dataclass(frozen=True)
class GoldenScenario:
    entity_type: str
    entity: str
    metric: str
    year: int
    month: int

    @property
    def scenario_id(self) -> str:
        return f"{self.entity_type}:{self.entity}:{self.metric}:{self.year}-{self.month:02d}"


@dataclass(frozen=True)
class ScenarioReadiness:
    scenario: GoldenScenario
    status: ScenarioStatus
    reason: str | None = None
