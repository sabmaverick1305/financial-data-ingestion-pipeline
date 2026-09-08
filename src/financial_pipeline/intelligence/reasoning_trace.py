"""Structured audit trace for FIES reasoning executions."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any


class ReasoningTraceEventType(StrEnum):
    PLAN_STARTED = "plan_started"
    ACTION_STARTED = "action_started"
    ACTION_FINISHED = "action_finished"
    EVIDENCE_EVALUATED = "evidence_evaluated"
    REPLAN_CREATED = "replan_created"
    LOOP_STOPPED = "loop_stopped"


@dataclass(frozen=True)
class ReasoningTraceEvent:
    event_type: ReasoningTraceEventType
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    round: int | None = None
    plan_id: str | None = None
    action_id: str | None = None
    action_type: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    query: str
    events: list[ReasoningTraceEvent] = field(default_factory=list)

    def record(self, event_type: ReasoningTraceEventType, **kwargs: Any) -> None:
        self.events.append(ReasoningTraceEvent(event_type=event_type, **kwargs))

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "events": [
                {
                    "event_type": event.event_type.value,
                    "timestamp": event.timestamp,
                    "round": event.round,
                    "plan_id": event.plan_id,
                    "action_id": event.action_id,
                    "action_type": event.action_type,
                    "payload": event.payload,
                }
                for event in self.events
            ],
        }
