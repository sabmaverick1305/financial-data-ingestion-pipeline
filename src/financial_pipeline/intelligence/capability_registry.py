"""Capability registry that turns research actions into executable work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from financial_pipeline.intelligence.research_plan import (
    ActionObservation,
    ActionStatus,
    ActionType,
    ResearchAction,
)

CapabilityHandler = Callable[[ResearchAction], Any]


@dataclass(frozen=True)
class CapabilityResult:
    """Normalized capability output consumed by the registry."""

    result: Any
    evidence_refs: tuple[str, ...] = ()
    status: ActionStatus = ActionStatus.SUCCEEDED
    tradeoff_reason: str | None = None


@dataclass(frozen=True)
class Capability:
    action_type: ActionType
    handler: CapabilityHandler
    trusted: bool = True


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[ActionType, Capability] = {}

    def register(
        self,
        action_type: ActionType,
        handler: CapabilityHandler,
        *,
        trusted: bool = True,
    ) -> None:
        if action_type in self._capabilities:
            raise ValueError(f"capability already registered: {action_type}")
        self._capabilities[action_type] = Capability(action_type, handler, trusted)

    def has(self, action_type: ActionType) -> bool:
        return action_type in self._capabilities

    def get(self, action_type: ActionType) -> Capability:
        try:
            return self._capabilities[action_type]
        except KeyError as exc:
            raise KeyError(f"no capability registered for: {action_type}") from exc

    def execute(self, action: ResearchAction) -> ActionObservation:
        capability = self.get(action.action_type)
        try:
            raw_result = capability.handler(action)
            if isinstance(raw_result, CapabilityResult):
                result = raw_result.result
                evidence_refs = raw_result.evidence_refs
                status = raw_result.status
                tradeoff_reason = raw_result.tradeoff_reason
            else:
                result = raw_result
                evidence_refs = ()
                status = ActionStatus.SUCCEEDED
                tradeoff_reason = None
            return ActionObservation(
                action_id=action.action_id,
                action_type=action.action_type,
                status=status,
                result=result,
                evidence_refs=evidence_refs,
                tradeoff_reason=tradeoff_reason,
            )
        except Exception as exc:
            return ActionObservation(
                action_id=action.action_id,
                action_type=action.action_type,
                status=ActionStatus.FAILED,
                error=str(exc),
            )
