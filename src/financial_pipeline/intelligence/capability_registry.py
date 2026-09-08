"""Capability registry that turns research actions into executable work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from financial_pipeline.intelligence.capability_contracts import CapabilityContract
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
    contract: CapabilityContract | None = None


class CapabilityRegistry:
    def __init__(self) -> None:
        self._capabilities: dict[ActionType, Capability] = {}

    def register(
        self,
        action_type: ActionType,
        handler: CapabilityHandler,
        *,
        trusted: bool = True,
        contract: CapabilityContract | None = None,
    ) -> None:
        if action_type in self._capabilities:
            raise ValueError(f"capability already registered: {action_type}")
        self._capabilities[action_type] = Capability(action_type, handler, trusted, contract)

    def has(self, action_type: ActionType) -> bool:
        return action_type in self._capabilities

    def get(self, action_type: ActionType) -> Capability:
        try:
            return self._capabilities[action_type]
        except KeyError as exc:
            raise KeyError(f"no capability registered for: {action_type}") from exc

    def supported_metrics(self, action_type: ActionType) -> tuple[str, ...] | None:
        """Return canonical metrics advertised by a capability contract."""
        capability = self.get(action_type)
        if capability.contract is None:
            return None
        return capability.contract.supported_metrics

    def supported_evidence_types(self, action_type: ActionType) -> tuple[str, ...] | None:
        """Return canonical evidence types advertised by a capability contract."""
        capability = self.get(action_type)
        if capability.contract is None:
            return None
        return capability.contract.supported_evidence_types

    def execute(self, action: ResearchAction) -> ActionObservation:
        capability = self.get(action.action_type)
        try:
            executable_action = action
            unsupported_metrics: tuple[str, ...] = ()
            unsupported_evidence: tuple[str, ...] = ()
            if capability.contract is not None:
                executable_action, unsupported_metrics, unsupported_evidence = (
                    capability.contract.normalize(action)
                )
                requested_any = bool(action.metrics or action.evidence_types)
                executable_any = bool(
                    executable_action.metrics or executable_action.evidence_types
                )
                if requested_any and not executable_any:
                    rejected = ", ".join((*unsupported_metrics, *unsupported_evidence))
                    return ActionObservation(
                        action_id=action.action_id,
                        action_type=action.action_type,
                        status=ActionStatus.FAILED,
                        tradeoff_reason=(
                            "capability contract rejected all requested semantics: "
                            + rejected
                        ),
                        error="unsupported capability semantics",
                    )

            raw_result = capability.handler(executable_action)
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
            if unsupported_metrics or unsupported_evidence:
                reasons: list[str] = []
                if unsupported_metrics:
                    reasons.append(
                        "unsupported metrics omitted by capability contract: "
                        + ", ".join(unsupported_metrics)
                    )
                if unsupported_evidence:
                    reasons.append(
                        "unsupported evidence types omitted by capability contract: "
                        + ", ".join(unsupported_evidence)
                    )
                contract_reason = "; ".join(reasons)
                if status is ActionStatus.SUCCEEDED:
                    status = ActionStatus.PARTIAL
                tradeoff_reason = (
                    f"{tradeoff_reason}; {contract_reason}"
                    if tradeoff_reason
                    else contract_reason
                )

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
