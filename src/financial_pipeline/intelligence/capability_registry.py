"""Capability registry that turns research actions into executable work."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from financial_pipeline.intelligence.capability_contracts import CapabilityContract
from financial_pipeline.intelligence.capability_metadata import CapabilityMetadata
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
    metadata: CapabilityMetadata | None = None


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
        metadata: CapabilityMetadata | None = None,
    ) -> None:
        if action_type in self._capabilities:
            raise ValueError(f"capability already registered: {action_type}")
        self._capabilities[action_type] = Capability(action_type, handler, trusted, contract, metadata)

    def has(self, action_type: ActionType) -> bool:
        return action_type in self._capabilities

    def get(self, action_type: ActionType) -> Capability:
        try:
            return self._capabilities[action_type]
        except KeyError as exc:
            raise KeyError(f"no capability registered for: {action_type}") from exc

    def metadata(self, action_type: ActionType) -> CapabilityMetadata | None:
        return self.get(action_type).metadata

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

    @staticmethod
    def _classify_exception(exc: Exception) -> tuple[str, bool, bool]:
        message = str(exc).lower()
        if (
            "not_found_error" in message
            or ("404" in message and "model" in message)
            or "model not found" in message
        ):
            return "configuration", False, False
        if any(token in message for token in ("timeout", "timed out", "429", "rate limit", "503", "502", "connection reset")):
            return "execution", True, False
        if any(token in message for token in ("scheme_code is required", "scheme_code or scheme_codes is required", "category is required", "category or categories is required")):
            return "dependency", False, True
        if isinstance(exc, ValueError):
            return "data_or_contract", False, True
        return "execution", False, False

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
                    metric_only = bool(action.metrics) and not action.evidence_types and not unsupported_evidence
                    evidence_only = bool(action.evidence_types) and not action.metrics and not unsupported_metrics
                    if metric_only:
                        reason = "capability contract rejected all requested metrics: " + rejected
                        error = "unsupported capability metrics"
                    elif evidence_only:
                        reason = "capability contract rejected all requested evidence types: " + rejected
                        error = "unsupported capability evidence types"
                    else:
                        reason = "capability contract rejected all requested semantics: " + rejected
                        error = "unsupported capability semantics"
                    return ActionObservation(
                        action_id=action.action_id,
                        action_type=action.action_type,
                        status=ActionStatus.FAILED,
                        tradeoff_reason=reason,
                        error=error,
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
            failure_domain, retryable, replannable = self._classify_exception(exc)
            return ActionObservation(
                action_id=action.action_id,
                action_type=action.action_type,
                status=ActionStatus.FAILED,
                error=str(exc),
                failure_domain=failure_domain,
                retryable=retryable,
                replannable=replannable,
            )
