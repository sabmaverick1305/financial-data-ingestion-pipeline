"""Deterministic replan decisions derived from missing evidence."""

from __future__ import annotations

from dataclasses import dataclass

from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.evidence import EvidenceEvaluation, EvidenceEvaluator
from financial_pipeline.intelligence.research_plan import ResearchAction, ResearchPlan


@dataclass(frozen=True)
class ReplanDecision:
    should_replan: bool
    reason: str
    actions: tuple[ResearchAction, ...] = ()


class EvidenceReplanner:
    def __init__(self, registry: CapabilityRegistry | None = None) -> None:
        self._registry = registry

    def decide(self, evaluation: EvidenceEvaluation) -> ReplanDecision:
        if evaluation.is_sufficient:
            return ReplanDecision(
                should_replan=False,
                reason="required evidence is sufficient",
            )

        dimensions = tuple(dict.fromkeys((*evaluation.failed, *evaluation.missing)))
        actions = tuple(
            self._recovery_action(dimension)
            for dimension in dimensions
        )
        return ReplanDecision(
            should_replan=True,
            reason="required evidence is missing or failed",
            actions=actions,
        )

    def _recovery_action(self, dimension) -> ResearchAction:
        action_type = EvidenceEvaluator.action_for_dimension(dimension)
        metrics: tuple[str, ...] = ()
        rationale = f"fill missing or failed evidence dimension: {dimension.value}"

        if self._registry is not None and self._registry.has(action_type):
            supported = self._registry.supported_metrics(action_type)
            if supported:
                metrics = supported
                rationale += "; retry with capability-supported metrics"

        return ResearchAction(
            action_type=action_type,
            metrics=metrics,
            rationale=rationale,
        )

    def build_plan(self, *, objective: str, decision: ReplanDecision) -> ResearchPlan:
        if not decision.should_replan:
            raise ValueError("cannot build replan when no replan is required")
        return ResearchPlan(
            objective=objective,
            actions=decision.actions,
        )
