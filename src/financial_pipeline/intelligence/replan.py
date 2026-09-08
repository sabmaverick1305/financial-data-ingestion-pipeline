"""Deterministic replan decisions derived from missing evidence."""

from __future__ import annotations

from dataclasses import dataclass

from financial_pipeline.intelligence.evidence import EvidenceEvaluation, EvidenceEvaluator
from financial_pipeline.intelligence.research_plan import ResearchAction, ResearchPlan


@dataclass(frozen=True)
class ReplanDecision:
    should_replan: bool
    reason: str
    actions: tuple[ResearchAction, ...] = ()


class EvidenceReplanner:
    def decide(self, evaluation: EvidenceEvaluation) -> ReplanDecision:
        if evaluation.is_sufficient:
            return ReplanDecision(
                should_replan=False,
                reason="required evidence is sufficient",
            )

        dimensions = tuple(dict.fromkeys((*evaluation.failed, *evaluation.missing)))
        actions = tuple(
            ResearchAction(
                action_type=EvidenceEvaluator.action_for_dimension(dimension),
                rationale=f"fill missing or failed evidence dimension: {dimension.value}",
            )
            for dimension in dimensions
        )
        return ReplanDecision(
            should_replan=True,
            reason="required evidence is missing or failed",
            actions=actions,
        )

    def build_plan(self, *, objective: str, decision: ReplanDecision) -> ResearchPlan:
        if not decision.should_replan:
            raise ValueError("cannot build replan when no replan is required")
        return ResearchPlan(
            objective=objective,
            actions=decision.actions,
        )
