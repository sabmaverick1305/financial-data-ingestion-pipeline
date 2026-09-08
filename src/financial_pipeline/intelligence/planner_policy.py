"""Deterministic policy guardrails for planner output."""

from __future__ import annotations

from financial_pipeline.intelligence.evidence import EvidenceDimension, EvidenceImportance, EvidenceRequirement
from financial_pipeline.intelligence.research_plan import ActionType, ResearchPlan


_FLAGSHIP_MINIMUM = (
    EvidenceDimension.CATEGORY,
    EvidenceDimension.FUND_DISCOVERY,
    EvidenceDimension.PERFORMANCE,
    EvidenceDimension.RISK,
    EvidenceDimension.PEER_COMPARISON,
    EvidenceDimension.FLOWS,
    EvidenceDimension.AUM,
    EvidenceDimension.DOCUMENTARY,
    EvidenceDimension.CONTRADICTION,
)


class PlannerPolicy:
    """Validate planner output independently of the model that produced it."""

    def required_actions(self, query: str) -> tuple[ActionType, ...]:
        q = query.lower()
        if "best mutual fund" in q or "best mutual funds" in q:
            return (
                ActionType.DISCOVER_CATEGORIES,
                ActionType.DISCOVER_FUNDS,
                ActionType.FETCH_PERFORMANCE,
                ActionType.COMPUTE_RISK,
                ActionType.COMPARE_PEERS,
                ActionType.FETCH_FLOWS,
                ActionType.FETCH_AUM,
                ActionType.RETRIEVE_EVIDENCE,
                ActionType.CHECK_CONTRADICTIONS,
            )
        return ()

    def missing_required_actions(
        self,
        query: str,
        plan: ResearchPlan,
    ) -> tuple[ActionType, ...]:
        present = {action.action_type for action in plan.actions}
        return tuple(
            action_type
            for action_type in self.required_actions(query)
            if action_type not in present
        )

    def validate_plan(self, query: str, plan: ResearchPlan) -> None:
        if not plan.actions:
            raise ValueError("planner produced an empty action plan")

        action_types = [action.action_type for action in plan.actions]
        if len(action_types) != len(set(action_types)):
            raise ValueError("planner produced duplicate action types")

        missing = self.missing_required_actions(query, plan)
        if missing:
            names = ", ".join(action.value for action in missing)
            raise ValueError(f"planner omitted required flagship actions: {names}")

    def enforce_requirements(
        self,
        query: str,
        requested: tuple[EvidenceRequirement, ...],
    ) -> tuple[EvidenceRequirement, ...]:
        q = query.lower()
        if "best mutual fund" not in q and "best mutual funds" not in q:
            return requested

        by_dimension = {requirement.dimension: requirement for requirement in requested}
        soft_dimensions = {
            EvidenceDimension.FLOWS,
            EvidenceDimension.AUM,
            EvidenceDimension.DOCUMENTARY,
        }
        enforced = {}
        for dimension in _FLAGSHIP_MINIMUM:
            requested_requirement = by_dimension.get(dimension)
            importance = (
                EvidenceImportance.SOFT
                if dimension in soft_dimensions
                else EvidenceImportance.HARD
            )
            enforced[dimension] = EvidenceRequirement(
                dimension=dimension,
                required=True,
                allow_partial=(
                    True if importance is EvidenceImportance.SOFT
                    else bool(requested_requirement.allow_partial) if requested_requirement else False
                ),
                importance=importance,
            )
        return tuple(enforced[dimension] for dimension in _FLAGSHIP_MINIMUM)
