"""Deterministic policy guardrails for planner output."""

from __future__ import annotations

from financial_pipeline.intelligence.evidence import EvidenceDimension, EvidenceImportance, EvidenceRequirement
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan


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

    def complete_plan(self, query: str, plan: ResearchPlan) -> ResearchPlan:
        """Deterministically add required flagship actions omitted by the LLM."""
        missing = self.missing_required_actions(query, plan)
        if not missing:
            return plan

        defaults = {
            ActionType.DISCOVER_CATEGORIES: ResearchAction(
                ActionType.DISCOVER_CATEGORIES,
                rationale="deterministic policy completion",
            ),
            ActionType.DISCOVER_FUNDS: ResearchAction(
                ActionType.DISCOVER_FUNDS,
                rationale="deterministic policy completion",
                parameters={"limit": 20},
            ),
            ActionType.FETCH_PERFORMANCE: ResearchAction(
                ActionType.FETCH_PERFORMANCE,
                metrics=("return_1y", "return_3y_cagr", "return_5y_cagr"),
                rationale="deterministic policy completion",
            ),
            ActionType.COMPUTE_RISK: ResearchAction(
                ActionType.COMPUTE_RISK,
                metrics=("volatility", "sharpe_ratio", "max_drawdown"),
                rationale="deterministic policy completion",
            ),
            ActionType.COMPARE_PEERS: ResearchAction(
                ActionType.COMPARE_PEERS,
                metrics=("percentile_rank", "peer_outperformance"),
                parameters={"rank_by": "return_3y_cagr"},
                rationale="deterministic policy completion",
            ),
            ActionType.FETCH_FLOWS: ResearchAction(
                ActionType.FETCH_FLOWS,
                metrics=("net_inflow",),
                rationale="deterministic policy completion",
            ),
            ActionType.FETCH_AUM: ResearchAction(
                ActionType.FETCH_AUM,
                metrics=("aum",),
                rationale="deterministic policy completion",
            ),
            ActionType.RETRIEVE_EVIDENCE: ResearchAction(
                ActionType.RETRIEVE_EVIDENCE,
                evidence_types=("prospectus", "fact_sheet", "strategy", "disclosures"),
                rationale="deterministic policy completion",
            ),
            ActionType.CHECK_CONTRADICTIONS: ResearchAction(
                ActionType.CHECK_CONTRADICTIONS,
                checks=("performance_consistency", "style_drift"),
                rationale="deterministic policy completion",
            ),
        }
        return ResearchPlan(
            objective=plan.objective,
            actions=(*plan.actions, *(defaults[action] for action in missing)),
            assumptions=plan.assumptions,
            plan_id=plan.plan_id,
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
