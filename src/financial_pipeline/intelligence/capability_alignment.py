"""Align planner actions to the registered production capability contracts."""
from __future__ import annotations

from dataclasses import replace

from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan

_ACTION_ALIASES: dict[ActionType, dict[str, str]] = {
    ActionType.FETCH_PERFORMANCE: {
        "ytd_return": "return_1y",
        "ytd_performance": "return_1y",
        "annual_return": "return_1y",
    },
    ActionType.COMPUTE_RISK: {
        "beta": "volatility",
        "sortino_ratio": "sharpe_ratio",
        "risk_adjusted_return": "sharpe_ratio",
    },
    ActionType.COMPARE_PEERS: {
        "percentile_ranking_by_return": "percentile_rank",
        "percentile_ranking_by_risk": "percentile_rank",
        "consistency_vs_peers": "peer_outperformance",
        "relative_performance": "peer_outperformance",
        "risk_adjusted_return": "percentile_rank",
    },
    ActionType.FETCH_AUM: {
        "total_assets_under_management": "aum",
        "current_assets_under_management": "aum",
        "aum_growth_1yr": "aum_trend",
    },
    ActionType.FETCH_FLOWS: {
        "flow_rate": "flow_trend",
        "flow_direction": "flow_trend",
        "net_flow_1yr": "net_inflow",
    },
    ActionType.COMPUTE_RETURNS: {
        "total_return": "total_return_3y",
        "excess_return_vs_benchmark": "annualized_return",
        "information_ratio": "annualized_return",
    },
}

_DEFAULTS: dict[ActionType, tuple[str, ...]] = {
    ActionType.FETCH_PERFORMANCE: (
        "return_1y",
        "return_3y_cagr",
        "return_5y_cagr",
        "return_10y_cagr",
    ),
    ActionType.COMPUTE_RISK: (
        "volatility",
        "sharpe_ratio",
        "max_drawdown",
    ),
    ActionType.COMPARE_PEERS: (
        "percentile_rank",
        "peer_outperformance",
    ),
    ActionType.COMPUTE_RETURNS: (
        "total_return_3y",
        "annualized_return",
    ),
}

class CapabilitySemanticAligner:
    def __init__(self, registry: CapabilityRegistry) -> None:
        self._registry = registry

    def align(self, plan: ResearchPlan) -> ResearchPlan:
        actions = tuple(self._align_action(action) for action in plan.actions)
        return ResearchPlan(
            objective=plan.objective,
            actions=actions,
            assumptions=plan.assumptions,
            plan_id=plan.plan_id,
        )

    def _align_action(self, action: ResearchAction) -> ResearchAction:
        if not self._registry.has(action.action_type):
            return action

        capability = self._registry.get(action.action_type)
        contract = capability.contract
        if contract is None:
            return action
        supported = contract.supported_metrics
        if supported is None:
            return action

        aliases = _ACTION_ALIASES.get(action.action_type, {})
        normalized: list[str] = []
        for raw in action.metrics:
            canonical = contract.metric_ontology.canonicalize(raw)
            mapped = aliases.get(raw, aliases.get(canonical, canonical))
            if mapped in supported and mapped not in normalized:
                normalized.append(mapped)

        if not normalized and action.action_type in _DEFAULTS:
            normalized.extend(
                metric
                for metric in _DEFAULTS[action.action_type]
                if metric in supported
            )

        parameters = dict(action.parameters)
        if action.action_type is ActionType.COMPARE_PEERS:
            rank_by = str(parameters.get("rank_by") or "")
            if rank_by not in {
                "return_1y",
                "return_3y_cagr",
                "return_5y_cagr",
                "return_10y_cagr",
                "rolling_volatility",
            }:
                parameters["rank_by"] = "return_3y_cagr"

        return replace(
            action,
            metrics=tuple(normalized),
            parameters=parameters,
        )
