"""Canonical metric ownership, derived semantics, and dependency metadata."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from financial_pipeline.intelligence.research_plan import ActionType

class MetricKind(StrEnum):
    RAW = "raw"
    DERIVED = "derived"
    COMPARATIVE = "comparative"

@dataclass(frozen=True)
class MetricDefinition:
    name: str
    owner: ActionType
    kind: MetricKind
    dependencies: tuple[str, ...] = ()
    description: str = ""

_DEFINITIONS = {
    "return_1y": MetricDefinition("return_1y", ActionType.FETCH_PERFORMANCE, MetricKind.RAW),
    "return_3y_cagr": MetricDefinition("return_3y_cagr", ActionType.FETCH_PERFORMANCE, MetricKind.RAW),
    "return_5y_cagr": MetricDefinition("return_5y_cagr", ActionType.FETCH_PERFORMANCE, MetricKind.RAW),
    "return_10y_cagr": MetricDefinition("return_10y_cagr", ActionType.FETCH_PERFORMANCE, MetricKind.RAW),
    "volatility": MetricDefinition("volatility", ActionType.COMPUTE_RISK, MetricKind.DERIVED, ("nav_history",)),
    "sharpe_ratio": MetricDefinition("sharpe_ratio", ActionType.COMPUTE_RISK, MetricKind.DERIVED, ("return_series", "risk_free_rate")),
    "max_drawdown": MetricDefinition("max_drawdown", ActionType.COMPUTE_RISK, MetricKind.DERIVED, ("nav_history",)),
    "downside_capture": MetricDefinition("downside_capture", ActionType.COMPUTE_RISK, MetricKind.DERIVED, ("fund_returns", "benchmark_returns")),
    "beta": MetricDefinition("beta", ActionType.COMPUTE_RISK, MetricKind.DERIVED, ("fund_returns", "benchmark_returns")),
    "aum": MetricDefinition("aum", ActionType.FETCH_AUM, MetricKind.RAW),
    "aum_trend": MetricDefinition("aum_trend", ActionType.FETCH_AUM, MetricKind.DERIVED, ("aum_history",)),
    "aum_growth_rate": MetricDefinition("aum_growth_rate", ActionType.FETCH_AUM, MetricKind.DERIVED, ("aum_history",)),
    "net_inflow": MetricDefinition("net_inflow", ActionType.FETCH_FLOWS, MetricKind.RAW),
    "flow_trend": MetricDefinition("flow_trend", ActionType.FETCH_FLOWS, MetricKind.DERIVED, ("flow_history",)),
    "redemption_rate": MetricDefinition("redemption_rate", ActionType.FETCH_FLOWS, MetricKind.DERIVED, ("redemptions", "opening_aum")),
    "relative_flows": MetricDefinition("relative_flows", ActionType.FETCH_FLOWS, MetricKind.COMPARATIVE, ("net_inflow", "peer_net_inflow")),
    "percentile_rank": MetricDefinition("percentile_rank", ActionType.COMPARE_PEERS, MetricKind.COMPARATIVE, ("peer_set", "metric_values")),
    "peer_outperformance": MetricDefinition("peer_outperformance", ActionType.COMPARE_PEERS, MetricKind.COMPARATIVE, ("peer_set", "return_metric")),
    "expense_ratio_below_median": MetricDefinition("expense_ratio_below_median", ActionType.COMPARE_PEERS, MetricKind.COMPARATIVE, ("expense_ratio", "peer_expense_ratios")),
    "asset_size_top_quartile": MetricDefinition("asset_size_top_quartile", ActionType.COMPARE_PEERS, MetricKind.COMPARATIVE, ("aum", "peer_aum")),
}

class MetricSemanticsRegistry:
    def definition(self, metric: str) -> MetricDefinition | None:
        return _DEFINITIONS.get(metric)
    def owner(self, metric: str) -> ActionType | None:
        definition = self.definition(metric)
        return definition.owner if definition else None
    def dependencies(self, metric: str) -> tuple[str, ...]:
        definition = self.definition(metric)
        return definition.dependencies if definition else ()
