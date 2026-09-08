"""Verified mock capability pack for pre-LLM end-to-end reasoning tests."""

from __future__ import annotations

from financial_pipeline.intelligence.capability_contracts import CapabilityContract
from financial_pipeline.intelligence.capability_registry import CapabilityRegistry, CapabilityResult
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType, ResearchAction


class MockVerifiedCapabilityPack:
    def __init__(self, *, partial_risk: bool = False) -> None:
        self.partial_risk = partial_risk
        self.calls: list[ActionType] = []

    def register_all(self, registry: CapabilityRegistry) -> None:
        contracts = {
            ActionType.DISCOVER_FUNDS: CapabilityContract(
                supported_metrics=("aum", "inception_date", "expense_ratio"),
            ),
            ActionType.FETCH_PERFORMANCE: CapabilityContract(
                supported_metrics=("return_1y", "return_3y_cagr", "return_5y_cagr", "return_10y_cagr"),
                metric_aliases={
                    "1yr_return": "return_1y",
                    "3yr_return": "return_3y_cagr",
                    "5yr_return": "return_5y_cagr",
                    "10yr_return": "return_10y_cagr",
                },
            ),
            ActionType.COMPUTE_RISK: CapabilityContract(
                supported_metrics=("volatility", "sharpe_ratio", "max_drawdown"),
                metric_aliases={"rolling_volatility": "volatility", "rolling_stddev": "volatility"},
            ),
            ActionType.COMPARE_PEERS: CapabilityContract(
                supported_metrics=("percentile_rank", "peer_outperformance"),
            ),
            ActionType.FETCH_FLOWS: CapabilityContract(
                supported_metrics=("net_inflow", "flow_trend"),
                metric_aliases={"net_flows": "net_inflow"},
            ),
            ActionType.FETCH_AUM: CapabilityContract(
                supported_metrics=("aum", "aum_trend"),
                metric_aliases={"total_aum": "aum"},
            ),
            ActionType.RETRIEVE_EVIDENCE: CapabilityContract(
                supported_metrics=("fund_manager_tenure", "expense_ratio", "portfolio_concentration"),
            ),
        }
        for action_type in ActionType:
            registry.register(
                action_type,
                self._handle,
                trusted=True,
                contract=contracts.get(action_type),
            )

    def _handle(self, action: ResearchAction) -> CapabilityResult:
        self.calls.append(action.action_type)

        if action.action_type is ActionType.DISCOVER_CATEGORIES:
            result = {"categories": ["Large Cap Fund", "Mid Cap Fund", "Hybrid Fund"]}
        elif action.action_type is ActionType.DISCOVER_FUNDS:
            result = {"funds": ["Fund A", "Fund B", "Fund C"]}
        elif action.action_type is ActionType.FETCH_PERFORMANCE:
            result = {
                "Fund A": {"return_1y": 14.2, "return_3y_cagr": 18.4, "return_5y_cagr": 20.1},
                "Fund B": {"return_1y": 17.8, "return_3y_cagr": 16.2, "return_5y_cagr": 17.4},
            }
        elif action.action_type is ActionType.COMPUTE_RISK and self.partial_risk:
            return CapabilityResult(
                result={"Fund A": {"drawdown": 12.5, "volatility": None}},
                evidence_refs=("verified:risk",),
                status=ActionStatus.PARTIAL,
                tradeoff_reason="volatility unavailable; drawdown evidence retained",
            )
        elif action.action_type is ActionType.COMPUTE_RISK:
            result = {"Fund A": {"drawdown": 12.5, "volatility": 16.2}}
        elif action.action_type is ActionType.COMPARE_PEERS:
            result = {"ranked": ["Fund A", "Fund B"], "basis": "peer-adjusted"}
        elif action.action_type is ActionType.FETCH_FLOWS:
            result = {"category_net_inflow": 5100}
        elif action.action_type is ActionType.FETCH_AUM:
            result = {"category_aum": 345678.9}
        elif action.action_type is ActionType.RETRIEVE_EVIDENCE:
            result = {"sources": ["AMFI", "SEBI"]}
        elif action.action_type is ActionType.CHECK_CONTRADICTIONS:
            result = {"contradictions": []}
        else:
            result = {"ok": True}

        return CapabilityResult(
            result=result,
            evidence_refs=(f"verified:{action.action_type.value}",),
        )
