from financial_pipeline.intelligence.capability_alignment import CapabilitySemanticAligner
from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.capability_contracts import CapabilityContract
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan

def _registry() -> CapabilityRegistry:
    registry = CapabilityRegistry()
    noop = lambda action: None
    registry.register(ActionType.FETCH_PERFORMANCE, noop, contract=CapabilityContract(
        supported_metrics=("return_1y","return_3y_cagr","return_5y_cagr","return_10y_cagr")
    ))
    registry.register(ActionType.COMPUTE_RISK, noop, contract=CapabilityContract(
        supported_metrics=("volatility","sharpe_ratio","max_drawdown")
    ))
    registry.register(ActionType.COMPARE_PEERS, noop, contract=CapabilityContract(
        supported_metrics=("percentile_rank","peer_outperformance")
    ))
    registry.register(ActionType.FETCH_AUM, noop, contract=CapabilityContract(
        supported_metrics=("aum","aum_trend","aum_growth_rate")
    ))
    registry.register(ActionType.FETCH_FLOWS, noop, contract=CapabilityContract(
        supported_metrics=("net_inflow","flow_trend","redemption_rate","relative_flows")
    ))
    registry.register(ActionType.COMPUTE_RETURNS, noop, contract=CapabilityContract(
        supported_metrics=("total_return_3y","annualized_return")
    ))
    return registry

def test_alignment_maps_first_pass_production_vocabulary():
    plan = ResearchPlan(
        objective="rank",
        actions=(
            ResearchAction(ActionType.FETCH_PERFORMANCE, metrics=("ytd_return","return_3y_cagr")),
            ResearchAction(ActionType.COMPUTE_RISK, metrics=("beta","sortino_ratio","max_drawdown")),
            ResearchAction(ActionType.COMPARE_PEERS, metrics=("percentile_ranking_by_return","consistency_vs_peers"), parameters={"rank_by":"risk_adjusted_return"}),
            ResearchAction(ActionType.FETCH_AUM, metrics=("total_assets_under_management",)),
            ResearchAction(ActionType.FETCH_FLOWS, metrics=("flow_rate","flow_direction")),
            ResearchAction(ActionType.COMPUTE_RETURNS, metrics=("excess_return_vs_benchmark","information_ratio")),
        ),
    )
    aligned = CapabilitySemanticAligner(_registry()).align(plan)
    by_type = {a.action_type: a for a in aligned.actions}
    assert by_type[ActionType.FETCH_PERFORMANCE].metrics == ("return_1y","return_3y_cagr")
    assert by_type[ActionType.COMPUTE_RISK].metrics == ("volatility","sharpe_ratio","max_drawdown")
    assert by_type[ActionType.COMPARE_PEERS].metrics == ("percentile_rank","peer_outperformance")
    assert by_type[ActionType.COMPARE_PEERS].parameters["rank_by"] == "return_3y_cagr"
    assert by_type[ActionType.FETCH_AUM].metrics == ("aum",)
    assert by_type[ActionType.FETCH_FLOWS].metrics == ("flow_trend",)
    assert by_type[ActionType.COMPUTE_RETURNS].metrics == ("annualized_return",)

def test_alignment_defaults_when_all_planner_metrics_are_unsupported():
    plan = ResearchPlan(
        objective="risk",
        actions=(ResearchAction(ActionType.COMPUTE_RISK, metrics=("unknown_metric",)),),
    )
    action = CapabilitySemanticAligner(_registry()).align(plan).actions[0]
    assert action.metrics == ("volatility","sharpe_ratio","max_drawdown")
