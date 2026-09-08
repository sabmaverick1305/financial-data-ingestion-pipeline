from financial_pipeline.intelligence.action_dependencies import ActionDependencyGraph
from financial_pipeline.intelligence.confidence import ConfidenceScorer
from financial_pipeline.intelligence.contradictions import ContradictionEngine
from financial_pipeline.intelligence.metric_routing import MetricOwnershipRouter
from financial_pipeline.intelligence.metric_semantics import MetricSemanticsRegistry
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan


def test_metric_ownership_routes_sharpe_out_of_performance() -> None:
    plan = ResearchPlan(
        objective="rank funds",
        actions=(
            ResearchAction(
                ActionType.FETCH_PERFORMANCE,
                metrics=("3-year return", "sharpe ratio"),
            ),
        ),
    )
    routed = MetricOwnershipRouter().route(plan)
    by_type = {action.action_type: action for action in routed.actions}
    assert by_type[ActionType.FETCH_PERFORMANCE].metrics == ("return_3y_cagr",)
    assert by_type[ActionType.COMPUTE_RISK].metrics == ("sharpe_ratio",)


def test_derived_metric_dependencies_are_explicit() -> None:
    semantics = MetricSemanticsRegistry()
    assert semantics.dependencies("aum_growth_rate") == ("aum_history",)
    assert semantics.dependencies("sharpe_ratio") == ("return_series", "risk_free_rate")
    assert semantics.owner("asset_size_top_quartile") is ActionType.COMPARE_PEERS


def test_action_dependency_graph_allows_parallel_fetch_layer() -> None:
    graph = ActionDependencyGraph()
    layers = graph.layers((
        ActionType.DISCOVER_CATEGORIES, ActionType.DISCOVER_FUNDS,
        ActionType.FETCH_PERFORMANCE, ActionType.FETCH_FLOWS, ActionType.FETCH_AUM,
    ))
    assert layers[0] == (ActionType.DISCOVER_CATEGORIES,)
    assert layers[1] == (ActionType.DISCOVER_FUNDS,)
    assert set(layers[2]) == {ActionType.FETCH_PERFORMANCE, ActionType.FETCH_FLOWS, ActionType.FETCH_AUM}


def test_contradiction_rules_are_deterministic() -> None:
    contradictions = ContradictionEngine().evaluate({
        "return_3y_cagr": 20, "net_inflow": -10,
        "return_5y_cagr": 18, "max_drawdown": 35,
    })
    assert {item.code for item in contradictions} == {
        "strong_return_persistent_outflow", "strong_cagr_extreme_drawdown"
    }


def test_confidence_penalizes_replans() -> None:
    state = ReasoningState(query="q")
    state.replan_count = 2
    score = ConfidenceScorer().score(state)
    assert score.replan_penalty == 0.1
    assert 0 <= score.score <= 1
