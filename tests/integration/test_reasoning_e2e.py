from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.evidence import EvidenceEvaluator
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import HarnessLimits, ReasoningHarness
from financial_pipeline.intelligence.mock_capabilities import MockVerifiedCapabilityPack
from financial_pipeline.intelligence.mock_planner import MockPlanner
from financial_pipeline.intelligence.reasoning_graph import GraphNode, ReasoningGraph
from financial_pipeline.intelligence.reasoning_trace import ReasoningTraceEventType
from financial_pipeline.intelligence.replan import EvidenceReplanner


def _graph(*, partial_risk: bool = False):
    registry = CapabilityRegistry()
    capabilities = MockVerifiedCapabilityPack(partial_risk=partial_risk)
    capabilities.register_all(registry)
    harness = ReasoningHarness(
        registry,
        HarnessLimits(
            max_investigation_rounds=3,
            max_tool_calls=30,
            max_llm_calls=0,
            max_replans=2,
        ),
    )
    graph = ReasoningGraph(
        planner=MockPlanner(),
        executor=ResearchExecutor(registry, harness),
        evaluator=EvidenceEvaluator(),
        replanner=EvidenceReplanner(),
        harness=harness,
    )
    return graph, capabilities


def test_best_mutual_funds_e2e_graph_loop_harness() -> None:
    graph, capabilities = _graph()

    result = graph.run("Give me some of the best mutual funds to invest in 2026")
    state = result.state

    assert result.visited_nodes == (
        GraphNode.PLAN,
        GraphNode.EXECUTE_AND_EVALUATE,
        GraphNode.FINALIZE,
    )
    assert state.abstention_reason is None
    assert state.investigation_round == 1
    assert state.replan_count == 0
    assert state.tool_calls == 9
    assert state.llm_calls == 0
    assert len(state.observations) == 9
    assert len(capabilities.calls) == 9

    assert state.trace is not None
    event_types = [event.event_type for event in state.trace.events]
    assert event_types[0] is ReasoningTraceEventType.PLAN_STARTED
    assert event_types[-1] is ReasoningTraceEventType.LOOP_STOPPED
    assert state.trace.events[-1].payload["reason"] == "evidence_sufficient"


def test_e2e_partial_risk_is_replanned_under_strict_evidence_policy() -> None:
    graph, capabilities = _graph(partial_risk=True)

    result = graph.run("Give me some of the best mutual funds to invest in 2026")
    state = result.state

    assert state.replan_count == 2
    assert state.abstention_reason == "replan budget exhausted before evidence became sufficient"
    assert result.visited_nodes[-1] is GraphNode.ABSTAIN
    assert capabilities.calls.count(capabilities.calls[3]) >= 1

    assert state.trace is not None
    evaluations = [
        e for e in state.trace.events
        if e.event_type is ReasoningTraceEventType.EVIDENCE_EVALUATED
    ]
    assert evaluations[0].payload["partial"] == ["risk"]
    assert evaluations[0].payload["tradeoffs"] == [
        "volatility unavailable; drawdown evidence retained"
    ]
