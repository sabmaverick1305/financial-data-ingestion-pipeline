from financial_pipeline.intelligence.capability_registry import CapabilityRegistry, CapabilityResult
from financial_pipeline.intelligence.evidence import EvidenceEvaluator
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import HarnessLimits, ReasoningHarness
from financial_pipeline.intelligence.mock_planner import MockPlanner
from financial_pipeline.intelligence.reasoning_loop import ReasoningLoop
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.reasoning_trace import ReasoningTraceEventType
from financial_pipeline.intelligence.replan import EvidenceReplanner
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction


def _run(query: str, fail_once: ActionType | None = None) -> ReasoningState:
    plan, requirements = MockPlanner().plan(query)
    registry = CapabilityRegistry()
    failed: set[ActionType] = set()

    def handler(action: ResearchAction):
        if fail_once is action.action_type and action.action_type not in failed:
            failed.add(action.action_type)
            raise RuntimeError("temporary failure")
        return CapabilityResult(
            result={"verified": True},
            evidence_refs=(f"verified:{action.action_type.value}",),
        )

    for action_type in ActionType:
        registry.register(action_type, handler)

    harness = ReasoningHarness(registry, HarnessLimits(max_tool_calls=30))
    return ReasoningLoop(
        ResearchExecutor(registry, harness),
        EvidenceEvaluator(),
        EvidenceReplanner(),
        harness,
    ).run(
        state=ReasoningState(query=query),
        initial_plan=plan,
        requirements=requirements,
    )


def test_trace_records_full_successful_reasoning_path() -> None:
    state = _run("Give me some of the best mutual funds to invest in 2026")
    assert state.trace is not None
    types = [event.event_type for event in state.trace.events]

    assert types[0] is ReasoningTraceEventType.PLAN_STARTED
    assert types.count(ReasoningTraceEventType.ACTION_STARTED) == 9
    assert types.count(ReasoningTraceEventType.ACTION_FINISHED) == 9
    assert ReasoningTraceEventType.EVIDENCE_EVALUATED in types
    assert types[-1] is ReasoningTraceEventType.LOOP_STOPPED
    assert state.trace.events[-1].payload["reason"] == "evidence_sufficient"


def test_trace_exposes_failed_evidence_and_replan() -> None:
    state = _run(
        "Give me some of the best mutual funds to invest in 2026",
        fail_once=ActionType.COMPUTE_RISK,
    )
    assert state.trace is not None

    evaluations = [
        event for event in state.trace.events
        if event.event_type is ReasoningTraceEventType.EVIDENCE_EVALUATED
    ]
    replans = [
        event for event in state.trace.events
        if event.event_type is ReasoningTraceEventType.REPLAN_CREATED
    ]

    assert evaluations[0].payload["is_sufficient"] is False
    assert "risk" in evaluations[0].payload["failed"]
    assert len(replans) == 1
    assert replans[0].payload["actions"] == [
        "compute_risk",
        "compare_peers",
        "fetch_flows",
        "fetch_aum",
        "retrieve_evidence",
        "check_contradictions",
    ]
    assert evaluations[-1].payload["is_sufficient"] is True


def test_trace_can_be_serialized_for_structured_logging() -> None:
    state = _run("Compare the best mid cap funds for 2026")
    assert state.trace is not None
    payload = state.trace.to_log_dict()

    assert payload["query"] == "Compare the best mid cap funds for 2026"
    assert payload["events"]
    assert payload["events"][0]["event_type"] == "plan_started"
