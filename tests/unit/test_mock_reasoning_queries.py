from financial_pipeline.intelligence.capability_registry import CapabilityRegistry, CapabilityResult
from financial_pipeline.intelligence.evidence import EvidenceDimension, EvidenceEvaluator
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import HarnessLimits, ReasoningHarness
from financial_pipeline.intelligence.mock_planner import MockPlanner
from financial_pipeline.intelligence.reasoning_loop import ReasoningLoop
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.replan import EvidenceReplanner
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType, ResearchAction


def _verified_registry(*, fail_once: ActionType | None = None) -> tuple[CapabilityRegistry, list[ActionType]]:
    registry = CapabilityRegistry()
    calls: list[ActionType] = []
    failed: set[ActionType] = set()

    def handler(action: ResearchAction):
        calls.append(action.action_type)
        if fail_once is action.action_type and action.action_type not in failed:
            failed.add(action.action_type)
            raise RuntimeError("temporary verified-source failure")
        return CapabilityResult(
            result={"action": action.action_type.value, "verified": True},
            evidence_refs=(f"verified:{action.action_type.value}",),
        )

    for action_type in ActionType:
        registry.register(action_type, handler, trusted=True)

    return registry, calls


def _run_query(query: str, *, fail_once: ActionType | None = None):
    planner = MockPlanner()
    plan, requirements = planner.plan(query)
    registry, calls = _verified_registry(fail_once=fail_once)
    harness = ReasoningHarness(
        registry,
        HarnessLimits(max_investigation_rounds=3, max_replans=2, max_tool_calls=30),
    )
    loop = ReasoningLoop(
        ResearchExecutor(registry, harness),
        EvidenceEvaluator(),
        EvidenceReplanner(),
        harness,
    )
    state = loop.run(
        state=ReasoningState(query=query),
        initial_plan=plan,
        requirements=requirements,
    )
    return plan, requirements, state, calls


def test_best_funds_query_selects_full_research_workflow() -> None:
    plan, requirements, state, calls = _run_query("Give me some of the best mutual funds to invest in 2026")

    assert [a.action_type for a in plan.actions] == [
        ActionType.DISCOVER_CATEGORIES,
        ActionType.DISCOVER_FUNDS,
        ActionType.FETCH_PERFORMANCE,
        ActionType.COMPUTE_RISK,
        ActionType.COMPARE_PEERS,
        ActionType.FETCH_FLOWS,
        ActionType.FETCH_AUM,
        ActionType.RETRIEVE_EVIDENCE,
        ActionType.CHECK_CONTRADICTIONS,
    ]
    assert {r.dimension for r in requirements} == set(EvidenceDimension)
    assert state.abstention_reason is None
    assert state.replan_count == 0
    assert calls == [a.action_type for a in plan.actions]
    assert all(o.status is ActionStatus.SUCCEEDED for o in state.observations)
    assert all(o.evidence_refs for o in state.observations)


def test_mid_cap_query_stays_within_peer_group() -> None:
    plan, _, state, _ = _run_query("Compare the best mid cap funds for 2026")

    assert all(
        action.category == "Mid Cap Fund"
        for action in plan.actions
    )
    assert ActionType.DISCOVER_CATEGORIES not in [a.action_type for a in plan.actions]
    assert state.abstention_reason is None


def test_flow_why_query_uses_flow_and_documentary_evidence() -> None:
    plan, requirements, state, _ = _run_query("Why did mid cap fund inflows fall?")

    assert [a.action_type for a in plan.actions] == [
        ActionType.FETCH_FLOWS,
        ActionType.FETCH_AUM,
        ActionType.RETRIEVE_EVIDENCE,
        ActionType.CHECK_CONTRADICTIONS,
    ]
    assert {r.dimension for r in requirements} == {
        EvidenceDimension.FLOWS,
        EvidenceDimension.AUM,
        EvidenceDimension.DOCUMENTARY,
        EvidenceDimension.CONTRADICTION,
    }
    assert state.abstention_reason is None


def test_reasoning_replans_after_temporary_missing_risk_evidence() -> None:
    _, _, state, calls = _run_query(
        "Give me some of the best mutual funds to invest in 2026",
        fail_once=ActionType.COMPUTE_RISK,
    )

    assert state.replan_count == 1
    assert calls.count(ActionType.COMPUTE_RISK) == 2
    assert state.abstention_reason is None


def test_mock_planner_rejects_unsupported_query() -> None:
    planner = MockPlanner()
    try:
        planner.plan("Tell me a joke")
    except ValueError as exc:
        assert "does not support" in str(exc)
    else:
        raise AssertionError("expected ValueError")
