from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import HarnessLimits, ReasoningHarness
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import (
    ActionStatus,
    ActionType,
    ResearchAction,
    ResearchPlan,
)


def test_executor_runs_ordered_trusted_actions() -> None:
    registry = CapabilityRegistry()
    calls: list[ActionType] = []

    def handler(action: ResearchAction) -> dict[str, str]:
        calls.append(action.action_type)
        return {"ok": action.action_type.value}

    registry.register(ActionType.DISCOVER_CATEGORIES, handler)
    registry.register(ActionType.DISCOVER_FUNDS, handler)

    plan = ResearchPlan(
        objective="shortlist funds",
        actions=(
            ResearchAction(ActionType.DISCOVER_CATEGORIES),
            ResearchAction(ActionType.DISCOVER_FUNDS),
        ),
    )
    state = ReasoningState(query="best mutual funds")
    harness = ReasoningHarness(registry)
    result = ResearchExecutor(registry, harness).execute(state, plan)

    assert calls == [ActionType.DISCOVER_CATEGORIES, ActionType.DISCOVER_FUNDS]
    assert result.is_complete
    assert result.tool_calls == 2
    assert all(o.status is ActionStatus.SUCCEEDED for o in result.observations)


def test_registry_captures_capability_failure_as_observation() -> None:
    registry = CapabilityRegistry()

    def broken(_: ResearchAction) -> None:
        raise RuntimeError("source unavailable")

    registry.register(ActionType.FETCH_FLOWS, broken)
    action = ResearchAction(ActionType.FETCH_FLOWS)

    observation = registry.execute(action)

    assert observation.status is ActionStatus.FAILED
    assert observation.error == "source unavailable"


def test_harness_blocks_untrusted_capability() -> None:
    registry = CapabilityRegistry()
    registry.register(ActionType.RETRIEVE_EVIDENCE, lambda _: [], trusted=False)
    harness = ReasoningHarness(registry)
    state = ReasoningState(query="test")

    try:
        harness.validate_action(state, ResearchAction(ActionType.RETRIEVE_EVIDENCE))
    except PermissionError as exc:
        assert "untrusted capability" in str(exc)
    else:
        raise AssertionError("expected PermissionError")


def test_harness_enforces_tool_budget() -> None:
    registry = CapabilityRegistry()
    registry.register(ActionType.DISCOVER_FUNDS, lambda _: [])
    harness = ReasoningHarness(registry, HarnessLimits(max_tool_calls=1))
    state = ReasoningState(query="test", tool_calls=1)

    try:
        harness.validate_action(state, ResearchAction(ActionType.DISCOVER_FUNDS))
    except RuntimeError as exc:
        assert "budget exhausted" in str(exc)
    else:
        raise AssertionError("expected RuntimeError")
