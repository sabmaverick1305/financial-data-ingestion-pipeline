from financial_pipeline.intelligence.capability_registry import CapabilityRegistry, CapabilityResult
from financial_pipeline.intelligence.evidence import EvidenceDimension, EvidenceEvaluator, EvidenceRequirement
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import ReasoningHarness
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType, ResearchAction, ResearchPlan


def test_executor_continues_after_failed_action() -> None:
    registry = CapabilityRegistry()
    calls: list[ActionType] = []

    def handler(action: ResearchAction):
        calls.append(action.action_type)
        if action.action_type is ActionType.COMPUTE_RISK:
            raise RuntimeError("risk history unavailable")
        return {"ok": True}

    for action_type in (ActionType.FETCH_PERFORMANCE, ActionType.COMPUTE_RISK, ActionType.COMPARE_PEERS):
        registry.register(action_type, handler)

    harness = ReasoningHarness(registry)
    state = ResearchExecutor(registry, harness).execute(
        ReasoningState(query="compare funds"),
        ResearchPlan(
            objective="compare funds",
            actions=(
                ResearchAction(ActionType.FETCH_PERFORMANCE),
                ResearchAction(ActionType.COMPUTE_RISK),
                ResearchAction(ActionType.COMPARE_PEERS),
            ),
        ),
    )

    assert calls == [ActionType.FETCH_PERFORMANCE, ActionType.COMPUTE_RISK, ActionType.COMPARE_PEERS]
    assert [o.status for o in state.observations] == [
        ActionStatus.SUCCEEDED, ActionStatus.FAILED, ActionStatus.SUCCEEDED
    ]


def test_partial_evidence_can_be_accepted_with_tradeoff() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ActionType.COMPUTE_RISK,
        lambda _: CapabilityResult(
            result={"volatility": None, "drawdown": 12.5},
            status=ActionStatus.PARTIAL,
            tradeoff_reason="volatility unavailable because monthly history is incomplete; drawdown retained",
        ),
    )
    harness = ReasoningHarness(registry)
    state = ResearchExecutor(registry, harness).execute(
        ReasoningState(query="compare funds"),
        ResearchPlan(objective="compare funds", actions=(ResearchAction(ActionType.COMPUTE_RISK),)),
    )

    evaluation = EvidenceEvaluator().evaluate(
        state,
        (EvidenceRequirement(EvidenceDimension.RISK, allow_partial=True),),
    )

    assert evaluation.is_sufficient
    assert evaluation.partial == (EvidenceDimension.RISK,)
    assert evaluation.failed == ()
    assert evaluation.tradeoffs == (
        "volatility unavailable because monthly history is incomplete; drawdown retained",
    )


def test_partial_evidence_replans_when_requirement_is_strict() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ActionType.COMPUTE_RISK,
        lambda _: CapabilityResult(
            result={"drawdown": 12.5},
            status=ActionStatus.PARTIAL,
            tradeoff_reason="volatility unavailable",
        ),
    )
    harness = ReasoningHarness(registry)
    state = ResearchExecutor(registry, harness).execute(
        ReasoningState(query="compare funds"),
        ResearchPlan(objective="compare funds", actions=(ResearchAction(ActionType.COMPUTE_RISK),)),
    )

    evaluation = EvidenceEvaluator().evaluate(
        state,
        (EvidenceRequirement(EvidenceDimension.RISK, allow_partial=False),),
    )

    assert not evaluation.is_sufficient
    assert evaluation.partial == (EvidenceDimension.RISK,)
    assert evaluation.failed == (EvidenceDimension.RISK,)
