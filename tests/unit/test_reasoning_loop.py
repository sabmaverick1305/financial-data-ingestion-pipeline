from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.evidence import (
    EvidenceDimension,
    EvidenceEvaluator,
    EvidenceRequirement,
)
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import HarnessLimits, ReasoningHarness
from financial_pipeline.intelligence.reasoning_loop import ReasoningLoop
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.replan import EvidenceReplanner
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan


def _requirements() -> tuple[EvidenceRequirement, ...]:
    return (
        EvidenceRequirement(EvidenceDimension.PERFORMANCE),
        EvidenceRequirement(EvidenceDimension.RISK),
        EvidenceRequirement(EvidenceDimension.PEER_COMPARISON),
        EvidenceRequirement(EvidenceDimension.CONTRADICTION),
    )


def test_evaluator_reports_missing_evidence() -> None:
    registry = CapabilityRegistry()
    registry.register(ActionType.FETCH_PERFORMANCE, lambda _: {"return_5y": 18.0})

    harness = ReasoningHarness(registry)
    executor = ResearchExecutor(registry, harness)
    state = ReasoningState(query="best mutual funds")
    executor.execute(
        state,
        ResearchPlan(
            objective="shortlist funds",
            actions=(ResearchAction(ActionType.FETCH_PERFORMANCE),),
        ),
    )

    evaluation = EvidenceEvaluator().evaluate(state, _requirements())

    assert not evaluation.is_sufficient
    assert evaluation.satisfied == (EvidenceDimension.PERFORMANCE,)
    assert EvidenceDimension.RISK in evaluation.missing
    assert EvidenceDimension.CONTRADICTION in evaluation.missing


def test_reasoning_loop_fills_missing_evidence_without_human_approval() -> None:
    registry = CapabilityRegistry()
    calls: list[ActionType] = []

    def handler(action: ResearchAction) -> dict[str, str]:
        calls.append(action.action_type)
        return {"source": "verified"}

    for action_type in (
        ActionType.FETCH_PERFORMANCE,
        ActionType.COMPUTE_RISK,
        ActionType.COMPARE_PEERS,
        ActionType.CHECK_CONTRADICTIONS,
    ):
        registry.register(action_type, handler)

    harness = ReasoningHarness(
        registry,
        HarnessLimits(max_investigation_rounds=3, max_replans=2),
    )
    executor = ResearchExecutor(registry, harness)
    loop = ReasoningLoop(executor, EvidenceEvaluator(), EvidenceReplanner(), harness)

    state = loop.run(
        state=ReasoningState(query="best mutual funds"),
        initial_plan=ResearchPlan(
            objective="shortlist funds",
            actions=(ResearchAction(ActionType.FETCH_PERFORMANCE),),
        ),
        requirements=_requirements(),
    )

    assert state.abstention_reason is None
    assert state.replan_count == 1
    assert calls == [
        ActionType.FETCH_PERFORMANCE,
        ActionType.COMPUTE_RISK,
        ActionType.COMPARE_PEERS,
        ActionType.CHECK_CONTRADICTIONS,
    ]


def test_reasoning_loop_stops_when_replan_budget_is_exhausted() -> None:
    registry = CapabilityRegistry()
    registry.register(ActionType.FETCH_PERFORMANCE, lambda _: {"ok": True})
    registry.register(ActionType.COMPUTE_RISK, lambda _: (_ for _ in ()).throw(RuntimeError("missing data")))
    registry.register(ActionType.COMPARE_PEERS, lambda _: {"ok": True})
    registry.register(ActionType.CHECK_CONTRADICTIONS, lambda _: {"ok": True})

    harness = ReasoningHarness(
        registry,
        HarnessLimits(max_investigation_rounds=3, max_replans=1),
    )
    executor = ResearchExecutor(registry, harness)
    loop = ReasoningLoop(executor, EvidenceEvaluator(), EvidenceReplanner(), harness)

    state = loop.run(
        state=ReasoningState(query="best mutual funds"),
        initial_plan=ResearchPlan(
            objective="shortlist funds",
            actions=(ResearchAction(ActionType.FETCH_PERFORMANCE),),
        ),
        requirements=_requirements(),
    )

    assert state.replan_count == 1
    assert state.abstention_reason == "replan budget exhausted before evidence became sufficient"
