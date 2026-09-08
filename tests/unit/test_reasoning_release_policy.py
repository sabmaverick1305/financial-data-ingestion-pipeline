from datetime import date, timedelta

from financial_pipeline.intelligence.confidence import ConfidenceScorer
from financial_pipeline.intelligence.data_quality import FundDataQualityGate
from financial_pipeline.intelligence.evidence import (
    EvidenceDimension,
    EvidenceEvaluator,
    EvidenceImportance,
    EvidenceRequirement,
)
from financial_pipeline.intelligence.plan_dependencies import PlanDependencyResolver
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import (
    ActionObservation,
    ActionStatus,
    ActionType,
    ResearchAction,
    ResearchPlan,
)

def test_data_quality_rejects_stale_and_implausible_candidate():
    gate = FundDataQualityGate()
    result = gate.validate_candidate({
        "scheme_name": "Essel Liquid Fund Growth",
        "category": "Multi Cap",
        "latest_nav_date": date.today() - timedelta(days=365),
        "return_1y": 10567,
        "return_3y_cagr": 398,
        "rolling_volatility": 4134,
    })
    assert result.valid is False
    codes = {issue.code for issue in result.issues}
    assert "stale_nav" in codes
    assert "implausible_return_1y" in codes
    assert "category_name_mismatch" in codes

def test_dependency_resolver_runs_discovery_before_performance_and_risk():
    plan = ResearchPlan(
        objective="rank",
        actions=(
            ResearchAction(ActionType.FETCH_PERFORMANCE),
            ResearchAction(ActionType.COMPUTE_RISK),
            ResearchAction(ActionType.DISCOVER_FUNDS),
            ResearchAction(ActionType.DISCOVER_CATEGORIES),
        ),
    )
    ordered = PlanDependencyResolver().order(plan)
    types = [action.action_type for action in ordered.actions]
    assert types.index(ActionType.DISCOVER_CATEGORIES) < types.index(ActionType.DISCOVER_FUNDS)
    assert types.index(ActionType.DISCOVER_FUNDS) < types.index(ActionType.FETCH_PERFORMANCE)
    assert types.index(ActionType.FETCH_PERFORMANCE) < types.index(ActionType.COMPUTE_RISK)

def test_soft_documentary_failure_does_not_block_hard_evidence():
    state = ReasoningState(query="q")
    state.add_observation(ActionObservation(
        "p", ActionType.FETCH_PERFORMANCE, ActionStatus.SUCCEEDED
    ))
    state.add_observation(ActionObservation(
        "d", ActionType.RETRIEVE_EVIDENCE, ActionStatus.PARTIAL,
        tradeoff_reason="documentary unavailable",
    ))
    evaluation = EvidenceEvaluator().evaluate(
        state,
        (
            EvidenceRequirement(EvidenceDimension.PERFORMANCE),
            EvidenceRequirement(
                EvidenceDimension.DOCUMENTARY,
                allow_partial=True,
                importance=EvidenceImportance.SOFT,
            ),
        ),
    )
    assert evaluation.is_sufficient is True
    assert evaluation.soft_failed == (EvidenceDimension.DOCUMENTARY,)

def test_soft_evidence_gap_reduces_confidence():
    state = ReasoningState(query="q")
    for index, action_type in enumerate((
        ActionType.DISCOVER_CATEGORIES,
        ActionType.DISCOVER_FUNDS,
        ActionType.FETCH_PERFORMANCE,
        ActionType.COMPUTE_RISK,
        ActionType.COMPARE_PEERS,
        ActionType.CHECK_CONTRADICTIONS,
    )):
        state.add_observation(ActionObservation(
            str(index), action_type, ActionStatus.SUCCEEDED,
            evidence_refs=("verified:test",),
        ))
    base = ConfidenceScorer().score(state)
    state.soft_evidence_gaps = ["documentary", "flows"]
    degraded = ConfidenceScorer().score(state)
    assert degraded.score < base.score
    assert degraded.soft_evidence_penalty == 0.16
