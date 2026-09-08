from financial_pipeline.intelligence.observability import ReasoningObservability
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import (
    ActionObservation,
    ActionStatus,
    ActionType,
)

def test_observability_reports_documentary_gap_and_candidate_quality():
    state = ReasoningState(query="q")
    state.investigation_round = 1
    state.replan_count = 0
    state.tool_calls = 2
    state.llm_calls = 1
    state.confidence_score = 0.88
    state.final_answer = "ranked"
    state.soft_evidence_gaps = ["documentary"]
    state.candidate_decisions = {
        "1": {"eligible_for_ranking": True},
        "2": {"eligible_for_ranking": False},
    }
    state.add_observation(ActionObservation(
        "doc",
        ActionType.RETRIEVE_EVIDENCE,
        ActionStatus.PARTIAL,
        result={
            "documentary_coverage_ratio": 0.25,
            "ingestion_backlog": [
                {"fund_name": "A", "document_type": "fund_fact_sheet"},
                {"fund_name": "A", "document_type": "fund_prospectus"},
            ],
        },
    ))

    snapshot = ReasoningObservability().snapshot(state)
    assert snapshot.first_pass_success is True
    assert snapshot.documentary_coverage_ratio == 0.25
    assert snapshot.documentary_backlog_count == 2
    assert snapshot.candidate_quality_ratio == 0.5
    assert snapshot.partial_actions == ("retrieve_evidence",)
