from financial_pipeline.intelligence.confidence import ConfidenceScorer
from financial_pipeline.intelligence.decision_pipeline import CandidateDecisionPipeline
from financial_pipeline.intelligence.fund_ranking import FundRanker
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import (
    ActionObservation,
    ActionStatus,
    ActionType,
)

def _state() -> ReasoningState:
    state = ReasoningState(query="best mutual funds")
    state.eligible_categories = ["Flexi Cap"]
    state.add_observation(ActionObservation(
        "d", ActionType.DISCOVER_FUNDS, ActionStatus.SUCCEEDED,
        result={"funds": [
            {
                "scheme_code": "1",
                "scheme_name": "Eligible Fund",
                "category": "Flexi Cap",
                "data_quality": {"valid": True, "issues": []},
            },
            {
                "scheme_code": "2",
                "scheme_name": "Wrong Category",
                "category": "Sectoral/Thematic",
                "data_quality": {"valid": True, "issues": []},
            },
        ]},
    ))
    state.add_observation(ActionObservation(
        "p", ActionType.FETCH_PERFORMANCE, ActionStatus.SUCCEEDED,
        result={"scope": "scheme_batch", "rows": [
            {"scheme_code": "1", "metrics": {"return_1y": 15, "return_3y_cagr": 18}},
            {"scheme_code": "2", "metrics": {"return_1y": 30, "return_3y_cagr": 35}},
        ]},
        evidence_refs=("verified:performance",),
    ))
    state.add_observation(ActionObservation(
        "r", ActionType.COMPUTE_RISK, ActionStatus.SUCCEEDED,
        result={"scope": "scheme_batch", "rows": [
            {"scheme_code": "1", "metrics": {"sharpe_ratio": 1.1, "volatility": 12, "max_drawdown": 10}},
            {"scheme_code": "2", "metrics": {"sharpe_ratio": 1.5, "volatility": 18, "max_drawdown": 20}},
        ]},
        evidence_refs=("verified:risk",),
    ))
    state.add_observation(ActionObservation(
        "peer", ActionType.COMPARE_PEERS, ActionStatus.SUCCEEDED,
        result={"scope": "scheme_peer_sets", "categories": [
            {"category": "Flexi Cap", "peers": [
                {"scheme_code": "1", "percentile_rank": 90},
            ]},
            {"category": "Sectoral/Thematic", "peers": [
                {"scheme_code": "2", "percentile_rank": 99},
            ]},
        ]},
        evidence_refs=("verified:peers",),
    ))
    return state

def test_decision_pipeline_rejects_candidate_outside_mandate():
    state = _state()
    decisions = CandidateDecisionPipeline().evaluate(state)
    assert decisions["1"].eligible_for_ranking is True
    assert decisions["2"].eligible_for_ranking is False
    assert decisions["2"].gates["eligibility"].status.value == "fail"

def test_ranker_cannot_rank_rejected_candidate_even_with_better_returns():
    state = _state()
    decisions = CandidateDecisionPipeline().evaluate(state)
    allowed = {
        code for code, decision in decisions.items()
        if decision.eligible_for_ranking
    }
    ranked = FundRanker().rank(state, allowed_codes=allowed)
    assert [fund.scheme_code for fund in ranked] == ["1"]

def test_confidence_uses_candidate_gate_quality():
    state = _state()
    decisions = CandidateDecisionPipeline().evaluate(state)
    state.candidate_decisions = {
        code: {
            "eligible_for_ranking": decision.eligible_for_ranking,
        }
        for code, decision in decisions.items()
    }
    score = ConfidenceScorer().score(state)
    assert score.candidate_quality == 0.5
