from financial_pipeline.intelligence.documentary_validation import DocumentaryEvidenceValidator
from financial_pipeline.intelligence.evidence import EvidenceDimension, EvidenceEvaluator, EvidenceRequirement
from financial_pipeline.intelligence.fund_ranking import FundRanker
from financial_pipeline.intelligence.metric_routing import MetricOwnershipRouter
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionObservation, ActionStatus, ActionType, ResearchAction, ResearchPlan
from financial_pipeline.intelligence.state_binding import ActionStateBinder

def test_documentary_abstention_does_not_count_as_valid_evidence():
    validator = DocumentaryEvidenceValidator()
    valid, reason = validator.validate(
        answer="I don't have that information in the provided documents.",
        sources=[{"file_name": "monthly-aum.pdf", "preview": "AUM report"}],
        requested=("fund_prospectus", "fund_fact_sheet"),
    )
    assert valid is False
    assert reason

def test_compute_returns_gets_owned_default_metrics():
    plan = ResearchPlan(objective="returns", actions=(ResearchAction(ActionType.COMPUTE_RETURNS),))
    action = MetricOwnershipRouter().route(plan).actions[0]
    assert action.metrics == ("total_return_3y", "annualized_return")

def test_peer_binding_uses_candidate_categories_not_global_taxonomy():
    state = ReasoningState(query="q")
    state.add_observation(ActionObservation("c", ActionType.DISCOVER_CATEGORIES, ActionStatus.SUCCEEDED, result={"categories":["Large Cap","Mid Cap","Small Cap"]}))
    state.add_observation(ActionObservation("f", ActionType.DISCOVER_FUNDS, ActionStatus.SUCCEEDED, result={"funds":[{"scheme_code":"1","category":"Flexi Cap"},{"scheme_code":"2","category":"Focused Fund"}]}))
    action = ActionStateBinder().bind(state, ResearchAction(ActionType.COMPARE_PEERS))
    assert action.parameters["categories"] == ["Flexi Cap", "Focused Fund"]

def test_latest_partial_overrides_older_success_for_evidence():
    state = ReasoningState(query="q")
    state.add_observation(ActionObservation("1", ActionType.RETRIEVE_EVIDENCE, ActionStatus.SUCCEEDED))
    state.add_observation(ActionObservation("2", ActionType.RETRIEVE_EVIDENCE, ActionStatus.PARTIAL, tradeoff_reason="missing docs"))
    evaluation = EvidenceEvaluator().evaluate(state, (EvidenceRequirement(EvidenceDimension.DOCUMENTARY),))
    assert evaluation.is_sufficient is False
    assert evaluation.failed == (EvidenceDimension.DOCUMENTARY,)

def test_ranker_prefers_stronger_return_and_risk_adjusted_candidate():
    state = ReasoningState(query="q")
    state.add_observation(ActionObservation("f", ActionType.DISCOVER_FUNDS, ActionStatus.SUCCEEDED, result={"funds":[
        {"scheme_code":"1","scheme_name":"A","category":"Flexi Cap"},
        {"scheme_code":"2","scheme_name":"B","category":"Flexi Cap"},
    ]}))
    state.add_observation(ActionObservation("p", ActionType.FETCH_PERFORMANCE, ActionStatus.SUCCEEDED, result={"scope":"scheme_batch","rows":[
        {"scheme_code":"1","metrics":{"return_1y":15,"return_3y_cagr":18}},
        {"scheme_code":"2","metrics":{"return_1y":8,"return_3y_cagr":10}},
    ]}))
    state.add_observation(ActionObservation("r", ActionType.COMPUTE_RISK, ActionStatus.SUCCEEDED, result={"scope":"scheme_batch","rows":[
        {"scheme_code":"1","metrics":{"sharpe_ratio":1.2,"volatility":12,"max_drawdown":10}},
        {"scheme_code":"2","metrics":{"sharpe_ratio":0.4,"volatility":20,"max_drawdown":25}},
    ]}))
    ranked = FundRanker().rank(state)
    assert ranked[0].scheme_code == "1"
    assert ranked[0].score > ranked[1].score
