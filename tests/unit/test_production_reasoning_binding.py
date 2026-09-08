from financial_pipeline.intelligence.capability_contracts import CapabilityContract
from financial_pipeline.intelligence.metric_ontology import MetricOntology
from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.production_registry import build_production_registry
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionObservation, ActionStatus, ActionType, ResearchAction
from financial_pipeline.intelligence.state_binding import ActionStateBinder


def test_state_binding_injects_discovered_scheme_codes_and_categories() -> None:
    state = ReasoningState(query="q")
    state.add_observation(ActionObservation(
        action_id="funds",
        action_type=ActionType.DISCOVER_FUNDS,
        status=ActionStatus.SUCCEEDED,
        result={"funds": [
            {"scheme_code": "1", "category": "Mid Cap"},
            {"scheme_code": "2", "category": "Mid Cap"},
        ]},
    ))
    binder = ActionStateBinder()
    perf = binder.bind(state, ResearchAction(ActionType.FETCH_PERFORMANCE))
    risk = binder.bind(state, ResearchAction(ActionType.COMPUTE_RISK))
    peers = binder.bind(state, ResearchAction(ActionType.COMPARE_PEERS))
    assert perf.parameters["scheme_codes"] == ["1", "2"]
    assert risk.parameters["scheme_codes"] == ["1", "2"]
    assert peers.parameters["categories"] == ["Mid Cap"]


def test_live_metric_variants_normalize_before_contract_validation() -> None:
    ontology = MetricOntology()
    assert ontology.canonicalize("return_1yr") == "return_1y"
    assert ontology.canonicalize("return_3yr") == "return_3y_cagr"
    assert ontology.canonicalize("return_5yr") == "return_5y_cagr"
    assert ontology.canonicalize("return_3y") == "return_3y_cagr"
    assert ontology.canonicalize("return_5y") == "return_5y_cagr"
    assert ontology.canonicalize("aum_current") == "aum"
    assert ontology.canonicalize("inflow_1yr") == "net_inflow"
    assert ontology.canonicalize("return_rank") == "percentile_rank"
    assert ontology.canonicalize("risk_rank") == "percentile_rank"
    assert ontology.canonicalize("sharpe_rank") == "percentile_rank"


def test_production_rag_contract_accepts_documentary_semantics() -> None:
    class Pack:
        discover_categories = discover_funds = performance = returns = risk = peer_compare = aum = flows = documentary = contradictions = lambda self, action: None
    registry = build_production_registry(Pack())
    contract = registry.get(ActionType.RETRIEVE_EVIDENCE).contract
    action = ResearchAction(
        ActionType.RETRIEVE_EVIDENCE,
        evidence_types=("prospectus", "fact_sheet", "strategy", "filings", "reports", "disclosures"),
    )
    normalized, unsupported_metrics, unsupported_evidence = contract.normalize(action)
    assert unsupported_metrics == ()
    assert unsupported_evidence == ()
    assert normalized.evidence_types == (
        "fund_prospectus",
        "fund_fact_sheet",
        "fund_strategy_document",
        "regulatory_filing",
        "annual_report",
        "portfolio_disclosure",
    )


def test_production_registry_covers_all_action_types() -> None:
    class Pack:
        discover_categories = discover_funds = performance = returns = risk = peer_compare = aum = flows = documentary = contradictions = lambda self, action: None

    registry = build_production_registry(Pack())
    missing = [action_type for action_type in ActionType if not registry.has(action_type)]
    assert missing == []
