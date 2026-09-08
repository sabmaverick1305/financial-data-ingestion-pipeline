from financial_pipeline.intelligence.capability_contracts import CapabilityContract
from financial_pipeline.intelligence.capability_registry import CapabilityRegistry, CapabilityResult
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType, ResearchAction


def test_contract_normalizes_supported_metric_aliases_before_execution() -> None:
    registry = CapabilityRegistry()
    seen: list[tuple[str, ...]] = []

    def handler(action: ResearchAction):
        seen.append(action.metrics)
        return CapabilityResult(result={"ok": True}, evidence_refs=("verified:performance",))

    registry.register(
        ActionType.FETCH_PERFORMANCE,
        handler,
        contract=CapabilityContract(
            supported_metrics=("return_1y", "return_3y_cagr"),
            metric_aliases={"1yr_return": "return_1y", "3yr_return": "return_3y_cagr"},
        ),
    )

    observation = registry.execute(
        ResearchAction(
            ActionType.FETCH_PERFORMANCE,
            metrics=("1yr_return", "3yr_return"),
        )
    )

    assert observation.status is ActionStatus.SUCCEEDED
    assert seen == [("return_1y", "return_3y_cagr")]


def test_contract_returns_partial_when_some_metrics_are_unsupported() -> None:
    registry = CapabilityRegistry()
    seen: list[tuple[str, ...]] = []

    def handler(action: ResearchAction):
        seen.append(action.metrics)
        return CapabilityResult(result={"expense_ratio": 0.8}, evidence_refs=("verified:evidence",))

    registry.register(
        ActionType.RETRIEVE_EVIDENCE,
        handler,
        contract=CapabilityContract(supported_metrics=("expense_ratio",)),
    )

    observation = registry.execute(
        ResearchAction(
            ActionType.RETRIEVE_EVIDENCE,
            metrics=("expense_ratio", "morningstar_rating"),
        )
    )

    assert seen == [("expense_ratio",)]
    assert observation.status is ActionStatus.PARTIAL
    assert observation.evidence_refs == ("verified:evidence",)
    assert observation.tradeoff_reason == (
        "unsupported metrics omitted by capability contract: morningstar_rating"
    )


def test_contract_fails_without_calling_handler_when_all_metrics_are_unsupported() -> None:
    registry = CapabilityRegistry()
    calls = 0

    def handler(action: ResearchAction):
        nonlocal calls
        calls += 1
        return {"should_not": "run"}

    registry.register(
        ActionType.RETRIEVE_EVIDENCE,
        handler,
        contract=CapabilityContract(
            supported_metrics=("fund_manager_tenure", "expense_ratio"),
        ),
    )

    observation = registry.execute(
        ResearchAction(
            ActionType.RETRIEVE_EVIDENCE,
            metrics=("morningstar_rating", "fund_quality_score"),
        )
    )

    assert calls == 0
    assert observation.status is ActionStatus.FAILED
    assert observation.error == "unsupported capability metrics"
    assert "morningstar_rating" in observation.tradeoff_reason
    assert "fund_quality_score" in observation.tradeoff_reason


def test_metric_ontology_normalizes_natural_financial_language() -> None:
    registry = CapabilityRegistry()
    seen: list[tuple[str, ...]] = []

    def handler(action: ResearchAction):
        seen.append(action.metrics)
        return CapabilityResult(result={"ok": True}, evidence_refs=("verified:performance",))

    registry.register(
        ActionType.FETCH_PERFORMANCE,
        handler,
        contract=CapabilityContract(
            supported_metrics=("return_1y", "return_3y_cagr", "return_5y_cagr"),
        ),
    )

    observation = registry.execute(
        ResearchAction(
            ActionType.FETCH_PERFORMANCE,
            metrics=("1-year return", "3-year return", "5-year return"),
        )
    )

    assert observation.status is ActionStatus.SUCCEEDED
    assert seen == [("return_1y", "return_3y_cagr", "return_5y_cagr")]


def test_metric_ontology_keeps_genuinely_unsupported_concepts_visible() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ActionType.RETRIEVE_EVIDENCE,
        lambda action: CapabilityResult(result={"ok": True}),
        contract=CapabilityContract(supported_metrics=("expense_ratio",)),
    )

    observation = registry.execute(
        ResearchAction(
            ActionType.RETRIEVE_EVIDENCE,
            metrics=("expense ratio", "credit quality"),
        )
    )

    assert observation.status is ActionStatus.PARTIAL
    assert observation.tradeoff_reason == (
        "unsupported metrics omitted by capability contract: credit quality"
    )


def test_metric_ontology_normalizes_flow_aum_and_peer_language() -> None:
    cases = (
        (ActionType.FETCH_FLOWS, ("net_inflow", "flow_trend"), ("net inflows", "flow trend")),
        (ActionType.FETCH_AUM, ("aum", "aum_trend"), ("total AUM", "AUM trend")),
        (
            ActionType.COMPARE_PEERS,
            ("percentile_rank", "peer_outperformance"),
            ("return ranking", "relative performance"),
        ),
    )

    for action_type, supported, requested in cases:
        registry = CapabilityRegistry()
        seen: list[tuple[str, ...]] = []

        def handler(action: ResearchAction):
            seen.append(action.metrics)
            return CapabilityResult(result={"ok": True})

        registry.register(
            action_type,
            handler,
            contract=CapabilityContract(supported_metrics=supported),
        )
        observation = registry.execute(ResearchAction(action_type, metrics=requested))
        assert observation.status is ActionStatus.SUCCEEDED
        assert seen == [supported]


def test_evidence_ontology_normalizes_document_types_separately_from_metrics() -> None:
    registry = CapabilityRegistry()
    seen: list[ResearchAction] = []

    def handler(action: ResearchAction):
        seen.append(action)
        return CapabilityResult(result={"ok": True}, evidence_refs=("verified:evidence",))

    registry.register(
        ActionType.RETRIEVE_EVIDENCE,
        handler,
        contract=CapabilityContract(
            supported_metrics=("expense_ratio", "fund_manager_tenure"),
            supported_evidence_types=("fund_prospectus", "fund_fact_sheet", "regulatory_filing"),
        ),
    )
    observation = registry.execute(
        ResearchAction(
            ActionType.RETRIEVE_EVIDENCE,
            metrics=("expense ratio", "manager tenure"),
            evidence_types=("prospectus", "fact sheet", "SEBI filings"),
        )
    )

    assert observation.status is ActionStatus.SUCCEEDED
    assert seen[0].metrics == ("expense_ratio", "fund_manager_tenure")
    assert seen[0].evidence_types == ("fund_prospectus", "fund_fact_sheet", "regulatory_filing")


def test_evidence_ontology_rejects_unknown_document_type_without_corrupting_metrics() -> None:
    registry = CapabilityRegistry()
    seen: list[ResearchAction] = []

    def handler(action: ResearchAction):
        seen.append(action)
        return CapabilityResult(result={"ok": True}, evidence_refs=("verified:evidence",))

    registry.register(
        ActionType.RETRIEVE_EVIDENCE,
        handler,
        contract=CapabilityContract(
            supported_metrics=("expense_ratio",),
            supported_evidence_types=("fund_fact_sheet",),
        ),
    )
    observation = registry.execute(
        ResearchAction(
            ActionType.RETRIEVE_EVIDENCE,
            metrics=("expense ratio",),
            evidence_types=("fact sheet", "social media rumor"),
        )
    )

    assert observation.status is ActionStatus.PARTIAL
    assert seen[0].metrics == ("expense_ratio",)
    assert seen[0].evidence_types == ("fund_fact_sheet",)
    assert "social media rumor" in observation.tradeoff_reason


def test_retrieve_evidence_migrates_legacy_document_terms_out_of_metrics() -> None:
    registry = CapabilityRegistry()
    seen: list[ResearchAction] = []

    def handler(action: ResearchAction):
        seen.append(action)
        return CapabilityResult(result={"ok": True}, evidence_refs=("verified:evidence",))

    registry.register(
        ActionType.RETRIEVE_EVIDENCE,
        handler,
        contract=CapabilityContract(
            supported_metrics=("expense_ratio",),
            supported_evidence_types=(
                "fund_strategy_document", "regulatory_filing", "annual_report"
            ),
        ),
    )
    observation = registry.execute(
        ResearchAction(
            ActionType.RETRIEVE_EVIDENCE,
            metrics=("expense ratio", "strategy", "filings", "reports", "disclosures"),
        )
    )

    assert observation.status is ActionStatus.SUCCEEDED
    assert seen[0].metrics == ("expense_ratio",)
    assert seen[0].evidence_types == (
        "fund_strategy_document", "regulatory_filing", "annual_report"
    )


def test_documentary_replanner_populates_metrics_and_evidence_types() -> None:
    registry = CapabilityRegistry()
    registry.register(
        ActionType.RETRIEVE_EVIDENCE,
        lambda action: CapabilityResult(result={"ok": True}),
        contract=CapabilityContract(
            supported_metrics=("expense_ratio", "fund_manager_tenure"),
            supported_evidence_types=("fund_prospectus", "fund_fact_sheet"),
        ),
    )
    replanner = EvidenceReplanner(registry)
    evaluation = EvidenceEvaluator().evaluate(
        ReasoningState(query="best mutual funds"),
        (EvidenceRequirement(EvidenceDimension.DOCUMENTARY),),
    )

    decision = replanner.decide(evaluation)
    action = decision.actions[0]
    assert action.metrics == ("expense_ratio", "fund_manager_tenure")
    assert action.evidence_types == ("fund_prospectus", "fund_fact_sheet")
    assert "capability-supported semantics" in action.rationale
