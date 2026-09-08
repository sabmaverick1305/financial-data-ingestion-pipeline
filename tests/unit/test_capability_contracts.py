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
