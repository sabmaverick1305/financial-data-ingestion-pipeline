from financial_pipeline.augmentation.guardrails import PreGenerationGuardrails
from financial_pipeline.retrieval.query_understanding import QueryAnalyzer


QUERY = "What is the best performing fund for the last 5 years with good CAGR around 7%"


def test_cagr_performance_query_is_not_routed_to_analytical() -> None:
    intent = QueryAnalyzer().analyze(QUERY)

    assert "investment_advice" in intent.labels
    assert intent.aggregation == "cagr"
    assert intent.metric is None
    assert intent.needs_analytical is False
    assert intent.intent_type == "factual"


def test_cagr_performance_query_is_blocked_before_generation() -> None:
    result = PreGenerationGuardrails().check(
        question=QUERY,
        citations=[],
        intent_type="factual",
    )

    assert result.should_proceed is False
    assert result.is_investment_advice is True
    assert result.block_reason is not None
