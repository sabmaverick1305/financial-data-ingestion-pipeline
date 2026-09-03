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


def test_named_fund_nav_query_is_not_out_of_scope_when_amc_identified() -> None:
    # Regression test: "NAV ... today" for a named, identified fund is
    # answerable via mf_scheme_performance.latest_nav (fund_performance_sql)
    # and must not be hard-blocked as out-of-scope — see guardrails.py's
    # _AMBIGUOUS_NAV_PATTERNS comment for the verified false-positive this
    # covers ("What is the NAV of SBI Large cap fund today?").
    result = PreGenerationGuardrails().check(
        question="What is the NAV of SBI Large cap fund today?",
        citations=[],
        intent_type="factual",
        has_identified_fund=True,
    )

    assert result.should_proceed is True
    assert result.is_out_of_scope is False


def test_ambiguous_nav_query_without_identified_fund_is_still_out_of_scope() -> None:
    # Without a named AMC/scheme, "NAV today" is still ambiguous (mutual
    # funds have no intraday NAV) — the block should still apply.
    result = PreGenerationGuardrails().check(
        question="What is the current NAV today?",
        citations=[],
        intent_type="factual",
        has_identified_fund=False,
    )

    assert result.should_proceed is False
    assert result.is_out_of_scope is True


def test_genuinely_out_of_scope_query_is_blocked_even_with_identified_fund() -> None:
    # has_identified_fund only narrows the NAV-specific patterns — topics
    # that are always out of scope (stock prices, crypto, etc.) must still
    # block regardless.
    result = PreGenerationGuardrails().check(
        question="What is the current stock price of SBI?",
        citations=[],
        intent_type="factual",
        has_identified_fund=True,
    )

    assert result.should_proceed is False
    assert result.is_out_of_scope is True
