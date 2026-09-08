from financial_pipeline.processing.extractor import TextExtractor
from financial_pipeline.documentary.evidence_policy import (
    DEFAULT_BETA_REQUIREMENTS,
    REQUIREMENT_BY_KEY,
)


def test_authoritative_html_is_extractable_without_docling():
    raw = b"""
    <html>
      <head><style>.x{display:none}</style><script>ignore_me()</script></head>
      <body>
        <h1>Bandhan Small Cap Fund</h1>
        <p>Portfolio holdings and sector allocation as on June 30 2026.</p>
      </body>
    </html>
    """
    result = TextExtractor().extract(raw, "html")
    assert result.has_text_layer is True
    assert result.extraction_engine == "stdlib_html_parser"
    assert "Bandhan Small Cap Fund" in result.full_text
    assert "Portfolio holdings" in result.full_text
    assert "ignore_me" not in result.full_text


def test_beta_semantic_requirements_accept_authoritative_sid_and_factsheet():
    mandate = REQUIREMENT_BY_KEY["scheme_mandate"]
    strategy = REQUIREMENT_BY_KEY["investment_strategy"]
    portfolio = REQUIREMENT_BY_KEY["portfolio_composition"]

    assert "scheme_information_document" in mandate.accepted_document_types
    assert "scheme_information_document" in strategy.accepted_document_types
    assert "fund_fact_sheet" in portfolio.accepted_document_types
    assert DEFAULT_BETA_REQUIREMENTS == (
        "scheme_mandate",
        "investment_strategy",
        "portfolio_composition",
    )
