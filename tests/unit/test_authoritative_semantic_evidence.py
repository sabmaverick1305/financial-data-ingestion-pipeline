from financial_pipeline.documentary.evidence_policy import (
    DEFAULT_BETA_REQUIREMENTS,
    REQUIREMENT_BY_KEY,
    semantic_requirements_for_evidence_types,
)
from financial_pipeline.intelligence.documentary_validation import DocumentaryEvidenceValidator


def test_sid_can_satisfy_mandate_and_strategy_semantics():
    mandate = REQUIREMENT_BY_KEY["scheme_mandate"]
    strategy = REQUIREMENT_BY_KEY["investment_strategy"]

    assert "scheme_information_document" in mandate.accepted_document_types
    assert "scheme_information_document" in strategy.accepted_document_types
    assert "fund_strategy_document" not in mandate.accepted_document_types


def test_factsheet_or_disclosure_can_satisfy_portfolio_composition():
    portfolio = REQUIREMENT_BY_KEY["portfolio_composition"]
    assert set(portfolio.accepted_document_types) == {
        "fund_fact_sheet",
        "portfolio_disclosure",
    }


def test_beta_semantic_requirements_are_not_physical_file_names():
    assert DEFAULT_BETA_REQUIREMENTS == (
        "scheme_mandate",
        "investment_strategy",
        "portfolio_composition",
    )
    assert semantic_requirements_for_evidence_types(
        ("prospectus", "fact_sheet", "strategy", "disclosures")
    ) == (
        "scheme_mandate",
        "portfolio_composition",
        "investment_strategy",
    )


def test_grounding_validator_accepts_canonical_document_type_aliases():
    valid, reason = DocumentaryEvidenceValidator().validate(
        answer="The investment objective and strategy are described in the SID.",
        sources=[
            {
                "document_type": "scheme_information_document",
                "text": "Investment objective and investment strategy",
            }
        ],
        requested=("prospectus", "strategy"),
    )
    assert valid is True
    assert reason is None
