"""Semantic documentary evidence policy for mutual-fund research."""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class SemanticEvidenceRequirement:
    key: str
    accepted_document_types: tuple[str, ...]
    content_terms: tuple[str, ...] = ()

SEMANTIC_EVIDENCE_REQUIREMENTS: tuple[SemanticEvidenceRequirement, ...] = (
    SemanticEvidenceRequirement(
        key="scheme_mandate",
        accepted_document_types=(
            "fund_prospectus",
            "scheme_information_document",
            "key_information_memorandum",
        ),
        content_terms=(
            "investment objective",
            "asset allocation",
        ),
    ),
    SemanticEvidenceRequirement(
        key="investment_strategy",
        accepted_document_types=(
            "fund_strategy_document",
            "scheme_information_document",
            "fund_prospectus",
            "fund_fact_sheet",
        ),
        content_terms=(
            "investment strategy",
            "investment approach",
            "investment objective",
        ),
    ),
    SemanticEvidenceRequirement(
        key="portfolio_composition",
        accepted_document_types=(
            "fund_fact_sheet",
            "portfolio_disclosure",
        ),
        content_terms=(
            "portfolio",
            "holding",
            "sector allocation",
        ),
    ),
)

REQUIREMENT_BY_KEY = {
    requirement.key: requirement
    for requirement in SEMANTIC_EVIDENCE_REQUIREMENTS
}

DEFAULT_BETA_REQUIREMENTS = (
    "scheme_mandate",
    "investment_strategy",
    "portfolio_composition",
)
