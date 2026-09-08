"""Semantic documentary evidence policy for mutual-fund research."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SemanticEvidenceRequirement:
    key: str
    accepted_document_types: tuple[str, ...]
    content_terms: tuple[str, ...] = ()
    match_mode: str = "any"
    minimum_authority: str = "authoritative"


# Evidence requirements describe *claims we need to prove*, not filenames we
# expect to exist. One authoritative SID can therefore satisfy both mandate and
# strategy when its indexed content actually contains the required semantics.
SEMANTIC_EVIDENCE_REQUIREMENTS: tuple[SemanticEvidenceRequirement, ...] = (
    SemanticEvidenceRequirement(
        key="scheme_mandate",
        accepted_document_types=(
            "scheme_information_document",
            "fund_prospectus",
            "key_information_memorandum",
        ),
        content_terms=("investment objective", "asset allocation"),
        match_mode="all",
    ),
    SemanticEvidenceRequirement(
        key="investment_strategy",
        accepted_document_types=(
            "scheme_information_document",
            "fund_prospectus",
            "fund_fact_sheet",
            "fund_strategy_document",
        ),
        content_terms=(
            "investment strategy",
            "investment approach",
            "investment objective",
        ),
        match_mode="any",
    ),
    SemanticEvidenceRequirement(
        key="portfolio_composition",
        accepted_document_types=(
            "fund_fact_sheet",
            "portfolio_disclosure",
        ),
        content_terms=("portfolio", "holding", "sector allocation"),
        match_mode="any",
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

PLANNER_EVIDENCE_TO_SEMANTIC = {
    "prospectus": "scheme_mandate",
    "fund_prospectus": "scheme_mandate",
    "scheme_information_document": "scheme_mandate",
    "sid": "scheme_mandate",
    "key_information_memorandum": "scheme_mandate",
    "kim": "scheme_mandate",
    "fact_sheet": "portfolio_composition",
    "factsheet": "portfolio_composition",
    "fund_fact_sheet": "portfolio_composition",
    "strategy": "investment_strategy",
    "fund_strategy_document": "investment_strategy",
    "investment_strategy": "investment_strategy",
    "disclosures": "portfolio_composition",
    "portfolio_disclosure": "portfolio_composition",
}


def semantic_requirements_for_evidence_types(
    evidence_types: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    requirements: list[str] = []
    for value in evidence_types:
        normalized = str(value).lower().replace(" ", "_")
        semantic = PLANNER_EVIDENCE_TO_SEMANTIC.get(normalized)
        if semantic and semantic not in requirements:
            requirements.append(semantic)
    return tuple(requirements) or DEFAULT_BETA_REQUIREMENTS
