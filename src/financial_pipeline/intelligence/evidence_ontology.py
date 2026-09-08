"""Canonical documentary/evidence vocabulary for FIES retrieval actions."""

from __future__ import annotations

import re


def _key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = re.sub(r"[-_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


_ALIASES: dict[str, str] = {
    "prospectus": "fund_prospectus",
    "fund prospectus": "fund_prospectus",
    "scheme prospectus": "fund_prospectus",
    "offer document": "fund_prospectus",
    "sid": "fund_prospectus",
    "scheme information document": "fund_prospectus",
    "fact sheet": "fund_fact_sheet",
    "factsheet": "fund_fact_sheet",
    "fund fact sheet": "fund_fact_sheet",
    "monthly fact sheet": "fund_fact_sheet",
    "fund strategy": "fund_strategy_document",
    "strategy": "fund_strategy_document",
    "investment strategy": "fund_strategy_document",
    "investment objective": "fund_strategy_document",
    "strategy document": "fund_strategy_document",
    "regulatory filing": "regulatory_filing",
    "regulatory filings": "regulatory_filing",
    "filing": "regulatory_filing",
    "filings": "regulatory_filing",
    "disclosure": "regulatory_filing",
    "disclosures": "portfolio_disclosure",
    "sebi filing": "regulatory_filing",
    "sebi filings": "regulatory_filing",
    "disclosure": "regulatory_filing",
    "regulatory disclosure": "regulatory_filing",
    "annual report": "annual_report",
    "report": "annual_report",
    "reports": "annual_report",
    "scheme annual report": "annual_report",
    "portfolio disclosure": "portfolio_disclosure",
    "portfolio statement": "portfolio_disclosure",
}


class EvidenceOntology:
    def canonicalize(self, evidence_type: str) -> str:
        key = _key(evidence_type)
        if key in _ALIASES:
            return _ALIASES[key]
        return re.sub(r"[^a-z0-9]+", "_", key).strip("_") or evidence_type

    def is_known(self, value: str) -> bool:
        return _key(value) in _ALIASES
