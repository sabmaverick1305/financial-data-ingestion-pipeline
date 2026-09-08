"""Semantic validation for retrieved documentary evidence."""
from __future__ import annotations

_INSUFFICIENT_PHRASES = (
    "i don't have that information",
    "do not include",
    "not available in the provided documents",
    "not found in the provided documents",
    "provided context does not contain",
)

_KEYWORDS = {
    "prospectus": ("prospectus", "scheme information document", "sid"),
    "fund_prospectus": ("prospectus", "scheme information document", "sid"),
    "scheme_information_document": ("scheme information document", "sid", "investment objective"),
    "fact_sheet": ("fact sheet", "factsheet", "portfolio"),
    "factsheet": ("fact sheet", "factsheet", "portfolio"),
    "strategy": ("strategy", "investment approach", "investment objective"),
    "disclosures": ("portfolio", "disclosure", "holdings"),
    "fund_prospectus": ("prospectus", "scheme information document", "sid"),
    "fund_fact_sheet": ("fact sheet", "factsheet"),
    "fund_strategy_document": ("strategy", "investment objective"),
    "regulatory_filing": ("filing", "sebi", "regulatory"),
    "annual_report": ("annual report",),
    "portfolio_disclosure": ("portfolio", "disclosure", "holdings"),
}

class DocumentaryEvidenceValidator:
    def validate(self, *, answer: str, sources: list[dict], requested: tuple[str, ...]) -> tuple[bool, str | None]:
        text = (answer or "").strip().lower()
        if any(phrase in text for phrase in _INSUFFICIENT_PHRASES):
            return False, "retrieval completed but requested documentary evidence was not present"

        haystack = " ".join(
            str(source.get(key, ""))
            for source in sources
            for key in ("file_name", "title", "category", "preview", "document_type", "text")
        ).lower()

        if requested:
            matched_any = False
            for evidence_type in requested:
                keywords = _KEYWORDS.get(evidence_type, ())
                if keywords and any(keyword in haystack for keyword in keywords):
                    matched_any = True
                    break
            if not matched_any:
                return False, "retrieved sources do not match the requested documentary evidence types"

        return True, None
