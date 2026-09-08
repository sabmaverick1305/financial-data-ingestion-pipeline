"""Deterministic investment-mandate and fund-category eligibility policy."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
import re

class InvestmentMandate(StrEnum):
    CORE_DIVERSIFIED = "core_diversified"
    AGGRESSIVE_GROWTH = "aggressive_growth"
    BALANCED = "balanced"
    INCOME = "income"
    TAX_SAVING = "tax_saving"
    COMMODITY = "commodity"
    THEMATIC = "thematic"
    SPECIFIC_CATEGORY = "specific_category"

@dataclass(frozen=True)
class MandateDecision:
    mandate: InvestmentMandate
    eligible_categories: tuple[str, ...]
    excluded_categories: tuple[str, ...]
    rationale: str

_CORE = (
    "Large Cap",
    "Large & Mid Cap",
    "Flexi Cap",
    "Multi Cap",
    "Mid Cap",
    "Small Cap",
)
_AGGRESSIVE = (
    "Large & Mid Cap",
    "Flexi Cap",
    "Multi Cap",
    "Mid Cap",
    "Small Cap",
)
_BALANCED = (
    "Aggressive Hybrid",
    "Balanced Advantage",
    "Multi Asset Allocation",
    "Equity Savings",
)
_INCOME = (
    "Corporate Bond Fund",
    "Banking & PSU Fund",
    "Short Duration",
    "Medium Duration",
    "Gilt Fund",
    "Floater Fund",
)
_SPECIALIZED = (
    "Sectoral/Thematic",
    "Gold ETF",
    "Silver ETF",
    "Fund of Funds",
    "Equity ETF",
    "Debt ETF",
    "Credit Risk Fund",
    "Children Fund",
    "Retirement Fund",
)

_CATEGORY_PATTERNS: tuple[tuple[str, str], ...] = (
    (r"large\s*&\s*mid\s*cap", "Large & Mid Cap"),
    (r"large\s*cap", "Large Cap"),
    (r"mid\s*cap", "Mid Cap"),
    (r"small\s*cap", "Small Cap"),
    (r"flexi\s*cap", "Flexi Cap"),
    (r"multi\s*cap", "Multi Cap"),
    (r"aggressive\s*hybrid", "Aggressive Hybrid"),
    (r"balanced\s*advantage|dynamic\s*asset", "Balanced Advantage"),
    (r"multi\s*asset", "Multi Asset Allocation"),
    (r"equity\s*savings", "Equity Savings"),
    (r"elss|tax\s*saver|tax\s*saving", "ELSS"),
    (r"gold", "Gold ETF"),
    (r"silver", "Silver ETF"),
    (r"sectoral|thematic", "Sectoral/Thematic"),
    (r"corporate\s*bond", "Corporate Bond Fund"),
    (r"banking\s*(?:&|and)\s*psu", "Banking & PSU Fund"),
    (r"short\s*duration", "Short Duration"),
    (r"medium\s*duration", "Medium Duration"),
    (r"gilt", "Gilt Fund"),
)

class InvestmentMandatePolicy:
    def decide(self, query: str) -> MandateDecision:
        q = query.lower()

        explicit = []
        for pattern, category in _CATEGORY_PATTERNS:
            if re.search(pattern, q):
                explicit.append(category)
        explicit = list(dict.fromkeys(explicit))
        if explicit:
            category = explicit[0]
            if category == "ELSS":
                mandate = InvestmentMandate.TAX_SAVING
            elif category in {"Gold ETF", "Silver ETF"}:
                mandate = InvestmentMandate.COMMODITY
            elif category == "Sectoral/Thematic":
                mandate = InvestmentMandate.THEMATIC
            else:
                mandate = InvestmentMandate.SPECIFIC_CATEGORY
            return MandateDecision(
                mandate=mandate,
                eligible_categories=tuple(explicit),
                excluded_categories=tuple(
                    category for category in _SPECIALIZED if category not in explicit
                ),
                rationale="explicit category intent from user query",
            )

        if any(term in q for term in ("aggressive", "high growth", "higher growth", "high risk")):
            return MandateDecision(
                InvestmentMandate.AGGRESSIVE_GROWTH,
                _AGGRESSIVE,
                _SPECIALIZED,
                "aggressive-growth mandate inferred from query",
            )

        if any(term in q for term in ("balanced", "moderate risk", "hybrid")):
            return MandateDecision(
                InvestmentMandate.BALANCED,
                _BALANCED,
                _SPECIALIZED,
                "balanced mandate inferred from query",
            )

        if any(term in q for term in ("income", "debt fund", "lower risk", "capital preservation")):
            return MandateDecision(
                InvestmentMandate.INCOME,
                _INCOME,
                _SPECIALIZED,
                "income-oriented mandate inferred from query",
            )

        return MandateDecision(
            InvestmentMandate.CORE_DIVERSIFIED,
            _CORE,
            _SPECIALIZED,
            "default diversified-core research mandate for an unqualified best-funds query",
        )
