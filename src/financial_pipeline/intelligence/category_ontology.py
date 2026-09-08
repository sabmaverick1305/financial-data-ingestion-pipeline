"""Canonical mutual-fund category taxonomy for production reasoning."""
from __future__ import annotations
import re

_RULES: tuple[tuple[str, str], ...] = (
    (r"large\s*&\s*mid\s*cap", "Large & Mid Cap"),
    (r"large\s*cap", "Large Cap"),
    (r"mid\s*cap", "Mid Cap"),
    (r"small\s*cap", "Small Cap"),
    (r"flexi\s*cap", "Flexi Cap"),
    (r"multi\s*cap", "Multi Cap"),
    (r"focused|focussed", "Focused Fund"),
    (r"value\s*fund", "Value Fund"),
    (r"contra\s*fund", "Contra Fund"),
    (r"dividend\s*yield", "Dividend Yield Fund"),
    (r"sectoral|thematic", "Sectoral/Thematic"),
    (r"elss|tax\s*saver", "ELSS"),
    (r"aggressive\s*hybrid", "Aggressive Hybrid"),
    (r"conservative\s*hybrid", "Conservative Hybrid"),
    (r"balanced\s*advantage|dynamic\s*asset\s*allocation", "Balanced Advantage"),
    (r"multi\s*asset", "Multi Asset Allocation"),
    (r"equity\s*savings", "Equity Savings"),
    (r"arbitrage", "Arbitrage"),
    (r"liquid\s*fund|\bliquid\b", "Liquid Fund"),
    (r"overnight", "Overnight Fund"),
    (r"money\s*market", "Money Market Fund"),
    (r"ultra\s*short", "Ultra Short Duration"),
    (r"low\s*duration", "Low Duration"),
    (r"short\s*duration|short\s*term", "Short Duration"),
    (r"medium\s*duration|medium\s*term", "Medium Duration"),
    (r"long\s*duration|long\s*term", "Long Duration"),
    (r"corporate\s*bond", "Corporate Bond Fund"),
    (r"credit\s*risk", "Credit Risk Fund"),
    (r"banking\s*and\s*psu", "Banking & PSU Fund"),
    (r"gilt", "Gilt Fund"),
    (r"floater|floating\s*interest", "Floater Fund"),
    (r"index\s*fund", "Index Fund"),
    (r"equity\s*etf", "Equity ETF"),
    (r"debt\s*etf", "Debt ETF"),
    (r"gold\s*etf", "Gold ETF"),
    (r"silver\s*etf", "Silver ETF"),
    (r"fund\s*of\s*funds|fof", "Fund of Funds"),
    (r"retirement", "Retirement Fund"),
    (r"children", "Children Fund"),
)

class CategoryOntology:
    def canonicalize(self, raw: str) -> str | None:
        value = raw.strip()
        if not value:
            return None
        lowered = value.lower()
        for pattern, canonical in _RULES:
            if re.search(pattern, lowered):
                return canonical
        return None

    def is_supported_category(self, raw: str) -> bool:
        return self.canonicalize(raw) is not None
