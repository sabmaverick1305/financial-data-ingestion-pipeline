"""Deterministic contradiction rules for mutual-fund reasoning."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Contradiction:
    code: str
    severity: str
    reason: str

class ContradictionEngine:
    def evaluate(self, facts: dict[str, Any]) -> tuple[Contradiction, ...]:
        out: list[Contradiction] = []
        if facts.get("return_3y_cagr", 0) > 15 and facts.get("net_inflow", 0) < 0:
            out.append(Contradiction("strong_return_persistent_outflow", "medium", "Strong return conflicts with negative fund flows"))
        if facts.get("return_5y_cagr", 0) > 15 and facts.get("max_drawdown", 0) > 30:
            out.append(Contradiction("strong_cagr_extreme_drawdown", "high", "High CAGR carries extreme drawdown"))
        if facts.get("aum_trend", 0) > 0 and facts.get("percentile_rank", 100) > 75:
            out.append(Contradiction("rising_aum_weak_peer_rank", "medium", "AUM rising despite weak peer rank"))
        if facts.get("strategy_mismatch") is True:
            out.append(Contradiction("strategy_holdings_mismatch", "high", "Documented strategy conflicts with observed holdings"))
        return tuple(out)
