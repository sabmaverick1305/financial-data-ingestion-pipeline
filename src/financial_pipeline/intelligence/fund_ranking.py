"""Deterministic fund scoring from verified reasoning observations."""
from __future__ import annotations

from dataclasses import dataclass
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType

@dataclass(frozen=True)
class RankedFund:
    scheme_code: str
    scheme_name: str
    category: str
    score: float
    return_1y: float | None
    return_3y_cagr: float | None
    sharpe_ratio: float | None
    volatility: float | None
    max_drawdown: float | None
    peer_percentile: float | None

class FundRanker:
    _WEIGHTS = {
        "return_3y_cagr": 0.30,
        "return_1y": 0.15,
        "sharpe_ratio": 0.20,
        "max_drawdown": 0.15,
        "volatility": 0.10,
        "peer_percentile": 0.10,
    }

    def rank(self, state: ReasoningState, *, limit: int = 10) -> tuple[RankedFund, ...]:
        discovered = {}
        perf = {}
        risk = {}
        peer = {}

        for observation in state.observations:
            if observation.status not in (ActionStatus.SUCCEEDED, ActionStatus.PARTIAL):
                continue
            result = observation.result if isinstance(observation.result, dict) else {}

            if observation.action_type is ActionType.DISCOVER_FUNDS:
                for fund in result.get("funds", []):
                    if isinstance(fund, dict) and fund.get("scheme_code"):
                        discovered[str(fund["scheme_code"])] = fund

            elif observation.action_type is ActionType.FETCH_PERFORMANCE:
                rows = result.get("rows", [])
                if result.get("scope") == "scheme":
                    rows = [result]
                for row in rows:
                    code = str(row.get("scheme_code", ""))
                    if code:
                        perf[code] = row.get("metrics", {})

            elif observation.action_type is ActionType.COMPUTE_RISK:
                rows = result.get("rows", [])
                if result.get("scope") == "scheme":
                    rows = [result]
                for row in rows:
                    code = str(row.get("scheme_code", ""))
                    if code:
                        risk[code] = row.get("metrics", {})

            elif observation.action_type is ActionType.COMPARE_PEERS:
                groups = result.get("categories", [])
                if result.get("scope") == "scheme_peer_set":
                    groups = [result]
                for group in groups:
                    for row in group.get("peers", []):
                        code = str(row.get("scheme_code", ""))
                        if code:
                            peer[code] = row.get("percentile_rank")

        codes = [code for code in discovered if code in perf and code in risk]
        if not codes:
            return ()

        metrics = {
            "return_3y_cagr": {c: self._num(perf[c].get("return_3y_cagr")) for c in codes},
            "return_1y": {c: self._num(perf[c].get("return_1y")) for c in codes},
            "sharpe_ratio": {c: self._num(risk[c].get("sharpe_ratio")) for c in codes},
            "max_drawdown": {c: self._num(risk[c].get("max_drawdown")) for c in codes},
            "volatility": {c: self._num(risk[c].get("volatility")) for c in codes},
            "peer_percentile": {c: self._num(peer.get(c)) for c in codes},
        }

        scored: list[RankedFund] = []
        for code in codes:
            score = 0.0
            score += self._WEIGHTS["return_3y_cagr"] * self._normalize(metrics["return_3y_cagr"], code, high_good=True)
            score += self._WEIGHTS["return_1y"] * self._normalize(metrics["return_1y"], code, high_good=True)
            score += self._WEIGHTS["sharpe_ratio"] * self._normalize(metrics["sharpe_ratio"], code, high_good=True)
            score += self._WEIGHTS["max_drawdown"] * self._normalize(metrics["max_drawdown"], code, high_good=False)
            score += self._WEIGHTS["volatility"] * self._normalize(metrics["volatility"], code, high_good=False)
            score += self._WEIGHTS["peer_percentile"] * self._normalize(metrics["peer_percentile"], code, high_good=True)
            fund = discovered[code]
            scored.append(RankedFund(
                scheme_code=code,
                scheme_name=str(fund.get("scheme_name") or code),
                category=str(fund.get("category") or ""),
                score=round(score * 100, 2),
                return_1y=metrics["return_1y"][code],
                return_3y_cagr=metrics["return_3y_cagr"][code],
                sharpe_ratio=metrics["sharpe_ratio"][code],
                volatility=metrics["volatility"][code],
                max_drawdown=metrics["max_drawdown"][code],
                peer_percentile=metrics["peer_percentile"][code],
            ))

        scored.sort(key=lambda fund: fund.score, reverse=True)
        return tuple(scored[:limit])

    @staticmethod
    def _num(value):
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _normalize(values: dict[str, float | None], code: str, *, high_good: bool) -> float:
        available = [value for value in values.values() if value is not None]
        value = values.get(code)
        if value is None or not available:
            return 0.0
        low, high = min(available), max(available)
        if high == low:
            return 1.0
        normalized = (value - low) / (high - low)
        return normalized if high_good else 1.0 - normalized
