"""Canonical financial metric vocabulary used between planner and capabilities."""

from __future__ import annotations

import re


def _key(value: str) -> str:
    normalized = value.strip().lower()
    normalized = normalized.replace("%", " percent ")
    normalized = re.sub(r"[-_/]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


_ALIASES: dict[str, str] = {
    # Performance
    "1 year return": "return_1y",
    "1 yr return": "return_1y",
    "1y return": "return_1y",
    "1yr return": "return_1y",
    "return 1y": "return_1y",
    "3 year return": "return_3y_cagr",
    "3 yr return": "return_3y_cagr",
    "3y return": "return_3y_cagr",
    "3yr return": "return_3y_cagr",
    "5 year return": "return_5y_cagr",
    "5 yr return": "return_5y_cagr",
    "5y return": "return_5y_cagr",
    "5yr return": "return_5y_cagr",
    "10 year return": "return_10y_cagr",
    "10 yr return": "return_10y_cagr",
    "10y return": "return_10y_cagr",
    "10yr return": "return_10y_cagr",
    "ytd return": "ytd_return",

    # Risk
    "sharpe ratio": "sharpe_ratio",
    "max drawdown": "max_drawdown",
    "maximum drawdown": "max_drawdown",
    "standard deviation": "volatility",
    "stddev": "volatility",
    "rolling volatility": "volatility",

    # Flows / AUM
    "net inflow": "net_inflow",
    "net inflows": "net_inflow",
    "net flows": "net_inflow",
    "inflow outflow": "net_inflow",
    "flow trend": "flow_trend",
    "total aum": "aum",
    "assets under management": "aum",
    "aum trend": "aum_trend",

    # Peer comparison
    "return ranking": "percentile_rank",
    "risk ranking": "percentile_rank",
    "sharpe ratio ranking": "percentile_rank",
    "relative performance": "peer_outperformance",
    "relative risk": "percentile_rank",
    "peer outperformance": "peer_outperformance",
    "percentile rank": "percentile_rank",

    # Documentary / quality
    "expense ratio": "expense_ratio",
    "fund manager tenure": "fund_manager_tenure",
    "portfolio concentration": "portfolio_concentration",
}


class MetricOntology:
    def canonicalize(self, metric: str) -> str:
        key = _key(metric)
        if key in _ALIASES:
            return _ALIASES[key]
        snake = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
        return snake or metric
