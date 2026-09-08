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
    "total return 1yr": "return_1y",
    "total return 1 year": "return_1y",
    "one year return": "return_1y",
    "1 year performance": "return_1y",
    "annual return": "return_1y",
    "3 year return": "return_3y_cagr",
    "3 yr return": "return_3y_cagr",
    "3y return": "return_3y_cagr",
    "3yr return": "return_3y_cagr",
    "total return 3yr": "return_3y_cagr",
    "total return 3 year": "return_3y_cagr",
    "three year return": "return_3y_cagr",
    "3 year cagr": "return_3y_cagr",
    "3yr cagr": "return_3y_cagr",
    "3 year annualized return": "return_3y_cagr",
    "5 year return": "return_5y_cagr",
    "5 yr return": "return_5y_cagr",
    "5y return": "return_5y_cagr",
    "5yr return": "return_5y_cagr",
    "total return 5yr": "return_5y_cagr",
    "total return 5 year": "return_5y_cagr",
    "five year return": "return_5y_cagr",
    "5 year cagr": "return_5y_cagr",
    "5yr cagr": "return_5y_cagr",
    "5 year annualized return": "return_5y_cagr",
    "10 year return": "return_10y_cagr",
    "10 yr return": "return_10y_cagr",
    "10y return": "return_10y_cagr",
    "10yr return": "return_10y_cagr",
    "total return 10yr": "return_10y_cagr",
    "total return 10 year": "return_10y_cagr",
    "ten year return": "return_10y_cagr",
    "10 year cagr": "return_10y_cagr",
    "10yr cagr": "return_10y_cagr",
    "10 year annualized return": "return_10y_cagr",
    "ytd return": "ytd_return",
    "year to date return": "ytd_return",
    "year to date performance": "ytd_return",

    # Risk
    "sharpe ratio": "sharpe_ratio",
    "max drawdown": "max_drawdown",
    "maximum drawdown": "max_drawdown",
    "standard deviation": "volatility",
    "stddev": "volatility",
    "rolling volatility": "volatility",
    "annualized volatility": "volatility",
    "return volatility": "volatility",
    "standard deviation of returns": "volatility",
    "sharpe": "sharpe_ratio",
    "risk adjusted return": "sharpe_ratio",
    "drawdown": "max_drawdown",
    "downside drawdown": "max_drawdown",

    # Flows / AUM
    "net inflow": "net_inflow",
    "net inflows": "net_inflow",
    "net flows": "net_inflow",
    "inflow outflow": "net_inflow",
    "flow trend": "flow_trend",
    "net fund flow": "net_inflow",
    "net fund flows": "net_inflow",
    "fund inflow": "net_inflow",
    "fund inflows": "net_inflow",
    "inflows": "net_inflow",
    "inflow trend": "flow_trend",
    "fund flow trend": "flow_trend",
    "total aum": "aum",
    "assets under management": "aum",
    "aum trend": "aum_trend",
    "fund size": "aum",
    "asset size": "aum",
    "total assets": "aum",
    "assets under management trend": "aum_trend",
    "aum growth": "aum_trend",

    # Peer comparison
    "return ranking": "percentile_rank",
    "risk ranking": "percentile_rank",
    "sharpe ratio ranking": "percentile_rank",
    "relative performance": "peer_outperformance",
    "relative risk": "percentile_rank",
    "peer outperformance": "peer_outperformance",
    "percentile rank": "percentile_rank",
    "peer rank": "percentile_rank",
    "peer ranking": "percentile_rank",
    "category rank": "percentile_rank",
    "category ranking": "percentile_rank",
    "performance rank": "percentile_rank",
    "performance ranking": "percentile_rank",
    "relative rank": "percentile_rank",
    "performance vs benchmark": "peer_outperformance",
    "performance versus benchmark": "peer_outperformance",
    "benchmark outperformance": "peer_outperformance",
    "excess return": "peer_outperformance",
    "relative return": "peer_outperformance",
    "peer performance": "peer_outperformance",
    "performance vs peers": "peer_outperformance",

    # Documentary / quality
    "expense ratio": "expense_ratio",
    "fund manager tenure": "fund_manager_tenure",
    "portfolio concentration": "portfolio_concentration",
    "management tenure": "fund_manager_tenure",
    "manager tenure": "fund_manager_tenure",
    "fund manager experience": "fund_manager_tenure",
    "ter": "expense_ratio",
    "total expense ratio": "expense_ratio",
    "management expense ratio": "expense_ratio",
    "portfolio concentration ratio": "portfolio_concentration",
    "top holdings concentration": "portfolio_concentration",
    "top 10 concentration": "portfolio_concentration",
    "holding concentration": "portfolio_concentration",
    "return percentile": "percentile_rank",
    "risk percentile": "percentile_rank",
    "sharpe percentile": "percentile_rank",
    "performance percentile": "percentile_rank",
    "asset size top quartile": "asset_size_top_quartile",
    "aum top quartile": "asset_size_top_quartile",
    "expense ratio below median": "expense_ratio_below_median",
    "below median expense ratio": "expense_ratio_below_median",
    "net inflows 1yr": "net_inflow",
    "net flows 1yr": "net_inflow",
    "net flows ytd": "net_inflow",
    "current aum": "aum",
    "latest aum": "aum",
    "aum growth rate": "aum_growth_rate",
    "aum growth percent": "aum_growth_rate",
    "downside capture": "downside_capture",
    "downside capture ratio": "downside_capture",
    "redemption rate": "redemption_rate",
    "relative flows": "relative_flows",
    "flows vs peers": "relative_flows",
    "std deviation": "volatility",
    "std deviation returns": "volatility",
    "fund age": "fund_age",
    "return 1yr": "return_1y",
    "return 3yr": "return_3y_cagr",
    "return 5yr": "return_5y_cagr",
    "return 10yr": "return_10y_cagr",
    "aum current": "aum",
    "inflow 1yr": "net_inflow",
    "return rank": "percentile_rank",
    "risk rank": "percentile_rank",
    "sharpe rank": "percentile_rank",
}


class MetricOntology:
    def canonicalize(self, metric: str) -> str:
        key = _key(metric)
        if key in _ALIASES:
            return _ALIASES[key]
        snake = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
        return snake or metric
