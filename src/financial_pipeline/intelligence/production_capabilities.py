"""Production reasoning capabilities backed by FIES Postgres and RAG."""
from __future__ import annotations
import math
import statistics
from typing import Any
from financial_pipeline.intelligence.capability_registry import CapabilityResult
from financial_pipeline.intelligence.research_plan import ResearchAction

class ProductionCapabilityPack:
    def __init__(self, *, repository=None, rag_pipeline=None, risk_free_rate: float = 0.065) -> None:
        self.repository = repository
        self.rag_pipeline = rag_pipeline
        self.risk_free_rate = risk_free_rate

    def _repo(self):
        if self.repository is None:
            raise RuntimeError("production reasoning repository is not configured")
        return self.repository

    def discover_funds(self, action: ResearchAction) -> CapabilityResult:
        category = action.parameters.get("category")
        limit = int(action.parameters.get("limit", 20))
        rows = self._repo().discover_funds(category=category, limit=limit)
        return CapabilityResult(result={"scope": "scheme", "funds": rows}, evidence_refs=("verified:postgres:mf_scheme_master", "verified:postgres:mf_scheme_performance"))

    def performance(self, action: ResearchAction) -> CapabilityResult:
        scheme_code = action.parameters.get("scheme_code")
        if not scheme_code:
            raise ValueError("scheme_code is required for scheme performance")
        row = self._repo().performance(str(scheme_code))
        if row is None:
            raise ValueError(f"no performance row for scheme {scheme_code}")
        result = {metric: row.get(metric) for metric in action.metrics}
        return CapabilityResult(result={"scope": "scheme", "scheme_code": str(scheme_code), "metrics": result}, evidence_refs=(f"verified:postgres:mf_scheme_performance:{scheme_code}",))

    def risk(self, action: ResearchAction) -> CapabilityResult:
        scheme_code = action.parameters.get("scheme_code")
        if not scheme_code:
            raise ValueError("scheme_code is required for risk computation")
        history = self._repo().nav_history(str(scheme_code))
        if len(history) < 2:
            raise ValueError(f"insufficient NAV history for scheme {scheme_code}")
        returns = [(history[i][1] / history[i-1][1]) - 1.0 for i in range(1, len(history)) if history[i-1][1] > 0]
        if not returns:
            raise ValueError(f"invalid NAV history for scheme {scheme_code}")
        daily_std = statistics.stdev(returns) if len(returns) > 1 else 0.0
        volatility = daily_std * math.sqrt(252) * 100
        annualized_return = statistics.mean(returns) * 252
        sharpe = ((annualized_return - self.risk_free_rate) / (daily_std * math.sqrt(252))) if daily_std else None
        peak = history[0][1]
        max_drawdown = 0.0
        for _, nav in history:
            peak = max(peak, nav)
            if peak:
                max_drawdown = max(max_drawdown, (peak - nav) / peak)
        values = {"volatility": volatility, "sharpe_ratio": sharpe, "max_drawdown": max_drawdown * 100}
        requested = {metric: values.get(metric) for metric in action.metrics}
        return CapabilityResult(result={"scope": "scheme", "scheme_code": str(scheme_code), "metrics": requested}, evidence_refs=(f"verified:postgres:mf_nav_history:{scheme_code}",))

    def peer_compare(self, action: ResearchAction) -> CapabilityResult:
        category = action.parameters.get("category")
        if not category:
            raise ValueError("category is required for peer comparison")
        peers = self._repo().peer_performance(category=str(category), limit=int(action.parameters.get("peer_limit", 100)))
        metric = str(action.parameters.get("compare_metric", "return_3y_cagr"))
        ranked = [p for p in peers if p.get(metric) is not None]
        ranked.sort(key=lambda row: float(row[metric]), reverse=True)
        total = len(ranked)
        for index, row in enumerate(ranked, 1):
            row["rank"] = index
            row["percentile_rank"] = ((total - index) / max(1, total - 1)) * 100 if total > 1 else 100.0
        return CapabilityResult(result={"scope": "scheme_peer_set", "category": category, "metric": metric, "peers": ranked}, evidence_refs=("verified:postgres:mf_scheme_master", "verified:postgres:mf_scheme_performance"))

    def aum(self, action: ResearchAction) -> CapabilityResult:
        category = action.parameters.get("category")
        if action.parameters.get("scheme_code"):
            raise ValueError("scheme-level AUM is not present in the current production data contract")
        rows = self._repo().latest_category_facts(metric="aum", category=category)
        refs = tuple(f"verified:amfi:{row.get('source_document_id')}" for row in rows if row.get("source_document_id"))
        return CapabilityResult(result={"scope": "fund_category", "metric": "aum", "rows": rows}, evidence_refs=refs)

    def flows(self, action: ResearchAction) -> CapabilityResult:
        category = action.parameters.get("category")
        if action.parameters.get("scheme_code"):
            raise ValueError("scheme-level flows are not present in the current production data contract")
        metric = "net_inflow"
        rows = self._repo().latest_category_facts(metric=metric, category=category)
        refs = tuple(f"verified:amfi:{row.get('source_document_id')}" for row in rows if row.get("source_document_id"))
        return CapabilityResult(result={"scope": "fund_category", "metric": metric, "rows": rows}, evidence_refs=refs)

    def documentary(self, action: ResearchAction) -> CapabilityResult:
        if self.rag_pipeline is None:
            raise RuntimeError("RAG pipeline is not configured")
        query = action.parameters.get("query") or " ".join((*action.metrics, *action.evidence_types))
        response = self.rag_pipeline.ask(str(query))
        refs = tuple(f"verified:rag:{source.get('document_id', source.get('source', index))}" for index, source in enumerate(response.sources))
        return CapabilityResult(result={"scope": "document", "answer": response.answer, "sources": response.sources}, evidence_refs=refs)
