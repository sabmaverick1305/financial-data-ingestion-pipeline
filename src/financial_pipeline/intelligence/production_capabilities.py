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

    def discover_categories(self, action: ResearchAction) -> CapabilityResult:
        categories = self._repo().discover_categories()
        return CapabilityResult(
            result={"scope": "scheme_category", "categories": categories},
            evidence_refs=("verified:postgres:mf_scheme_master",),
        )

    def discover_funds(self, action: ResearchAction) -> CapabilityResult:
        category = action.parameters.get("category")
        limit = int(action.parameters.get("limit", 20))
        rows = self._repo().discover_funds(category=category, limit=limit)
        return CapabilityResult(result={"scope": "scheme", "funds": rows}, evidence_refs=("verified:postgres:mf_scheme_master", "verified:postgres:mf_scheme_performance"))

    def performance(self, action: ResearchAction) -> CapabilityResult:
        scheme_codes = action.parameters.get("scheme_codes")
        scheme_code = action.parameters.get("scheme_code")
        if scheme_code and not scheme_codes:
            scheme_codes = [scheme_code]
        if not scheme_codes:
            raise ValueError("scheme_code or scheme_codes is required for scheme performance")

        rows = []
        refs = []
        for code in scheme_codes:
            row = self._repo().performance(str(code))
            if row is None:
                continue
            rows.append({
                "scheme_code": str(code),
                "scheme_name": row.get("scheme_name"),
                "category": row.get("category"),
                "metrics": {metric: row.get(metric) for metric in action.metrics},
            })
            refs.append(f"verified:postgres:mf_scheme_performance:{code}")

        if not rows:
            raise ValueError("no performance rows found for discovered schemes")
        if len(rows) == 1 and scheme_code:
            single = rows[0]
            return CapabilityResult(
                result={
                    "scope": "scheme",
                    "scheme_code": single["scheme_code"],
                    "metrics": single["metrics"],
                },
                evidence_refs=tuple(dict.fromkeys(refs)),
            )
        return CapabilityResult(
            result={"scope": "scheme_batch", "rows": rows},
            evidence_refs=tuple(dict.fromkeys(refs)),
        )

    def returns(self, action: ResearchAction) -> CapabilityResult:
        """Deterministic return capability over persisted scheme performance."""
        return self.performance(action)

    def risk(self, action: ResearchAction) -> CapabilityResult:
        scheme_codes = action.parameters.get("scheme_codes")
        scheme_code = action.parameters.get("scheme_code")
        if scheme_code and not scheme_codes:
            scheme_codes = [scheme_code]
        if not scheme_codes:
            raise ValueError("scheme_code or scheme_codes is required for risk computation")

        rows = []
        refs = []
        for code in scheme_codes:
            history = self._repo().nav_history(str(code))
            if len(history) < 2:
                continue
            returns = [
                (history[i][1] / history[i - 1][1]) - 1.0
                for i in range(1, len(history))
                if history[i - 1][1] > 0
            ]
            if not returns:
                continue
            daily_std = statistics.stdev(returns) if len(returns) > 1 else 0.0
            volatility = daily_std * math.sqrt(252) * 100
            annualized_return = statistics.mean(returns) * 252
            sharpe = (
                (annualized_return - self.risk_free_rate) / (daily_std * math.sqrt(252))
                if daily_std else None
            )
            peak = history[0][1]
            max_drawdown = 0.0
            for _, nav in history:
                peak = max(peak, nav)
                if peak:
                    max_drawdown = max(max_drawdown, (peak - nav) / peak)
            values = {
                "volatility": volatility,
                "sharpe_ratio": sharpe,
                "max_drawdown": max_drawdown * 100,
            }
            rows.append({
                "scheme_code": str(code),
                "metrics": {metric: values.get(metric) for metric in action.metrics},
            })
            refs.append(f"verified:postgres:mf_nav_history:{code}")

        if not rows:
            raise ValueError("no schemes had sufficient NAV history for risk computation")
        if len(rows) == 1 and scheme_code:
            single = rows[0]
            return CapabilityResult(
                result={
                    "scope": "scheme",
                    "scheme_code": single["scheme_code"],
                    "metrics": single["metrics"],
                },
                evidence_refs=tuple(dict.fromkeys(refs)),
            )
        return CapabilityResult(
            result={"scope": "scheme_batch", "rows": rows},
            evidence_refs=tuple(dict.fromkeys(refs)),
        )

    def peer_compare(self, action: ResearchAction) -> CapabilityResult:
        categories = action.parameters.get("categories")
        category = action.parameters.get("category")
        if category and not categories:
            categories = [category]
        if not categories:
            raise ValueError("category or categories is required for peer comparison")

        metric = str(
            action.parameters.get("compare_metric")
            or action.parameters.get("rank_by")
            or "return_3y_cagr"
        )
        if metric not in {
            "return_1y",
            "return_3y_cagr",
            "return_5y_cagr",
            "return_10y_cagr",
            "rolling_volatility",
        }:
            metric = "return_3y_cagr"

        groups = []
        for current_category in categories:
            peers = self._repo().peer_performance(
                category=str(current_category),
                limit=int(action.parameters.get("peer_limit", 100)),
            )
            ranked = [dict(row) for row in peers if row.get(metric) is not None]
            ranked.sort(
                key=lambda row: float(row[metric]),
                reverse=metric != "rolling_volatility",
            )
            total = len(ranked)
            for index, row in enumerate(ranked, 1):
                row["rank"] = index
                row["percentile_rank"] = (
                    ((total - index) / max(1, total - 1)) * 100
                    if total > 1 else 100.0
                )
                row["peer_outperformance"] = index <= max(1, math.ceil(total * 0.25))
            groups.append({
                "category": str(current_category),
                "metric": metric,
                "peers": ranked,
            })

        if len(groups) == 1 and category:
            group = groups[0]
            return CapabilityResult(
                result={
                    "scope": "scheme_peer_set",
                    "category": group["category"],
                    "metric": group["metric"],
                    "peers": group["peers"],
                },
                evidence_refs=(
                    "verified:postgres:mf_scheme_master",
                    "verified:postgres:mf_scheme_performance",
                ),
            )
        return CapabilityResult(
            result={"scope": "scheme_peer_sets", "categories": groups},
            evidence_refs=(
                "verified:postgres:mf_scheme_master",
                "verified:postgres:mf_scheme_performance",
            ),
        )

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

    def contradictions(self, action: ResearchAction) -> CapabilityResult:
        from financial_pipeline.intelligence.contradictions import ContradictionEngine
        facts = dict(action.parameters.get("facts") or {})
        contradictions = ContradictionEngine().evaluate(facts)
        return CapabilityResult(
            result={
                "scope": "deterministic_validation",
                "checks": list(action.checks),
                "contradictions": [
                    {"code": item.code, "severity": item.severity, "reason": item.reason}
                    for item in contradictions
                ],
            },
            evidence_refs=("verified:deterministic:contradiction_rules",),
        )

    def documentary(self, action: ResearchAction) -> CapabilityResult:
        if self.rag_pipeline is None:
            raise RuntimeError("RAG pipeline is not configured")
        query = action.parameters.get("query") or " ".join((*action.metrics, *action.evidence_types))
        response = self.rag_pipeline.ask(str(query))
        refs = tuple(f"verified:rag:{source.get('document_id', source.get('source', index))}" for index, source in enumerate(response.sources))
        return CapabilityResult(result={"scope": "document", "answer": response.answer, "sources": response.sources}, evidence_refs=refs)
