"""Production reasoning capabilities backed by FIES Postgres and RAG."""
from __future__ import annotations
import math
import statistics
from datetime import timedelta
from typing import Any
from financial_pipeline.intelligence.capability_registry import CapabilityResult
from financial_pipeline.intelligence.research_plan import ActionStatus, ResearchAction
from financial_pipeline.intelligence.documentary_validation import DocumentaryEvidenceValidator
from financial_pipeline.intelligence.scheme_family import SchemeFamilyPolicy

class ProductionCapabilityPack:
    def __init__(self, *, repository=None, rag_pipeline=None, risk_free_rate: float = 0.065) -> None:
        self.repository = repository
        self.rag_pipeline = rag_pipeline
        self.risk_free_rate = risk_free_rate
        self._documentary_validator = DocumentaryEvidenceValidator()
        self._scheme_families = SchemeFamilyPolicy()
        self._nav_snapshot_cache: dict[tuple[str, ...], dict[str, list[tuple]]] = {}

    def _nav_snapshot(self, scheme_codes: list[str]) -> dict[str, list[tuple]]:
        codes = tuple(dict.fromkeys(str(code) for code in scheme_codes))
        if codes in self._nav_snapshot_cache:
            return self._nav_snapshot_cache[codes]
        repo = self._repo()
        if hasattr(repo, "nav_history_many"):
            snapshot = repo.nav_history_many(list(codes))
        else:
            snapshot = {code: repo.nav_history(code) for code in codes}
        self._nav_snapshot_cache[codes] = snapshot
        return snapshot

    def _repo(self):
        if self.repository is None:
            raise RuntimeError("production reasoning repository is not configured")
        return self.repository

    def discover_categories(self, action: ResearchAction) -> CapabilityResult:
        eligible = action.parameters.get("eligible_categories")
        if eligible:
            categories = list(dict.fromkeys(str(value) for value in eligible))
        else:
            categories = self._repo().discover_categories()
        return CapabilityResult(
            result={
                "scope": "scheme_category",
                "categories": categories,
                "mandate": action.parameters.get("mandate"),
            },
            evidence_refs=(
                "verified:postgres:mf_scheme_master",
                "verified:deterministic:investment_mandate",
            ),
        )

    def discover_funds(self, action: ResearchAction) -> CapabilityResult:
        category = action.parameters.get("category") or action.category
        eligible = action.parameters.get("eligible_categories")
        requested_limit = int(action.parameters.get("limit", 20))
        limit = min(requested_limit, 20)

        if category:
            rows = self._repo().discover_funds(category=category, limit=max(limit * 3, limit))
        elif eligible:
            categories = list(dict.fromkeys(str(value) for value in eligible))
            repo = self._repo()
            if hasattr(repo, "discover_funds_many"):
                pooled = repo.discover_funds_many(
                    categories=categories,
                    limit=max(limit * 3, limit),
                )
            else:
                per_category = max(6, math.ceil((limit * 3) / max(1, len(categories))))
                pooled = []
                for eligible_category in categories:
                    pooled.extend(
                        repo.discover_funds(
                            category=eligible_category,
                            limit=per_category,
                        )
                    )
            deduped = {}
            for row in pooled:
                code = str(row.get("scheme_code") or "")
                if code:
                    deduped[code] = row
            rows = self._scheme_families.deduplicate(list(deduped.values()))
            rows.sort(
                key=lambda row: (
                    float(row.get("return_3y_cagr") or float("-inf")),
                    float(row.get("return_1y") or float("-inf")),
                ),
                reverse=True,
            )
            rows = rows[:limit]
        else:
            rows = self._repo().discover_funds(category=None, limit=max(limit * 3, limit))

        rows = self._scheme_families.deduplicate(rows)
        rows.sort(
            key=lambda row: (
                float(row.get("return_3y_cagr") or float("-inf")),
                float(row.get("return_1y") or float("-inf")),
            ),
            reverse=True,
        )
        rows = rows[:limit]

        if not rows:
            raise ValueError("no production fund candidates passed mandate and data-quality gates")

        return CapabilityResult(
            result={
                "scope": "scheme",
                "funds": rows,
                "mandate": action.parameters.get("mandate"),
                "eligible_categories": list(eligible or ([category] if category else [])),
            },
            evidence_refs=(
                "verified:postgres:mf_scheme_master",
                "verified:postgres:mf_scheme_performance",
                "verified:deterministic:investment_mandate",
                "verified:deterministic:data_quality_gate",
            ),
        )

    def performance(self, action: ResearchAction) -> CapabilityResult:
        scheme_codes = action.parameters.get("scheme_codes")
        scheme_code = action.parameters.get("scheme_code")
        if scheme_code and not scheme_codes:
            scheme_codes = [scheme_code]
        if not scheme_codes:
            raise ValueError("scheme_code or scheme_codes is required for scheme performance")

        rows = []
        refs = []
        repo = self._repo()
        if hasattr(repo, "performance_many"):
            performance_map = repo.performance_many([str(code) for code in scheme_codes])
        else:
            performance_map = {
                str(code): repo.performance(str(code))
                for code in scheme_codes
            }
        for code in scheme_codes:
            row = performance_map.get(str(code))
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
        """Deterministically compute total and annualized returns from NAV history."""
        scheme_codes = action.parameters.get("scheme_codes")
        scheme_code = action.parameters.get("scheme_code")
        if scheme_code and not scheme_codes:
            scheme_codes = [scheme_code]
        if not scheme_codes:
            raise ValueError("scheme_code or scheme_codes is required for return computation")

        rows = []
        refs = []
        history_map = self._nav_snapshot([str(code) for code in scheme_codes])
        for code in scheme_codes:
            history = history_map.get(str(code), [])
            if len(history) < 2:
                continue
            end_date, end_nav = history[-1]
            target_start = end_date - timedelta(days=365 * 3)
            candidates = [point for point in history if point[0] >= target_start]
            start_date, start_nav = candidates[0] if candidates else history[0]
            if start_nav <= 0 or end_nav <= 0:
                continue
            days = max(1, (end_date - start_date).days)
            total_return = ((end_nav / start_nav) - 1.0) * 100
            annualized_return = (((end_nav / start_nav) ** (365.0 / days)) - 1.0) * 100
            values = {
                "total_return_3y": total_return,
                "annualized_return": annualized_return,
            }
            rows.append({
                "scheme_code": str(code),
                "start_date": start_date,
                "end_date": end_date,
                "metrics": {metric: values.get(metric) for metric in action.metrics},
            })
            refs.append(f"verified:postgres:mf_nav_history:{code}")

        if not rows:
            raise ValueError("no schemes had sufficient NAV history for return computation")
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

    def risk(self, action: ResearchAction) -> CapabilityResult:
        scheme_codes = action.parameters.get("scheme_codes")
        scheme_code = action.parameters.get("scheme_code")
        if scheme_code and not scheme_codes:
            scheme_codes = [scheme_code]
        if not scheme_codes:
            raise ValueError("scheme_code or scheme_codes is required for risk computation")

        rows = []
        refs = []
        history_map = self._nav_snapshot([str(code) for code in scheme_codes])
        for code in scheme_codes:
            history = history_map.get(str(code), [])
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
        repo = self._repo()
        peer_limit = min(int(action.parameters.get("peer_limit", 50)), 50)
        if hasattr(repo, "peer_performance_many"):
            peer_map = repo.peer_performance_many(
                categories=[str(value) for value in categories],
                limit_per_category=peer_limit,
            )
        else:
            peer_map = {
                str(value): repo.peer_performance(
                    category=str(value),
                    limit=peer_limit,
                )
                for value in categories
            }
        for current_category in categories:
            peers = peer_map.get(str(current_category), [])
            family_peers = self._scheme_families.deduplicate(
                [dict(row) for row in peers]
            )
            ranked = [dict(row) for row in family_peers if row.get(metric) is not None]
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
        candidate_names = list(action.parameters.get("candidate_names") or [])
        document_search_names = list(
            action.parameters.get("document_search_names")
            or candidate_names
        )
        if document_search_names and hasattr(self.rag_pipeline, "ask_documentary"):
            response = self.rag_pipeline.ask_documentary(
                str(query),
                fund_names=document_search_names,
                document_types=list(action.evidence_types),
            )
        else:
            response = self.rag_pipeline.ask(str(query))
        refs = tuple(
            f"verified:rag:{source.get('document_id', source.get('source', index))}"
            for index, source in enumerate(response.sources)
        )
        valid, reason = self._documentary_validator.validate(
            answer=response.answer,
            sources=response.sources,
            requested=action.evidence_types,
        )
        if not response.sources and candidate_names:
            valid = False
            reason = "fund-specific documentary evidence is not indexed in the current corpus"
        return CapabilityResult(
            result={
                "scope": "document",
                "answer": response.answer,
                "sources": response.sources,
                "evidence_valid": valid,
                "candidate_names": candidate_names,
            },
            evidence_refs=refs,
            status=ActionStatus.SUCCEEDED if valid else ActionStatus.PARTIAL,
            tradeoff_reason=reason,
        )
