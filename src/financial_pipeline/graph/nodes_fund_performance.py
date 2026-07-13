"""LangGraph node that routes per-scheme fund-performance queries through
Vanna text-to-SQL, targeting mf_scheme_master / mf_nav_history /
mf_scheme_performance — a distinct dataset from amfi_fund_stats (see
nodes_sql.py's query_sql, which targets AMFI's aggregate fund_category/AMC
tables). Same downstream contract as query_sql: pre_guardrail -> generate
(short-circuits on structured_answer) -> post_guardrail.

Routed here instead of query_sql whenever the resolved canonical metrics
include a per-scheme concept — see edges_analytical.py's is_range_query,
which checks FUND_PERFORMANCE_METRIC_IDS before anything else.
"""
from __future__ import annotations

import time
from typing import Any

import structlog

from financial_pipeline.augmentation.citations import Citation
from financial_pipeline.graph.state import RAGState

try:
    from financial_pipeline.text_to_sql.vanna_agent import FinancialVanna, ask as vanna_ask
    _VANNA_IMPORT_ERROR: Exception | None = None
except ImportError as exc:  # pragma: no cover - exercised when optional deps are absent
    FinancialVanna = Any  # type: ignore[assignment]
    vanna_ask = None  # type: ignore[assignment]
    _VANNA_IMPORT_ERROR = exc

log = structlog.get_logger()

# Canonical metric ids (domain/semantic/vocabulary.yaml) that identify a query as
# being about a SPECIFIC scheme's NAV/returns rather than AMFI's aggregate
# fund_category/AMC statistics. Checked first in is_range_query — a fund-
# performance query might otherwise also match needs_analytical/tabular and
# get sent to plan_years/query_sql, neither of which know these tables exist.
FUND_PERFORMANCE_METRIC_IDS = frozenset({
    "scheme_code", "scheme_name", "amc_name",
    "nav", "latest_nav",
    "return_1m", "return_3m", "return_6m", "return_1y",
    "return_3y_cagr", "return_5y_cagr", "since_launch_return",
})


class FundPerformanceNodeFactory:
    """Holds the Vanna agent and exposes the fund_performance_sql LangGraph node."""

    def __init__(self, vanna: FinancialVanna) -> None:
        self._vanna = vanna

    # ── LangGraph node ─────────────────────────────────────────────────────────

    def query_fund_performance(self, state: RAGState) -> dict[str, Any]:
        """Generate SQL via Vanna against the per-scheme tables, validate,
        execute, then feed the augmentation pipeline — identical contract to
        nodes_sql.py's query_sql (see there for the full field-by-field
        rationale); only the citation source label differs.
        """
        if _VANNA_IMPORT_ERROR is not None or vanna_ask is None:
            raise RuntimeError(
                "Text-to-SQL support requires the optional 'vanna' dependency. "
                "Install the SQL extras or disable the fund-performance branch."
            ) from _VANNA_IMPORT_ERROR

        query = state.get("query", "")
        t0 = time.time()

        sql, df, answer, policy_warnings = vanna_ask(self._vanna, query)
        latency_ms = int((time.time() - t0) * 1000)

        blocked = "blocked by safety policy" in (answer or "")

        log.info(
            "fund_performance_node.done",
            query=query[:80],
            sql=sql or "",
            rows=len(df) if df is not None else 0,
            latency_ms=latency_ms,
            policy_warnings=policy_warnings,
            blocked=blocked,
        )

        if df is not None and not df.empty:
            excerpt    = answer
            confidence = 1.0
            answer     = f"{answer}\n\n*Source: [1]*"
        else:
            excerpt    = f"Database query result: {answer or 'No data returned.'}"
            confidence = 0.0

        sql_citation = Citation(
            number=1,
            source_type="sql",
            file_name="Mutual Fund Scheme Database",
            period_year=None,
            period_month=None,
            category=None,
            chunk_index=None,
            table_index=None,
            excerpt=excerpt,
            confidence=confidence,
            rank_method="sql",
        )

        return {
            "citations":       [sql_citation],
            "is_analytical":   True,
            "sql_context":     True,
            "structured_answer": answer,
            "generation_meta": {
                "sql":             sql,
                "policy_warnings": policy_warnings,
                "latency_ms":      latency_ms,
                "model":           "vanna-sql",
                "provider":        "sql",
                "prompt_tokens":   0,
                "completion_tokens": 0,
            },
        }
