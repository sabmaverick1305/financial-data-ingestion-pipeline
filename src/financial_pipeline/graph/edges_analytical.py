"""Edge functions for the analytical agent subgraph.

is_range_query       : after analyze_query → plan_years | route
after_plan_years     : after plan_years → query_db_stats | plan_months
after_extract        : (legacy) after extract_metric → retrieve_year | synthesize
after_extract_month  : after extract_month_metric → retrieve_month | aggregate_year
after_aggregate_year : after aggregate_year → plan_months (next year) | synthesize (done)
"""
from __future__ import annotations

from financial_pipeline.graph.nodes_fund_performance import FUND_PERFORMANCE_METRIC_IDS
from financial_pipeline.graph.state import RAGState


def is_range_query(state: RAGState) -> str:
    """Route after analyze_query.

    blocked=True                    → format_response (early policy block — advice/OOS)
    fund-performance metric resolved → fund_performance_sql (per-scheme NAV/returns)
    intent.needs_reasoning=True     → reasoning   (causal "why" queries)
    intent.intent_type == "tabular" → query_sql   (Vanna text-to-SQL path)
    intent.needs_analytical=True    → plan_years  (analytical agent loop)
    year_from + year_to set         → plan_years  (backward-compat fallback)
    otherwise                       → route       (standard parallel path)
    """
    if state.get("blocked"):
        return "format_response"

    intent = state.get("intent")

    # Checked first, before needs_reasoning/needs_analytical: a query about a
    # specific scheme's NAV/return/CAGR could otherwise also carry a year
    # range or a resolved metric and get caught by plan_years/query_sql below,
    # neither of which know mf_scheme_master/mf_nav_history/mf_scheme_performance
    # exist — they only query AMFI's aggregate fund_category/AMC tables.
    canonical_metrics = set(getattr(getattr(intent, "canonical", None), "metrics", None) or [])
    if canonical_metrics & FUND_PERFORMANCE_METRIC_IDS:
        return "fund_performance_sql"

    # Checked next: a causal "why did X change" query would otherwise be
    # caught by needs_analytical/tabular below (it usually also has a year
    # range and a resolved metric) and sent through plan_years/query_sql,
    # which can only report numbers — not explain them.
    if getattr(intent, "needs_reasoning", False):
        return "reasoning"

    # Primary check: explicit flag set by the 5-stage intent pipeline
    if getattr(intent, "needs_analytical", False):
        return "plan_years"

    # Tabular / trend intent: structured numeric queries → Vanna SQL
    if getattr(intent, "intent_type", None) in ("tabular", "trend"):
        return "query_sql"

    # Backward-compat: year range present but flag not set (e.g. old state)
    year_from = getattr(intent, "year_from", None)
    year_to   = getattr(intent, "year_to", None)
    if year_from and year_to:
        return "plan_years"

    return "route"


def after_plan_years(state: RAGState) -> str:
    """Route after plan_years.

    If the metric maps to a structured DB table for all requested years,
    go to query_db_stats (single SQL, no LLM loop).
    Otherwise fall through to plan_months (chunk-based LLM extraction).

    Pre-2020 note: amfi_amc_stats uses broad scheme types (Equity, ELSS, Income…),
    not fund categories (Large Cap, Mid Cap…). Route to query_db_stats only when
    the target scheme has a known pre-2020 mapping.
    """
    from financial_pipeline.graph.nodes_analytical import (
        _FUND_STATS_COLS, _AMC_STATS_COLS, _PRE2020_SCHEME_MAP,
    )

    pending = state.get("pending_years") or []
    metric  = state.get("extraction_metric", "funds_mobilized")
    scheme  = (state.get("target_scheme") or "").lower().strip()

    if not pending:
        return "route"

    year_from = min(pending)
    year_to   = max(pending)
    all_post  = year_from >= 2020
    all_pre   = year_to   <  2020

    if all_post and metric in _FUND_STATS_COLS:
        return "query_db_stats"

    # Pre-2020: always try the DB path — either the scheme maps to a known
    # amfi_amc_stats scheme_type (returns data) or it doesn't (returns empty
    # instantly, far cheaper than running 12×N LLM extraction calls).
    if all_pre and metric in _AMC_STATS_COLS:
        return "query_db_stats"

    return "plan_months"


def after_extract(state: RAGState) -> str:
    """(Legacy — single-year snapshot path) Route after extract_metric.

    pending_years non-empty → retrieve_year  (process next year)
    pending_years empty     → synthesize      (all years done)
    """
    if state.get("pending_years"):
        return "retrieve_year"
    return "synthesize"


def after_extract_month(state: RAGState) -> str:
    """Route after extract_month_metric (Option A inner loop).

    pending_months non-empty → retrieve_month   (process next month)
    pending_months empty     → aggregate_year   (all months done for this year)
    """
    if state.get("pending_months"):
        return "retrieve_month"
    return "aggregate_year"


def after_aggregate_year(state: RAGState) -> str:
    """Route after aggregate_year (Option A outer loop).

    pending_years non-empty → plan_months   (start month loop for next year)
    pending_years empty     → synthesize    (all years done)
    """
    if state.get("pending_years"):
        return "plan_months"
    return "synthesize"
