"""Edge functions for the analytical agent subgraph.

is_range_query   : after analyze_query → plan_years | route
after_extract    : after extract_metric → retrieve_year (loop) | synthesize (done)
"""
from __future__ import annotations

from financial_pipeline.graph.state import RAGState


def is_range_query(state: RAGState) -> str:
    """Route after analyze_query.

    year_from + year_to both set → plan_years  (analytical agent loop)
    otherwise                    → route        (existing parallel path)
    """
    intent    = state.get("intent")
    year_from = getattr(intent, "year_from", None)
    year_to   = getattr(intent, "year_to", None)

    if year_from and year_to:
        return "plan_years"
    return "route"


def after_extract(state: RAGState) -> str:
    """Route after extract_metric.

    pending_years non-empty → retrieve_year  (process next year)
    pending_years empty     → synthesize      (all years done)
    """
    if state.get("pending_years"):
        return "retrieve_year"
    return "synthesize"
