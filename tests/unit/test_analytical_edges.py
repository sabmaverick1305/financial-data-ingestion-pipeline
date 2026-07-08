from __future__ import annotations

from types import SimpleNamespace

from financial_pipeline.graph.edges_analytical import after_plan_years
from financial_pipeline.graph.edges_analytical import is_range_query


def test_after_plan_years_falls_back_to_route_without_pending_years() -> None:
    assert after_plan_years({}) == "route"


def test_year_range_queries_prefer_analytical_path_over_sql() -> None:
    intent = SimpleNamespace(intent_type="tabular", needs_analytical=True)

    assert is_range_query({"intent": intent}) == "plan_years"
