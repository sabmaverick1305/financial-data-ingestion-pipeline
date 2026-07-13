"""Phase 6: Data Quality evaluation metrics — aggregates
eval/runners/run_dataquality_eval.py's per-check results (duplicates,
orphans, relationship completeness, identifier coverage, freshness) the
same way phases 1-5 aggregate their own per-query results."""
from __future__ import annotations
from typing import Any


def overall_pass_rate(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r["passed"]) / len(results)


def duplicate_count_by_type(results: list[dict]) -> dict[str, int]:
    by_id = {r["id"]: r["detail_count"] for r in results}
    return {
        "belongs_to": by_id.get("DQ001", 0),
        "manages": by_id.get("DQ002", 0),
    }


def orphan_count_by_type(results: list[dict]) -> dict[str, int]:
    by_id = {r["id"]: r["detail_count"] for r in results}
    return {
        "scheme_plan": by_id.get("DQ003", 0),
    }


def relationship_completeness_ratio(results: list[dict]) -> float | None:
    for r in results:
        if r["id"] == "DQ006":
            return r["measured_value"]
    return None


def identifier_mapping_coverage_ratio(results: list[dict]) -> float | None:
    for r in results:
        if r["id"] == "DQ007":
            return r["measured_value"]
    return None


def freshness_violation_count(results: list[dict]) -> dict[str, int]:
    by_id = {r["id"]: r["detail_count"] for r in results}
    return {
        "nav": by_id.get("DQ008", 0),
        "sync": by_id.get("DQ009", 0),
    }


def summary(results: list[dict]) -> dict[str, Any]:
    return {
        "overall_pass_rate": overall_pass_rate(results),
        "duplicate_count_by_type": duplicate_count_by_type(results),
        "orphan_count_by_type": orphan_count_by_type(results),
        "relationship_completeness_ratio": relationship_completeness_ratio(results),
        "identifier_mapping_coverage_ratio": identifier_mapping_coverage_ratio(results),
        "freshness_violation_count": freshness_violation_count(results),
    }
