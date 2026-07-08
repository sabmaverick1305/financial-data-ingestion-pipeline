"""Phase 2: SQL Quality evaluation metrics."""
from __future__ import annotations
import re
from typing import Any


def parse_rate(results: list[dict]) -> float:
    """Fraction of queries where Vanna returned parseable SQL (not None/empty)."""
    if not results:
        return 0.0
    parsed = sum(1 for r in results if r.get("sql") and r["sql"].strip())
    return parsed / len(results)


def policy_pass_rate(results: list[dict]) -> float:
    """Fraction of queries that passed all policy checks (no hard blocks)."""
    if not results:
        return 0.0
    passed = sum(1 for r in results if not r.get("hard_blocked", False))
    return passed / len(results)


def table_accuracy(results: list[dict], labels: list[dict]) -> float:
    """Fraction where generated SQL references the expected table."""
    correct = 0
    total = 0
    for r, l in zip(results, labels):
        expected_table = l.get("expected_table")
        if not expected_table:
            continue
        total += 1
        sql = r.get("sql", "") or ""
        if re.search(rf"\b{re.escape(expected_table)}\b", sql, re.IGNORECASE):
            correct += 1
    return correct / total if total else 0.0


def column_recall(results: list[dict], labels: list[dict]) -> float:
    """Average fraction of expected columns present in generated SQL."""
    scores = []
    for r, l in zip(results, labels):
        expected_cols = l.get("expected_columns", [])
        if not expected_cols:
            continue
        sql = r.get("sql", "") or ""
        hits = sum(1 for col in expected_cols if re.search(rf"\b{re.escape(col)}\b", sql, re.IGNORECASE))
        scores.append(hits / len(expected_cols))
    return sum(scores) / len(scores) if scores else 0.0


def execution_accuracy(results: list[dict], expected: list[dict], tolerance_pct: float = 2.0) -> float:
    """Fraction of queries where DB result matches expected_results ranges.

    Matching is value-based, not column-name-based, because Vanna may generate
    different column aliases across runs.  For each expected key-value entry we
    search every numeric cell across all rows; if we find a value in range we
    count it as matched.  All kv entries must match for a query to pass.
    """
    correct = 0
    total = 0
    for r, e in zip(results, expected):
        if not e or not e.get("seeded", False):
            continue
        key_values = e.get("key_values", [])
        if not key_values:
            continue
        total += 1
        df = r.get("df")
        if df is None or len(df) == 0:
            continue

        # Collect all numeric values in the dataframe (all rows × all columns)
        all_values: list[float] = []
        for col in df.columns:
            for raw in df[col]:
                try:
                    all_values.append(float(raw))
                except (TypeError, ValueError):
                    pass

        all_match = True
        for kv in key_values:
            low  = kv.get("min", float("-inf"))
            high = kv.get("max", float("inf"))
            # Ensure lo ≤ hi (handles negatives seeded before the min/max fix)
            lo, hi = min(low, high), max(low, high)
            # Absolute margin so negatives work: margin = X% of the span
            margin = (hi - lo) * tolerance_pct / 100
            found = any((lo - margin) <= v <= (hi + margin) for v in all_values)
            if not found:
                all_match = False
                break

        if all_match:
            correct += 1
    return correct / total if total else 0.0


def row_count_accuracy(results: list[dict], expected: list[dict]) -> float:
    """Fraction of queries where result row count matches expected."""
    correct = 0
    total = 0
    for r, e in zip(results, expected):
        if not e.get("seeded", False):
            continue
        df = r.get("df")
        if df is None:
            continue
        total += 1
        actual_rows = len(df)
        if "row_count" in e:
            if actual_rows == e["row_count"]:
                correct += 1
        elif "row_count_min" in e and "row_count_max" in e:
            if e["row_count_min"] <= actual_rows <= e["row_count_max"]:
                correct += 1
    return correct / total if total else 0.0


def policy_check_rate(results: list[dict], labels: list[dict]) -> dict[str, float]:
    """Per-check pass rates: select_only, approved_tables, limit_present, date_filter."""
    checks = {"select_only": 0, "approved_tables": 0, "limit_present": 0, "date_filter": 0}
    totals = {k: 0 for k in checks}
    for r, l in zip(results, labels):
        policy = l.get("policy_checks", {})
        sql = r.get("sql", "") or ""
        if policy.get("must_be_select"):
            totals["select_only"] += 1
            if re.match(r"^\s*SELECT\b", sql, re.IGNORECASE):
                checks["select_only"] += 1
        if policy.get("approved_tables_only"):
            totals["approved_tables"] += 1
            if not r.get("hard_blocked", False):
                checks["approved_tables"] += 1
        if policy.get("limit_enforced"):
            totals["limit_present"] += 1
            if re.search(r"\bLIMIT\b", sql, re.IGNORECASE):
                checks["limit_present"] += 1
        if policy.get("date_filter_required"):
            totals["date_filter"] += 1
            if re.search(r"\b(period_year|period_month)\b", sql, re.IGNORECASE):
                checks["date_filter"] += 1
    return {k: (checks[k] / totals[k] if totals[k] else None) for k in checks}


def summary(results: list[dict], labels: list[dict], expected: list[dict]) -> dict[str, Any]:
    return {
        "parse_rate": parse_rate(results),
        "policy_pass_rate": policy_pass_rate(results),
        "table_accuracy": table_accuracy(results, labels),
        "column_recall": column_recall(results, labels),
        "execution_accuracy": execution_accuracy(results, expected),
        "row_count_accuracy": row_count_accuracy(results, expected),
        "policy_check_rates": policy_check_rate(results, labels),
    }
