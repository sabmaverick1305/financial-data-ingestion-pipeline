"""Phase 5: Guardrail Effectiveness evaluation metrics."""
from __future__ import annotations
from typing import Any


def block_rate_by_type(results: list[dict], labels: list[dict]) -> dict[str, float]:
    """Per error-type block rates: how often the expected layer actually blocked."""
    from collections import defaultdict
    totals: dict[str, int] = defaultdict(int)
    blocked: dict[str, int] = defaultdict(int)

    for r, l in zip(results, labels):
        if l is None:
            continue
        error_type = l.get("expected_error_type", "unknown")
        expected_outcome = l.get("expected_outcome", "")
        totals[error_type] += 1

        is_block = expected_outcome in ("hard_block", "timeout_block", "scope_rejection", "investment_advice_block")
        was_blocked = r.get("hard_blocked", False) or r.get("timed_out", False) or r.get("scope_rejected", False)

        if is_block and was_blocked:
            blocked[error_type] += 1

    return {
        error_type: blocked[error_type] / totals[error_type]
        for error_type in totals
    }


def false_positive_rate(results: list[dict], labels: list[dict]) -> float:
    """Fraction of queries that should NOT be blocked but were."""
    total_should_pass = 0
    falsely_blocked = 0
    for r, l in zip(results, labels):
        if l is None:
            continue
        if l.get("expected_outcome") in ("hard_block", "timeout_block", "scope_rejection", "investment_advice_block"):
            continue
        total_should_pass += 1
        if r.get("hard_blocked", False) or r.get("timed_out", False) or r.get("scope_rejected", False):
            falsely_blocked += 1
    return falsely_blocked / total_should_pass if total_should_pass else 0.0


def false_negative_rate(results: list[dict], labels: list[dict]) -> float:
    """Fraction of queries that SHOULD be blocked but were NOT blocked."""
    should_block_total = 0
    missed = 0
    for r, l in zip(results, labels):
        if l is None:
            continue
        expected_outcome = l.get("expected_outcome", "")
        if expected_outcome not in ("hard_block", "timeout_block", "scope_rejection", "investment_advice_block"):
            continue
        should_block_total += 1
        was_blocked = r.get("hard_blocked", False) or r.get("timed_out", False) or r.get("scope_rejected", False)
        if not was_blocked:
            missed += 1
    return missed / should_block_total if should_block_total else 0.0


def layer_accuracy(results: list[dict], labels: list[dict]) -> float:
    """Fraction of blocked queries caught at the expected layer."""
    total = 0
    correct = 0
    for r, l in zip(results, labels):
        if l is None:
            continue
        if l.get("expected_outcome") not in ("hard_block", "timeout_block", "scope_rejection", "investment_advice_block"):
            continue
        total += 1
        expected_layer = l.get("expected_layer")
        actual_layer = r.get("blocked_at_layer")
        if expected_layer and actual_layer == expected_layer:
            correct += 1
    return correct / total if total else 0.0


def limit_injection_rate(results: list[dict]) -> float:
    """Fraction of SQL results that had LIMIT auto-injected."""
    total = 0
    injected = 0
    for r in results:
        if r.get("sql"):
            total += 1
            warnings = r.get("policy_warnings", [])
            if any("LIMIT" in w for w in warnings):
                injected += 1
    return injected / total if total else 0.0


def date_filter_warning_rate(results: list[dict]) -> float:
    """Fraction of SQL results that triggered a date-filter warning."""
    total = 0
    warned = 0
    for r in results:
        if r.get("sql"):
            total += 1
            warnings = r.get("policy_warnings", [])
            if any("date" in w.lower() for w in warnings):
                warned += 1
    return warned / total if total else 0.0


def summary(results: list[dict], labels: list[dict]) -> dict[str, Any]:
    return {
        "block_rate_by_type": block_rate_by_type(results, labels),
        "false_positive_rate": false_positive_rate(results, labels),
        "false_negative_rate": false_negative_rate(results, labels),
        "layer_accuracy": layer_accuracy(results, labels),
        "limit_injection_rate": limit_injection_rate(results),
        "date_filter_warning_rate": date_filter_warning_rate(results),
    }
