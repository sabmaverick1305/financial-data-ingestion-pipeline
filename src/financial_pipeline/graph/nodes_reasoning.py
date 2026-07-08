"""LangGraph node for causal "why" queries.

Reuses reasoning_engine.py + reasoning_data.py to answer questions like
"Why did AUM increase while net inflow decreased for Large Cap funds between
2020 and 2024?" with a real, data-grounded explanation, instead of the prior
behavior — abstaining, because AMFI source documents contain no explanatory
prose (fies/ontology/templates.yaml's T_REASON_001 was marked
`status: unimplemented` for exactly this reason).

Follows nodes_sql.py's query_sql pattern: package the result as a synthetic
Citation feeding pre_guardrail -> generate -> post_guardrail, with
structured_answer set so generate() short-circuits — the explanation is
already deterministically rendered from reasoning_rules.yaml, no LLM call
needed.
"""
from __future__ import annotations

import structlog

from financial_pipeline.augmentation.citations import Citation
from financial_pipeline.graph.state import RAGState
from financial_pipeline.reasoning.reasoning_data import fetch_observations
from financial_pipeline.reasoning.reasoning_engine import evaluate

log = structlog.get_logger()

# Always fetch this full set (not just the metric(s) literally named in the
# query) so reasoning_rules.yaml's multi-metric conditions (e.g.
# RULE_ALIGNED_GROWTH needs funds_mobilized + net_inflow + aum together) have
# a chance to match even when the query only names one or two of them.
_DEFAULT_METRICS = ["aum", "net_inflow", "funds_mobilized", "redemption", "folios"]

# No explicit year range in the query -> default lookback window.
_DEFAULT_LOOKBACK_YEARS = 5
_LATEST_KNOWN_YEAR = 2026


class ReasoningNodeFactory:
    """Holds the repo needed to fetch trend data for causal reasoning."""

    def __init__(self, repo) -> None:
        self._repo = repo

    def reasoning_node(self, state: RAGState) -> dict:
        intent = state.get("intent")
        scheme_types = getattr(intent, "scheme_types", None) or []
        scheme = scheme_types[0] if scheme_types else "all"

        canonical = getattr(intent, "canonical", None)
        canonical_metrics = list(canonical.metrics) if canonical else []
        metric_ids = canonical_metrics + [m for m in _DEFAULT_METRICS if m not in canonical_metrics]

        year_to   = getattr(intent, "year_to", None) or getattr(intent, "year", None) or _LATEST_KNOWN_YEAR
        year_from = getattr(intent, "year_from", None) or max(year_to - _DEFAULT_LOOKBACK_YEARS, 2020)

        observations = fetch_observations(self._repo, scheme, metric_ids, year_from, year_to)

        if not observations:
            excerpt = (
                f"No sufficient data was found for {scheme} funds between "
                f"{year_from} and {year_to} to compute a causal explanation."
            )
            confidence = 0.0
            rule_id = None
            answer = excerpt
        else:
            result = evaluate(observations)
            detail_lines = [
                f"- {mid}: {obs.start_label}={obs.start_value:,.1f} -> "
                f"{obs.end_label}={obs.end_value:,.1f} ({obs.pct_change:+.1f}%)"
                for mid, obs in observations.items()
            ]
            excerpt = result.explanation + "\n\nUnderlying data:\n" + "\n".join(detail_lines)
            confidence = 0.85 if result.matched else 0.4
            rule_id = result.rule_id
            answer = f"{excerpt}\n\n*Source: [1]*"

        citation = Citation(
            number=1,
            source_type="sql",
            file_name="AMFI Database (derived)",
            period_year=year_to,
            period_month=None,
            category=None,
            chunk_index=None,
            table_index=None,
            excerpt=excerpt,
            confidence=confidence,
            rank_method="reasoning_engine",
        )

        log.info("node.reasoning", scheme=scheme, metrics=list(observations.keys()),
                 year_from=year_from, year_to=year_to, matched=bool(rule_id), rule_id=rule_id)

        return {
            "citations":         [citation],
            "is_analytical":     True,   # bypass number_consistent check in post_guardrail — numbers come from the DB
            "structured_answer": answer,
            "generation_meta": {
                "model":             "reasoning-engine",
                "provider":          "rules",
                "prompt_tokens":     0,
                "completion_tokens": 0,
                "rule_id":           rule_id,
            },
        }
