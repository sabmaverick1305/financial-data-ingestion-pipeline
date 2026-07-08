"""LangGraph node that routes tabular-intent queries through Vanna text-to-SQL.

After executing the SQL query this node hands off to the standard
pre_guardrail → generate → post_guardrail pipeline, but `generate`
short-circuits for structured SQL results so the answer stays exact:
  - Investment-advice pre-check (pre_guardrail)
  - Deterministic pass-through of the SQL result (generate)
  - Citation-validity and safety checks     (post_guardrail)

The SQL table is packaged as a synthetic Citation so PromptBuilder can
include it as passage [1] in the LLM context, enabling the model to
cite its claims correctly.
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


class SQLNodeFactory:
    """Holds the Vanna agent and exposes a LangGraph-compatible node function."""

    def __init__(self, vanna: FinancialVanna) -> None:
        self._vanna = vanna

    # ── LangGraph node ─────────────────────────────────────────────────────────

    def query_sql(self, state: RAGState) -> dict[str, Any]:
        """Generate SQL via Vanna, validate, execute, then feed the augmentation pipeline.

        Returns state keys consumed by pre_guardrail → generate → post_guardrail:
          citations    — synthetic [1] citation whose excerpt is the SQL result table
          is_analytical — True, so post_guardrail skips number_consistent check
                          (SQL numbers come from the DB and are ground truth)
          sql_context  — True, so generate switches to the "sql" system prompt
          generation_meta — carries the raw SQL for LangSmith tracing
        """
        if _VANNA_IMPORT_ERROR is not None or vanna_ask is None:
            raise RuntimeError(
                "Text-to-SQL support requires the optional 'vanna' dependency. "
                "Install the SQL extras or disable the SQL branch."
            ) from _VANNA_IMPORT_ERROR

        query = state.get("query", "")
        t0 = time.time()

        sql, df, answer, policy_warnings = vanna_ask(self._vanna, query)
        latency_ms = int((time.time() - t0) * 1000)

        blocked = "blocked by safety policy" in (answer or "")

        # Full SQL in structured log — surfaces in LangSmith trace
        log.info(
            "sql_node.done",
            query=query[:80],
            sql=sql or "",
            rows=len(df) if df is not None else 0,
            latency_ms=latency_ms,
            policy_warnings=policy_warnings,
            blocked=blocked,
        )

        # ── Package result as a synthetic citation ─────────────────────────
        # PromptBuilder._context_block() iterates citations and uses
        # c.excerpt as the passage text shown to the LLM.
        if df is not None and not df.empty:
            excerpt    = answer   # markdown table produced by _format_dataframe
            confidence = 1.0     # DB result is authoritative
            # generate() short-circuits on structured_answer and never reaches
            # the "sql" LLM prompt, so the [1] marker has to be attached here
            # or the deterministic answer never cites its source at all.
            answer     = f"{answer}\n\n*Source: [1]*"
        else:
            # No data or blocked: give the LLM enough context to explain why
            excerpt    = f"Database query result: {answer or 'No data returned.'}"
            confidence = 0.0

        sql_citation = Citation(
            number=1,
            source_type="sql",
            file_name="AMFI Database",
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
            # Feed augmentation / guardrail pipeline
            "citations":       [sql_citation],
            "is_analytical":   True,   # bypass number_consistent check in post_guardrail
            "sql_context":     True,   # generate uses "sql" system prompt
            "structured_answer": answer,

            # Preserve SQL for LangSmith tracing (overwritten by generate later,
            # but captured in the sql_node.done log line above)
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
