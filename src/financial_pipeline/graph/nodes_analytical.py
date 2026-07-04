"""Analytical agent nodes — year-range aggregation loop.

For queries like "large cap fund investment from 2020 to 2026", the
single-pass RAG pipeline can't aggregate across 75 monthly documents.

These four nodes implement a loop:
    plan_years → retrieve_year → extract_metric → [loop or synthesize]

Each iteration processes one year:
  1. retrieve_year  : dense + table search scoped to period_year=N
  2. extract_metric : focused LLM call → one JSON value per year

After all years are processed, synthesize builds the final year-wise table.

The loop runs at most (year_to - year_from + 1) times, then exits to
synthesize → post_guardrail → format_response.
"""
from __future__ import annotations

import json
import re
import time

import structlog
from sqlalchemy import text as sa_text

from financial_pipeline.augmentation.generator import AnswerGenerator
from financial_pipeline.graph.state import RAGState

log = structlog.get_logger()

# ── Metric detection keywords ─────────────────────────────────────────────────

_METRIC_KEYWORDS: dict[str, list[str]] = {
    "funds_mobilized": [
        "investment", "invest", "mobilized", "mobilised", "mobilization",
        "inflow", "collected", "raised", "subscription",
    ],
    "folios": [
        "folio", "investor", "account",
    ],
    "aum": [
        "aum", "asset under management", "assets under management",
        "net asset", "corpus",
    ],
    "nav": [
        "nav", "net asset value", "price",
    ],
}

_METRIC_COLUMN: dict[str, str] = {
    "funds_mobilized": "Funds Mobilized",
    "folios":          "No. of Folios",
    "aum":             "Net Assets Under Management",
    "nav":             "NAV",
}


def _detect_metric(query: str) -> str:
    q = query.lower()
    for metric, keywords in _METRIC_KEYWORDS.items():
        if any(kw in q for kw in keywords):
            return metric
    return "funds_mobilized"   # default for investment-style queries


# ── Prompts ───────────────────────────────────────────────────────────────────

_EXTRACT_SYSTEM = (
    "You are a precise financial data extractor. "
    "Read AMFI monthly report tables and extract one specific number. "
    "Reply ONLY with valid JSON — no explanation, no markdown fences."
)

_EXTRACT_USER_TMPL = """\
AMFI Monthly Report context (year {year}):

{context}

Task: Find the "{column}" value for "Large Cap Fund".

Rules:
1. The row may be labelled:
   - "ii Large Cap Fund"  (standalone)
   - "18 Large Cap Fund"  (numeric label, older reports)
   - "i Multi Cap Fund ii Large Cap Fund"  (merged with prior scheme — Large Cap is SECOND)
   - "ii Large Cap Fund iii Large & Mid Cap Fund"  (merged with next scheme — Large Cap is FIRST)
2. For PAIRED values in a single cell (e.g. "1,951.77 3,602.80" or "5,049.69 5,819.64"):
   - If row starts with "i Multi Cap Fund ii Large Cap": use the SECOND number (Large Cap).
   - If row starts with "ii Large Cap Fund iii": use the FIRST number (Large Cap).
3. For SINGLE values in a cell when the row contains both scheme names, treat the
   sequential columns as belonging alternately: 1st column → Multi Cap, 2nd → Large Cap.
4. AMFI column order (left to right): Scheme Name → No. of Schemes → No. of Folios →
   Funds Mobilized → Repurchase/Redemption → Net Inflow → Net Assets (AUM) → Avg AUM.
   "Funds Mobilized" is the 4th column. AUM (Net Assets) is the 7th column — much larger
   numbers (lakh crore). Do NOT return the AUM value.
5. "{column}" may appear in the column header as-is or with the year/month appended.
6. Ignore rows labelled "Large Cap ETF" or only "Large & Mid Cap".
7. Values are in crore unless stated.

Reply with EXACTLY this JSON (no other text):
{{"value": "<number as string, e.g. 3449.52>", "month": <integer 1-12>, "year": {year}}}

If "{column}" for Large Cap Fund is not clearly visible, reply:
{{"value": null, "month": null, "year": {year}}}
"""

_SYNTHESIS_SYSTEM = (
    "You are a financial analyst specialising in Indian mutual funds. "
    "Summarise year-wise data in a clear, concise table. "
    "Do NOT use [N] citation markers — values come from structured extraction, not passages. "
    "Acknowledge gaps explicitly."
)

_SYNTHESIS_USER_TMPL = """\
Year-wise {metric_label} data for Large Cap Funds extracted from AMFI monthly reports:

{table}

Provide:
1. A markdown table with Year | Value (INR crore) | Source Month columns.
2. A two-sentence trend summary.
3. Explicitly list years where data was not found.

Note: values are from one representative month per year (typically the latest
available), not annual totals. Annual totals would require summing all 12 months.
"""


# ── Factory ───────────────────────────────────────────────────────────────────

class AnalyticalNodeFactory:
    """Nodes for the year-range analytical agent.

    Instantiate once at app startup alongside NodeFactory:

        analytical = AnalyticalNodeFactory(repo=repo, generator=generator)
    """

    def __init__(self, repo, generator: AnswerGenerator | None = None) -> None:
        self._repo      = repo
        self._generator = generator or AnswerGenerator()

    # ── Node A1: plan_years ───────────────────────────────────────────────────

    def plan_years(self, state: RAGState) -> dict:
        """Build the year work queue and detect which metric to extract."""
        intent   = state.get("intent")
        year_from = getattr(intent, "year_from", None)
        year_to   = getattr(intent, "year_to", None)
        query     = state.get("query", "")

        years  = list(range(year_from, year_to + 1)) if year_from and year_to else []
        metric = _detect_metric(query)

        log.info("node.plan_years",
                 years=years, metric=metric,
                 year_from=year_from, year_to=year_to)
        return {
            "pending_years":     years,
            "current_year":      None,
            "year_results":      {},
            "extraction_metric": metric,
        }

    # ── Node A2: retrieve_year ────────────────────────────────────────────────

    def retrieve_year(self, state: RAGState) -> dict:
        """Retrieve chunks for the next pending year.

        Prefers December (year-end figure); falls back to the latest
        available month.  Keeps to 5 chunks — enough context for one
        LLM extraction call.
        """
        pending = state.get("pending_years", [])
        if not pending:
            return {"fused_results": [], "current_year": None}

        year   = pending[0]
        intent = state.get("intent")
        scheme = (intent.scheme_types[0] if intent and intent.scheme_types
                  else "large cap")

        try:
            # Prefer December (annual snapshot); fall back to latest month
            sql = """
                SELECT dc.chunk_id, dc.chunk_index, dc.text,
                       dc.period_year, dc.period_month, dc.category,
                       dm.file_name, dm.document_type, dm.s3_processed_key,
                       1.0 AS similarity
                FROM document_chunks dc
                JOIN document_metadata dm ON dc.document_id = dm.document_id
                WHERE dc.text ILIKE :pattern
                  AND dc.period_year = :year
                ORDER BY dc.period_month DESC,   -- latest month first
                         dc.chunk_index ASC
                LIMIT 5
            """
            params = {"pattern": f"%{scheme}%", "year": year}
            with self._repo._engine.connect() as conn:
                rows = conn.execute(sa_text(sql), params).mappings().all()

            chunks = []
            for r in rows:
                d = dict(r)
                d["similarity"]  = float(d.get("similarity") or 1.0)
                d["_source"]     = "chunk"
                d["search_mode"] = "analytical"
                chunks.append(d)

            log.info("node.retrieve_year", year=year, chunks=len(chunks))
            return {"fused_results": chunks, "current_year": year}

        except Exception as exc:
            log.warning("node.retrieve_year.failed", year=year, error=str(exc))
            return {"fused_results": [], "current_year": year}

    # ── Node A3: extract_metric ───────────────────────────────────────────────

    def extract_metric(self, state: RAGState) -> dict:
        """Focused LLM call: extract one number from the current year's chunks.

        Pops the processed year from pending_years and accumulates the
        result in year_results.  Handles parse errors and null values
        gracefully — a failed extraction records {"value": null}.
        """
        year    = state.get("current_year")
        chunks  = state.get("fused_results", [])
        metric  = state.get("extraction_metric", "funds_mobilized")
        column  = _METRIC_COLUMN.get(metric, "Funds Mobilized")

        # Build targeted context: table header + the Large Cap Fund row only.
        # Taking first N chars misses the data row which can be thousands of
        # chars into a wide-table chunk. Instead, extract:
        #   lines 0-1  → column header + separator  (schema context)
        #   the line(s) containing "large cap fund" → actual data
        context_parts = []
        for i, c in enumerate(chunks[:5], 1):   # scan all 5; data row may not be in first 3
            txt   = (c.get("text") or "")
            month = c.get("period_month", "?")
            fname = c.get("file_name", "")
            lines = txt.splitlines()

            # Header: first two non-empty lines (column names + separator)
            header_lines = [l for l in lines[:6] if l.strip()][:2]

            # Data: lines containing "large cap" (case-insensitive)
            data_lines = [l for l in lines
                          if "large cap" in l.lower() and "|" in l]

            focused = "\n".join(header_lines + data_lines)
            if not focused.strip():
                focused = txt[:800]   # fallback if pattern not found

            if focused.strip():
                context_parts.append(
                    f"[Source {i} — {fname}, month {month}]\n{focused}"
                )
        context = "\n\n".join(context_parts) or "No data retrieved for this year."

        extracted: dict = {"value": None, "month": None, "year": year}

        try:
            messages = [
                {"role": "system", "content": _EXTRACT_SYSTEM},
                {"role": "user",   "content": _EXTRACT_USER_TMPL.format(
                    year=year, context=context, column=column,
                )},
            ]
            result = self._generator.generate(
                messages,
                intent_type="factual",
                max_tokens=80,
            )
            raw = result.answer.strip()
            # Strip markdown fences if the model added them
            raw = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
            parsed = json.loads(raw)
            extracted = {
                "value": str(parsed["value"]) if parsed.get("value") else None,
                "month": parsed.get("month"),
                "year":  year,
            }
            log.info("node.extract_metric",
                     year=year, value=extracted["value"],
                     month=extracted["month"])
        except Exception as exc:
            log.warning("node.extract_metric.failed", year=year, error=str(exc))

        # Accumulate and advance the work queue
        year_results = dict(state.get("year_results") or {})
        year_results[year] = extracted

        pending = list(state.get("pending_years", []))
        if pending:
            pending = pending[1:]   # pop processed year

        return {
            "year_results":  year_results,
            "pending_years": pending,
        }

    # ── Node A4: synthesize ───────────────────────────────────────────────────

    def synthesize(self, state: RAGState) -> dict:
        """Build the final year-wise answer from all extracted values.

        Formats a markdown table, calls the LLM for a trend summary,
        then writes to `answer` so the existing post_guardrail →
        format_response path can handle it unchanged.
        """
        year_results  = state.get("year_results") or {}
        metric        = state.get("extraction_metric", "funds_mobilized")
        metric_label  = _METRIC_COLUMN.get(metric, "Funds Mobilized")

        # Build citation-style table rows for the LLM
        table_lines = []
        for yr in sorted(year_results.keys()):
            r = year_results[yr]
            val   = r.get("value")
            month = r.get("month")
            if val and month:
                table_lines.append(
                    f"| {yr} | ₹{val} crore | Month {month}/{yr} |"
                )
            else:
                table_lines.append(f"| {yr} | — (not found) | — |")

        table = (
            "| Year | Value | Source Month |\n"
            "|---|---|---|\n"
            + "\n".join(table_lines)
        )

        found    = sum(1 for r in year_results.values() if r.get("value"))
        missing  = [yr for yr, r in sorted(year_results.items()) if not r.get("value")]

        log.info("node.synthesize",
                 total_years=len(year_results),
                 found=found,
                 missing=missing)

        try:
            messages = [
                {"role": "system", "content": _SYNTHESIS_SYSTEM},
                {"role": "user",   "content": _SYNTHESIS_USER_TMPL.format(
                    metric_label=metric_label,
                    table=table,
                )},
            ]
            result = self._generator.generate(
                messages,
                intent_type="trend",
                max_tokens=600,
            )
            answer = result.answer
            generation_meta = {
                "model":             result.model,
                "provider":          result.provider,
                "prompt_tokens":     result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "latency_ms":        result.latency_ms,
            }
        except Exception as exc:
            log.warning("node.synthesize.failed", error=str(exc))
            answer = (
                f"Year-wise {metric_label} for Large Cap Funds:\n\n{table}\n\n"
                f"Data found for {found} of {len(year_results)} years. "
                f"Missing: {missing if missing else 'none'}."
            )
            generation_meta = {}

        return {
            "answer":          answer,
            "generation_meta": generation_meta,
            "is_analytical":   True,   # tells post_guardrail to skip number_consistent
        }
