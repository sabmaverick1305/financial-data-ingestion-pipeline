"""Analytical agent nodes — year-range aggregation loop.

For queries like "large cap fund investment from 2020 to 2026", the
single-pass RAG pipeline can't aggregate across 75 monthly documents.

Option A — Full aggregation: nested loops over years and months.

Outer loop (per year):
    plan_years → plan_months → [inner loop] → aggregate_year → [next year or synthesize]

Inner loop (per month within a year):
    retrieve_month → extract_month_metric → [next month or aggregate_year]

Each (year, month) pair:
  1. retrieve_month       : AMFI chunks for that specific year+month
  2. extract_month_metric : Claude Haiku JSON call → one value

After all months for a year are done, aggregate_year sums them.
After all years are done, synthesize builds the annual totals table
deterministically so the final answer does not depend on a free-form LLM
summary step.
"""
from __future__ import annotations

import json
import re
import time

import structlog
from sqlalchemy import text as sa_text

from financial_pipeline.augmentation.citations import Citation
from financial_pipeline.augmentation.generator import AnswerGenerator
from financial_pipeline.graph.state import RAGState

log = structlog.get_logger()

# ── Quarter → month mapping ───────────────────────────────────────────────────

_QUARTER_TO_MONTHS: dict[str, list[int]] = {
    "Q1": [1, 2, 3],
    "Q2": [4, 5, 6],
    "Q3": [7, 8, 9],
    "Q4": [10, 11, 12],
}

# ── Scheme type → display label ───────────────────────────────────────────────

_SCHEME_DISPLAY: dict[str, str] = {
    "large cap":        "Large Cap Fund",
    "mid cap":          "Mid Cap Fund",
    "small cap":        "Small Cap Fund",
    "multi cap":        "Multi Cap Fund",
    "large & mid cap":  "Large & Mid Cap Fund",
    "flexi cap":        "Flexi Cap Fund",
    "focused fund":     "Focused Fund",
    "value":            "Value Fund",
    "contra":           "Contra Fund",
    "elss":             "ELSS",
    "balanced":         "Balanced Fund",
    "hybrid":           "Hybrid Fund",
}


def _scheme_display_name(raw: str) -> str:
    """Normalise raw scheme_type to a display label for LLM prompts."""
    key = raw.lower().strip()
    if key in _SCHEME_DISPLAY:
        return _SCHEME_DISPLAY[key]
    # Generic title-case fallback
    return " ".join(w.capitalize() for w in key.split()) + " Fund"


# ── DB-first path: structured table coverage ─────────────────────────────────
# When metric + year range map cleanly to amfi_fund_stats (≥2020) or
# amfi_amc_stats (<2020), skip the month-by-month LLM loop and do one SQL query.

# amfi_fund_stats  (post-2020): fund_category column
_FUND_STATS_COLS: dict[str, str] = {
    "aum":             "aum",
    "avg_aum":         "avg_aum",
    "funds_mobilized": "funds_mobilized",
    "net_inflow":      "net_inflow",
    "redemption":      "redemption",
    "folios":          "no_of_folios",
}

# amfi_amc_stats (pre-2020): scheme_type column
_AMC_STATS_COLS: dict[str, str] = {
    "aum":             "aum",
    "funds_mobilized": "total_mobilized",
    "net_inflow":      "net_inflow",
    "redemption":      "redemption",
}

# Pre-2020 scheme mapping: intent scheme keyword → amfi_amc_stats scheme_type value.
# SEBI fund category reclassification only happened in Oct 2017, so "large cap",
# "mid cap" etc. don't exist in pre-2020 data — only broad types do.
_PRE2020_SCHEME_MAP: dict[str, str] = {
    "elss":         "ELSS - Equity",
    "equity":       "Equity",
    "income":       "Income",
    "gilt":         "Gilt",
    "balanced":     "Balanced",
    "gold etf":     "Gold ETF",
    "gold":         "Gold ETF",
    "liquid":       "Liquid/Money Market",
    "money market": "Liquid/Money Market",
    "total":        "Total",
    "industry":     "Total",
}

# ── Metric aggregation semantics ─────────────────────────────────────────────
# Stock metrics (snapshots): aggregate by average across months + year-end value.
# Summing monthly AUM snapshots is meaningless — we want avg or Dec figure.
_STOCK_METRICS = {"aum", "avg_aum", "nav"}

# Per-metric monthly value bounds for outlier detection (min, max in INR crore or count).
# Keep generous upper bounds — the old flat 50,000 cap was the bug for AUM.
_METRIC_BOUNDS: dict[str, tuple[float, float]] = {
    "funds_mobilized": (0.01,   100_000.0),     # crore/month
    "net_inflow":      (-100_000.0, 100_000.0), # can be negative outflow
    "redemption":      (0.01,   100_000.0),
    "aum":             (100.0,  100_000_000.0), # 100 cr → 100 lakh crore
    "avg_aum":         (100.0,  100_000_000.0),
    "folios":          (1.0,    500_000_000.0), # count, not crore
    "nav":             (0.01,   100_000.0),
}

# ── Metric detection keywords ─────────────────────────────────────────────────

_METRIC_KEYWORDS: dict[str, list[str]] = {
    # net_inflow must come before funds_mobilized — "net inflow" is more specific
    "net_inflow": [
        "net inflow", "net outflow", "net flow", "nett inflow",
    ],
    "redemption": [
        "redemption", "redempt", "repurchase", "withdrawal", "outflow",
    ],
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
    "net_inflow":      "Net Inflow",
    "redemption":      "Redemption",
    "folios":          "No. of Folios",
    "aum":             "Net Assets Under Management",
    "avg_aum":         "Average AUM",
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
AMFI Monthly Report context ({month}/{year}):

{context}

Task: Find the "{column}" value for "{scheme_name}".

Rules:
1. Find the row labelled "{scheme_name}" (case-insensitive). The row may have a number or
   roman numeral prefix (e.g. "ii {scheme_name}", "18 {scheme_name}").
2. If two scheme names appear together in one row (paired row), the cell may contain:
   - Two numbers separated by a space (e.g. "1,951.77 3,602.80"):
     first number → first scheme, second number → second scheme.
   - Single number: identify which scheme it belongs to by column position.
3. AMFI column order (left to right): Scheme Name → No. of Schemes → No. of Folios →
   Funds Mobilized → Repurchase/Redemption → Net Inflow → Net Assets (AUM) → Avg AUM.
   "Funds Mobilized" is the 4th column. AUM (Net Assets) is the 7th column — much larger
   numbers (lakh crore range). Do NOT return the AUM value when asked for Funds Mobilized.
4. "{column}" may appear in the column header as-is or with the year/month appended.
5. Values are in crore unless stated.

Reply with EXACTLY this JSON (no other text):
{{"value": "<number as string, e.g. 3449.52>", "month": <integer 1-12>, "year": {year}}}

If "{column}" for "{scheme_name}" is not clearly visible, reply:
{{"value": null, "month": null, "year": {year}}}
"""

_SYNTHESIS_SYSTEM = (
    "You are a financial analyst specialising in Indian mutual funds. "
    "Summarise year-wise data in a clear, concise table. "
    "Do NOT use [N] citation markers — values come from structured extraction, not passages. "
    "Acknowledge gaps explicitly."
)

_SYNTHESIS_FLOW_TMPL = """\
Annual {metric_label} data for {scheme_name} (sum of monthly values):

{table}

Provide:
1. A markdown table: Year | Annual Total (INR crore) | Months Summed | Missing Months.
2. A two-sentence trend summary (direction, notable years, CAGR if calculable).
3. Explicitly list years or months where data was unavailable.

Note: "Annual Total" = sum of monthly values for all available months in that year.
Years with fewer than 12 months indicate missing AMFI reports in the dataset.
"""

_SYNTHESIS_STOCK_TMPL = """\
Year-wise {metric_label} data for {scheme_name}:

{table}

Provide:
1. A markdown table: Year | Avg Monthly {metric_label} (₹ crore) | Year-end {metric_label} (₹ crore) | Months Available | Missing Months.
2. A two-sentence trend summary (direction, year-on-year growth, notable years).
3. Explicitly list years or months where data was unavailable.

Note: {metric_label} is a point-in-time snapshot, not a flow — values are NOT summed.
"Avg Monthly" = mean across available months. "Year-end" = value for the latest available month (ideally December).
Years with fewer than 12 months indicate missing AMFI reports in the dataset.
"""


# ── Factory ───────────────────────────────────────────────────────────────────

class AnalyticalNodeFactory:
    """Nodes for the year-range analytical agent.

    Instantiate once at app startup alongside NodeFactory:

        analytical = AnalyticalNodeFactory(repo=repo, generator=generator)
    """

    def __init__(self, repo, generator: AnswerGenerator | None = None) -> None:
        self._repo      = repo
        # Both call sites in this file (extract_month_metric, extract_metric) are
        # small structured-JSON extractions (max_tokens=80) run up to ~75x per
        # query — routed to the cheaper OpenAI tier by default. Pass an explicit
        # generator to override (e.g. back onto Claude) if needed.
        self._generator = generator or AnswerGenerator(provider="openai")

    # ── Node A1: plan_years ───────────────────────────────────────────────────

    def plan_years(self, state: RAGState) -> dict:
        """Build the year work queue and detect which metric to extract."""
        intent    = state.get("intent")
        year_from = getattr(intent, "year_from", None)
        year_to   = getattr(intent, "year_to", None)
        query     = state.get("query", "")

        years = list(range(year_from, year_to + 1)) if year_from and year_to else []
        # Prefer metric already extracted by the 5-stage intent pipeline;
        # fall back to keyword detection if the analyzer didn't set it.
        metric = getattr(intent, "metric", None) or _detect_metric(query)

        # Scheme: use first scheme_type from intent; default to "large cap"
        scheme_types  = getattr(intent, "scheme_types", None) or []
        target_scheme = scheme_types[0] if scheme_types else "large cap"

        # Quarter: convert "Q1"…"Q4" to a month allow-list; None = all months
        quarter       = getattr(intent, "quarter", None)
        quarter_months = _QUARTER_TO_MONTHS.get(quarter) if quarter else None

        log.info("node.plan_years",
                 years=years, metric=metric,
                 year_from=year_from, year_to=year_to,
                 target_scheme=target_scheme, quarter=quarter,
                 quarter_months=quarter_months)
        return {
            "pending_years":     years,
            "current_year":      None,
            "year_results":      {},
            "extraction_metric": metric,
            "target_scheme":     target_scheme,
            "quarter_months":    quarter_months,
        }

    # ── Node A1b: query_db_stats ──────────────────────────────────────────────

    def query_db_stats(self, state: RAGState) -> dict:
        """DB-first analytical path: single SQL query instead of the LLM month loop.

        Called when all requested years are covered by amfi_fund_stats (≥2020)
        or amfi_amc_stats (<2020) and the metric maps to a known column.
        Builds year_results in the same format as aggregate_year so synthesize
        can consume it unchanged.
        """
        pending_years  = state.get("pending_years", [])
        metric         = state.get("extraction_metric", "funds_mobilized")
        target_scheme  = state.get("target_scheme") or "large cap"
        quarter_months = state.get("quarter_months")
        is_stock       = metric in _STOCK_METRICS

        year_from = min(pending_years)
        year_to   = max(pending_years)
        all_post  = year_from >= 2020
        all_pre   = year_to   <  2020

        mapped = None
        if all_post:
            table   = "amfi_fund_stats"
            col     = _FUND_STATS_COLS[metric]
            cat_col = "fund_category"
            scheme_pattern = f"%{target_scheme}%"
        else:
            table   = "amfi_amc_stats"
            col     = _AMC_STATS_COLS[metric]
            cat_col = "scheme_type"
            # Resolve target_scheme → amfi_amc_stats scheme_type exact value
            scheme_key = target_scheme.lower().strip()
            mapped = next(
                (v for k, v in _PRE2020_SCHEME_MAP.items()
                 if scheme_key.startswith(k) or k in scheme_key),
                None,
            )
            scheme_pattern = mapped if mapped else f"%{target_scheme}%"

        month_filter = ""
        params: dict = {
            "scheme":    scheme_pattern,
            "year_from": year_from,
            "year_to":   year_to,
        }
        if quarter_months:
            month_filter = "AND period_month = ANY(:months)"
            params["months"] = quarter_months

        # For pre-2020 mapped scheme_types use exact match; post-2020 use ILIKE
        match_op = "=" if (not all_post and mapped) else "ILIKE"
        sql = f"""
            SELECT period_year, period_month, {col}
            FROM   {table}
            WHERE  {cat_col} {match_op} :scheme
              AND  period_year BETWEEN :year_from AND :year_to
              {month_filter}
            ORDER BY period_year, period_month
        """

        try:
            with self._repo._engine.connect() as conn:
                rows = conn.execute(sa_text(sql), params).fetchall()
        except Exception as exc:
            log.warning("node.query_db_stats.failed", error=str(exc))
            # Fall back to empty year_results — synthesize will acknowledge the gap
            rows = []

        # Group by year
        from collections import defaultdict
        by_year: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for yr, mo, val in rows:
            if val is not None:
                try:
                    by_year[int(yr)].append((int(mo), float(val)))
                except (TypeError, ValueError):
                    pass

        year_results: dict = {}
        for yr in pending_years:
            month_pairs = sorted(by_year.get(yr, []))  # sorted by month
            months_total = len(month_pairs)
            monthly_detail = [{"month": m, "value": str(v)} for m, v in month_pairs]

            if months_total == 0:
                year_results[yr] = {
                    "value": None, "months_found": 0,
                    "months_total": 0, "monthly_detail": [],
                    "is_stock": is_stock,
                }
                continue

            if is_stock:
                nums    = [v for _, v in month_pairs]
                avg_val = sum(nums) / len(nums)
                ye_val  = month_pairs[-1][1]   # latest month = year-end
                year_results[yr] = {
                    "value":          f"{avg_val:.2f}",
                    "year_end_value": f"{ye_val:.2f}",
                    "months_found":   months_total,
                    "months_total":   months_total,
                    "monthly_detail": monthly_detail,
                    "is_stock":       True,
                }
            else:
                total = sum(v for _, v in month_pairs)
                year_results[yr] = {
                    "value":         f"{total:.2f}",
                    "months_found":  months_total,
                    "months_total":  months_total,
                    "monthly_detail": monthly_detail,
                    "is_stock":      False,
                }

        # Pre-2020 + unmapped scheme: all years will be empty.
        # Inject a note so synthesize can produce a meaningful explanation
        # rather than a bare "no data" table.
        pre2020_no_mapping = (not all_post) and (mapped is None)
        pre2020_note = (
            f"Note: AMFI did not publish fund-category-level data (e.g. Large Cap, "
            f"Mid Cap) before the SEBI reclassification of October 2017. "
            f"Pre-2020 data is available only for broad scheme types: "
            f"{', '.join(_PRE2020_SCHEME_MAP.values())}."
        ) if pre2020_no_mapping else None

        log.info("node.query_db_stats",
                 table=table, metric=metric, col=col,
                 years=list(pending_years),
                 is_stock=is_stock,
                 rows_fetched=len(rows),
                 pre2020_no_mapping=pre2020_no_mapping)
        return {
            "year_results":      year_results,
            "pending_years":     [],
            "is_analytical":     True,
            "db_context_note":   pre2020_note,
        }

    # ── Node A2: plan_months ─────────────────────────────────────────────────

    def plan_months(self, state: RAGState) -> dict:
        """Query the DB for months available for the next pending year.

        Sets pending_months to all months that have embedded documents for
        that year, resets month_values to an empty list, and sets current_year.
        """
        pending_years = state.get("pending_years", [])
        if not pending_years:
            return {"pending_months": [], "month_values": [], "current_year": None}

        year = pending_years[0]
        try:
            with self._repo._engine.connect() as conn:
                months = conn.execute(sa_text("""
                    SELECT DISTINCT period_month
                    FROM document_metadata
                    WHERE period_year = :year
                      AND processing_status = 'embedded'
                    ORDER BY period_month ASC
                """), {"year": year}).scalars().all()
            available = list(months)
        except Exception as exc:
            log.warning("node.plan_months.failed", year=year, error=str(exc))
            available = list(range(1, 13))   # fallback: try all months

        # Apply quarter filter if set (e.g. Q1 → keep only months 1, 2, 3)
        quarter_months = state.get("quarter_months")
        if quarter_months:
            available = [m for m in available if m in quarter_months]

        log.info("node.plan_months", year=year, months=available,
                 quarter_months=quarter_months)
        return {
            "current_year":   year,
            "pending_months": available,
            "month_values":   [],
            "current_month":  None,
        }

    # ── Node A3: retrieve_month ───────────────────────────────────────────────

    def retrieve_month(self, state: RAGState) -> dict:
        """Retrieve chunks for the next pending (year, month) pair."""
        pending_months = state.get("pending_months", [])
        if not pending_months:
            return {"fused_results": [], "current_month": None}

        year   = state.get("current_year")
        month  = pending_months[0]
        intent = state.get("intent")
        scheme = (intent.scheme_types[0] if intent and intent.scheme_types
                  else "large cap")

        try:
            sql = """
                SELECT dc.chunk_id, dc.chunk_index, dc.text,
                       dc.period_year, dc.period_month, dc.category,
                       dm.file_name, dm.document_type, dm.s3_processed_key,
                       1.0 AS similarity
                FROM document_chunks dc
                JOIN document_metadata dm ON dc.document_id = dm.document_id
                WHERE dc.text ILIKE :pattern
                  AND dc.period_year  = :year
                  AND dc.period_month = :month
                ORDER BY dc.chunk_index ASC
                LIMIT 5
            """
            params = {"pattern": f"%{scheme}%", "year": year, "month": month}
            with self._repo._engine.connect() as conn:
                rows = conn.execute(sa_text(sql), params).mappings().all()

            chunks = []
            for r in rows:
                d = dict(r)
                d["similarity"]  = float(d.get("similarity") or 1.0)
                d["_source"]     = "chunk"
                d["search_mode"] = "analytical"
                chunks.append(d)

            log.debug("node.retrieve_month", year=year, month=month,
                      chunks=len(chunks))
            return {"fused_results": chunks, "current_month": month}

        except Exception as exc:
            log.warning("node.retrieve_month.failed",
                        year=year, month=month, error=str(exc))
            return {"fused_results": [], "current_month": month}

    # ── Node A4: extract_month_metric ─────────────────────────────────────────

    def extract_month_metric(self, state: RAGState) -> dict:
        """Extract one value from the current month's chunks.

        Appends result to month_values and pops pending_months[0].
        Null values (month not found / not parseable) are recorded as
        {"value": None} so aggregate_year can count missing months.
        """
        year   = state.get("current_year")
        month  = state.get("current_month")
        chunks = state.get("fused_results", [])
        metric = state.get("extraction_metric", "funds_mobilized")
        column = _METRIC_COLUMN.get(metric, "Funds Mobilized")

        # Build targeted context: header + scheme-matching rows only
        target_scheme = state.get("target_scheme") or "large cap"
        scheme_name   = _scheme_display_name(target_scheme)
        context_parts = []
        for c in chunks[:5]:
            txt   = (c.get("text") or "")
            fname = c.get("file_name", "")
            lines = txt.splitlines()
            header_lines = [l for l in lines[:6] if l.strip()][:2]
            data_lines   = [l for l in lines
                            if target_scheme.lower() in l.lower() and "|" in l]
            focused = "\n".join(header_lines + data_lines)
            if focused.strip():
                context_parts.append(f"[{fname}]\n{focused}")

        context   = "\n\n".join(context_parts) or "No data for this month."
        extracted = {"value": None, "month": month, "year": year}

        if context_parts:   # only call LLM if we have real context
            try:
                messages = [
                    {"role": "system", "content": _EXTRACT_SYSTEM},
                    {"role": "user",   "content": _EXTRACT_USER_TMPL.format(
                        month=month, year=year,
                        context=context, column=column,
                        scheme_name=scheme_name,
                    )},
                ]
                result = self._generator.generate(
                    messages, intent_type="factual", max_tokens=80,
                )
                raw    = re.sub(r"```(?:json)?\s*|\s*```", "",
                                result.answer.strip()).strip()
                parsed = json.loads(raw)
                if parsed.get("value"):
                    extracted["value"] = str(parsed["value"])
            except Exception as exc:
                log.warning("node.extract_month_metric.failed",
                            year=year, month=month, error=str(exc))

        log.debug("node.extract_month_metric",
                  year=year, month=month, value=extracted["value"])

        month_values   = list(state.get("month_values") or [])
        month_values.append(extracted)

        pending_months = list(state.get("pending_months", []))
        if pending_months:
            pending_months = pending_months[1:]

        return {"month_values": month_values, "pending_months": pending_months}

    # ── Node A5: aggregate_year ───────────────────────────────────────────────

    def aggregate_year(self, state: RAGState) -> dict:
        """Aggregate all monthly values for the current year and advance to next year.

        Flow metrics (funds_mobilized, net_inflow, redemption): annual sum.
        Stock metrics (aum, avg_aum, nav): average monthly + year-end (latest month).
        Bounds are per-metric to avoid silently dropping valid AUM figures.
        """
        year         = state.get("current_year")
        month_values = state.get("month_values") or []
        metric       = state.get("extraction_metric", "funds_mobilized")
        is_stock     = metric in _STOCK_METRICS
        lo, hi       = _METRIC_BOUNDS.get(metric, (0.01, 100_000.0))

        valid_values: list[tuple[int, float]] = []   # (month, value)

        for mv in month_values:
            v = mv.get("value")
            if v:
                try:
                    num = float(str(v).replace(",", ""))
                    if not (lo <= num <= hi):
                        log.warning("node.aggregate_year.outlier_skipped",
                                    year=year, month=mv.get("month"),
                                    value=v, metric=metric, lo=lo, hi=hi)
                        continue
                    valid_values.append((mv.get("month", 0), num))
                except (ValueError, TypeError):
                    pass

        months_found = len(valid_values)
        monthly_detail = [{"month": m, "value": str(v)} for m, v in valid_values]

        if months_found == 0:
            year_result = {
                "value":          None,
                "months_found":   0,
                "months_total":   len(month_values),
                "monthly_detail": [],
                "is_stock":       is_stock,
            }
        elif is_stock:
            # Stock: report average and year-end (latest-month) value
            nums       = [v for _, v in valid_values]
            avg_val    = sum(nums) / len(nums)
            # Year-end = value from the highest month number available
            year_end_v = max(valid_values, key=lambda t: t[0])[1]
            year_result = {
                "value":          f"{avg_val:.2f}",      # average used for table display
                "year_end_value": f"{year_end_v:.2f}",
                "months_found":   months_found,
                "months_total":   len(month_values),
                "monthly_detail": monthly_detail,
                "is_stock":       True,
            }
        else:
            # Flow: annual sum
            annual_total = sum(v for _, v in valid_values)
            year_result = {
                "value":          f"{annual_total:.2f}",
                "months_found":   months_found,
                "months_total":   len(month_values),
                "monthly_detail": monthly_detail,
                "is_stock":       False,
            }

        log.info("node.aggregate_year",
                 year=year,
                 metric=metric,
                 is_stock=is_stock,
                 value=year_result["value"],
                 months_found=months_found,
                 months_total=len(month_values))

        year_results   = dict(state.get("year_results") or {})
        year_results[year] = year_result

        # Pop the processed year
        pending_years  = list(state.get("pending_years", []))
        if pending_years:
            pending_years = pending_years[1:]

        return {
            "year_results":  year_results,
            "pending_years": pending_years,
            "month_values":  [],    # reset for next year
        }

    # ── Node A2 (legacy): retrieve_year ──────────────────────────────────────

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

        # Build targeted context: table header + scheme-matching rows only.
        target_scheme = state.get("target_scheme") or "large cap"
        scheme_name   = _scheme_display_name(target_scheme)
        context_parts = []
        for i, c in enumerate(chunks[:5], 1):   # scan all 5; data row may not be in first 3
            txt        = (c.get("text") or "")
            month      = c.get("period_month", "?")
            fname      = c.get("file_name", "")
            lines      = txt.splitlines()

            header_lines = [l for l in lines[:6] if l.strip()][:2]
            data_lines   = [l for l in lines
                            if target_scheme.lower() in l.lower() and "|" in l]

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
                    month="?", year=year, context=context,
                    column=column, scheme_name=scheme_name,
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

    # ── Node A6: synthesize ───────────────────────────────────────────────────

    def synthesize(self, state: RAGState) -> dict:
        """Build the final year-wise answer from all aggregated year totals.

        For Option A (full aggregation), each year_result has:
          value         — annual sum across all available months
          months_found  — how many months had extractable data
          months_total  — how many months were attempted
        """
        year_results    = state.get("year_results") or {}
        metric          = state.get("extraction_metric", "funds_mobilized")
        metric_label    = _METRIC_COLUMN.get(metric, "Funds Mobilized")
        target_scheme   = state.get("target_scheme") or "large cap"
        scheme_name     = _scheme_display_name(target_scheme)
        db_context_note = state.get("db_context_note")

        is_stock = metric in _STOCK_METRICS
        table_lines = []
        numeric_rows: list[tuple[int, float, int, int]] = []
        for yr in sorted(year_results.keys()):
            r          = year_results[yr]
            val        = r.get("value")
            mf         = r.get("months_found", 0)
            mt         = r.get("months_total", 0)
            missing_mo = mt - mf

            if val and is_stock:
                ye_val = r.get("year_end_value", val)
                table_lines.append(
                    f"| {yr} | ₹{val} crore | ₹{ye_val} crore | {mf}/{mt} | {missing_mo} |"
                )
                try:
                    numeric_rows.append((yr, float(str(val).replace(",", "")), mf, mt))
                except (TypeError, ValueError):
                    pass
            elif val:
                table_lines.append(
                    f"| {yr} | ₹{val} crore | {mf}/{mt} | {missing_mo} |"
                )
                try:
                    numeric_rows.append((yr, float(str(val).replace(",", "")), mf, mt))
                except (TypeError, ValueError):
                    pass
            else:
                table_lines.append(f"| {yr} | — (no data) | — | {mt} | {mt} |")

        if is_stock:
            header = (
                f"| Year | Avg Monthly {metric_label} (₹ crore) "
                f"| Year-end {metric_label} (₹ crore) | Months Available | Missing |\n"
                "|---|---|---|---|---|\n"
            )
            tmpl = _SYNTHESIS_STOCK_TMPL
        else:
            header = (
                "| Year | Annual Total (₹ crore) | Months Summed | Missing |\n"
                "|---|---|---|---|\n"
            )
            tmpl = _SYNTHESIS_FLOW_TMPL

        table = header + "\n".join(table_lines)

        found   = sum(1 for r in year_results.values() if r.get("value"))
        missing = [yr for yr, r in sorted(year_results.items()) if not r.get("value")]

        log.info("node.synthesize",
                 total_years=len(year_results),
                 is_stock=is_stock,
                 found=found,
                 missing=missing)

        lines = [
            f"Year-wise {metric_label} for {scheme_name}:",
            "",
            table,
        ]

        if numeric_rows:
            years = [yr for yr, _, _, _ in numeric_rows]
            values = [val for _, val, _, _ in numeric_rows]
            if is_stock:
                highest_year, highest_val, _, _ = max(numeric_rows, key=lambda t: t[1])
                lowest_year, lowest_val, _, _ = min(numeric_rows, key=lambda t: t[1])
                lines.extend([
                    "",
                    f"Across the available years, average monthly {metric_label} ranged from ₹{lowest_val:.2f} crore in {lowest_year} to ₹{highest_val:.2f} crore in {highest_year}.",
                    f"Year-end values are shown separately in the table. Missing years: {missing if missing else 'none'}.",
                ])
            else:
                highest_year, highest_val, _, _ = max(numeric_rows, key=lambda t: t[1])
                lowest_year, lowest_val, _, _ = min(numeric_rows, key=lambda t: t[1])
                lines.extend([
                    "",
                    f"Annual totals ranged from ₹{lowest_val:.2f} crore in {lowest_year} to ₹{highest_val:.2f} crore in {highest_year}.",
                    f"Missing years: {missing if missing else 'none'}.",
                ])
        else:
            lines.extend([
                "",
                "No structured year results were available for the requested range.",
            ])

        if db_context_note:
            lines.extend(["", f"Note: {db_context_note}"])

        answer = "\n".join(lines).strip()

        # Same reasoning as nodes_sql.py's query_sql: this answer is built
        # deterministically and never passes through PromptBuilder, so the
        # [1] source marker has to be attached here directly.
        source_citation = Citation(
            number=1,
            source_type="sql",
            file_name="AMFI Database",
            period_year=None,
            period_month=None,
            category=None,
            chunk_index=None,
            table_index=None,
            excerpt=table,
            confidence=1.0 if numeric_rows else 0.0,
            rank_method="sql",
        )
        answer = f"{answer}\n\n*Source: [1]*"

        generation_meta = {
            "model":             "deterministic",
            "provider":          "rules",
            "prompt_tokens":     0,
            "completion_tokens": 0,
            "latency_ms":        0,
        }

        return {
            "answer":          answer,
            "structured_answer": answer,
            "citations":       [source_citation],
            "generation_meta": generation_meta,
            "is_analytical":   True,   # tells post_guardrail to skip number_consistent
        }
