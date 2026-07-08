"""QueryAnalyzer — 5-stage financial intent classification pipeline.

Stage 1: Rule-based parser         — regex + keyword matching, <1 ms, always runs
Stage 2: Embedding classifier      — top-3 cosine similarity vote vs labelled examples, ~50 ms
Stage 3: LLM structured extractor  — small model JSON call with rich schema, ~300–500 ms
Stage 4: Confidence + ambiguity    — merge outputs, resolve conflicts
Stage 5: Final QueryIntent         — structured dataclass returned to caller

Stages 2 and 3 are optional: the analyzer degrades gracefully to stage-1-only
when no embedder or generator is supplied (useful in tests and CLI tools).

Typical latencies (p50):
  - Rule-based only    : <1 ms
  - + Embedding        : ~50 ms  (model already warm from retrieval)
  - + LLM extractor    : ~400 ms (Haiku, max_tokens=200)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

import structlog

from .ontology_resolver import CanonicalQuery, resolve as resolve_ontology

log = structlog.get_logger()

# ── Confidence thresholds ──────────────────────────────────────────────────────

_CONF_THRESHOLD_RULE = 0.82   # skip stages 2+3 when stage-1 confidence >= this
_CONF_THRESHOLD_EMB  = 0.78   # skip stage 3 when stage-2 confidence >= this
                               # AND stage-1 intent_type agrees

# ── Vocabulary tables ─────────────────────────────────────────────────────────

MONTH_MAP: dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sep": 9, "sept": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

QUARTER_MAP: dict[str, tuple[int, int]] = {
    "q1": (1, 3), "q2": (4, 6), "q3": (7, 9), "q4": (10, 12),
}

TABLE_SIGNALS = frozenset({
    # "table" intentionally excluded — substring-matches "notable", causing false positives
    # "how many" excluded — ambiguous: single-count queries (factual) fire it falsely
    "breakdown", "distribution", "count", "number of",
    "total number", "scheme count", "fund count",
    "list of", "show me", "give me the list", "aum breakdown",
    "folio count", "category-wise", "type-wise", "wise break",
})

TREND_SIGNALS = frozenset({
    "growth", "trend", "over the years", "historical", "changed",
    "increase", "decrease", "decline", "rise", "fell", "grew", "grown",
    "performance over", "how has", "how did", "evolution", "trajectory",
    "evolve", "evolving",
    "year wise", "year-wise", "yearwise", "year by year",
    "annually", "each year", "every year", "from 20",
    # Monthly/pattern signals that imply time-series (were only in _TREND_TYPE_KWS)
    "pattern", "track", "month-by-month", "month by month",
})

CAUSAL_SIGNALS = frozenset({
    "why did", "why has", "why is", "why was", "why does", "why do",
    "reason for", "reasons for", "explain why", "what caused",
    "what explains", "what is causing", "what's causing",
})

COMPARISON_SIGNALS = frozenset({
    "compare", "versus", "vs", "vs.", "difference between",
    "equity vs", "debt vs", "better than", "higher than",
    "lower than", "relative to", "against", "benchmark",
})

DEFINITION_SIGNALS = frozenset({
    "what is", "what are", "define", "explain", "meaning of",
    "what does", "how does", "description of", "definition",
})

LOOKUP_SIGNALS = frozenset({
    "get", "fetch", "download", "show me the report",
    "give me the", "monthly report for", "quarterly report for",
    "repository for",
    "which document", "which report", "which amfi report",
    "which amfi monthly", "covers the data for", "covers fund data",
})

DOCUMENT_SUMMARY_SIGNALS = frozenset({
    "summarize", "summarise", "summary of", "overview of",
    "key highlights", "highlights from", "what does the report say",
    "brief me on", "give me a summary", "what happened in",
    "key takeaways", "main points", "overview of the",
    # Patterns for AMFI report content queries
    "what did amfi", "highlights of amfi", "what were the highlights",
    "what information was", "notable data points", "data points in amfi",
    "key trends in the amfi", "key trends in amfi", "what trends",
})

RANKING_SIGNALS = frozenset({
    "highest", "lowest", "top", "best", "worst", "maximum",
    "minimum", "most", "least", "largest", "smallest", "biggest",
    "rank", "ranked", "top 5", "top 10", "leading",
})

REGULATORY_SIGNALS = frozenset({
    "sebi", "regulation", "regulations", "regulatory",
    "guideline", "guidelines", "circular", "compliance",
    "directive", "norms", "sebi circular", "amfi regulation",
    "amfi guidelines", "amfi circular", "regulatory framework",
    "expense ratio regulation", "expense ratio limit",
})

PERFORMANCE_ADVICE_SIGNALS = frozenset({
    "best performing fund", "best performing funds",
    "best fund", "best funds", "top fund", "top funds",
    "good cagr", "high cagr", "strong cagr",
    "maximum returns", "best returns", "long-term returns",
    "fund performance", "performing fund", "performing funds",
})

# ── Time-mode signals ─────────────────────────────────────────────────────────

YOY_SIGNALS = frozenset({
    "year-on-year", "year on year", "yoy", "y/y",
    "year over year", "annual change", "annual growth rate",
    "compared to last year", "vs last year", "vs previous year",
    "same period last year", "prior year",
})

MOM_SIGNALS = frozenset({
    "month-on-month", "month on month", "mom", "m/m",
    "monthly change", "monthly growth rate",
    "compared to last month", "vs last month", "vs previous month",
    "prior month",
})

SNAPSHOT_SIGNALS = frozenset({
    "as of", "as on", "as at", "point in time",
    "at the end of", "end of month", "end of year", "end of quarter",
})

LATEST_SIGNALS = frozenset({
    "latest", "current", "recent", "now", "today",
    "most recent", "newest", "up to date", "updated",
})

SCHEME_TYPES = [
    "equity", "debt", "hybrid", "liquid", "gilt", "balanced",
    "gold etf", "etf", "index fund", "index funds", "index",
    "elss", "fund of funds", "balanced advantage", "arbitrage",
    "overnight", "money market", "credit risk", "dynamic bond",
    "large cap", "mid cap", "small cap", "flexi cap", "multi cap",
    "thematic", "sectoral", "conservative hybrid", "aggressive hybrid",
    "fixed maturity", "fmp", "interval fund", "gold", "international",
    "income",
]

FINANCIAL_TERMS = frozenset({
    "aum", "nav", "folio", "sip", "lumpsum", "redemption",
    "mobilisation", "inflow", "outflow", "net", "gross",
    "scheme", "fund", "category", "subcategory",
})

# ── Metric keywords ───────────────────────────────────────────────────────────

_METRIC_MAP: dict[str, list[str]] = {
    "folios": [
        "folio", "folios", "investor count", "number of investors",
        "investor accounts", "no of folios", "no. of folios",
        "investors", "investor",
    ],
    "nav": [
        "nav", "net asset value", "price per unit", "unit price", "unit nav",
    ],
    "sip": [
        "sip", "systematic investment", "sip inflow", "sip amount",
        "sip contribution",
    ],
    "redemption": [
        "redemption", "redeem", "redemptions", "outflow", "withdrawal",
        "repurchase", "net redemption",
        "went out", "money out", "money went out", "outflows",
        "pulled out", "taken out", "withdrawn",
    ],
    "aum": [
        "aum", "asset under management", "assets under management",
        "net asset", "corpus", "total assets", "assets managed", "fund size",
    ],
    "schemes": [
        "number of schemes", "scheme count", "how many schemes",
        "no of schemes", "no. of schemes",
    ],
    "net_inflow": [
        "net inflow", "net outflow", "net flow", "nett inflow", "net investment",
        "net capital", "net collection",
    ],
    "funds_mobilized": [
        "investment", "invest", "mobilized", "mobilised", "mobilization",
        "mobilisation", "inflow", "collected", "raised", "subscription",
        "money invested", "amount invested",
        "came in", "come in", "money came in", "money in",
        "fund flows", "fund flow", "flows into", "fresh flow", "fresh fund",
    ],
}

# ── Aggregation keywords ──────────────────────────────────────────────────────

_AGGREGATION_MAP: dict[str, list[str]] = {
    "rank": [
        "highest", "lowest", "top", "best", "worst", "maximum", "minimum",
        "most", "least", "largest", "smallest", "biggest", "leading",
        "top 5", "top 10", "rank", "ranked",
    ],
    "cagr": [
        "cagr", "compounded", "annualized return", "compound annual",
        "growth rate",
    ],
    "year_over_year": [
        "year-on-year", "year on year", "yoy", "y/y",
        "year over year", "annual change", "compared to last year",
        "vs last year", "vs previous year", "same period last year",
        "prior year",
    ],
    "month_over_month": [
        "month-on-month", "month on month", "mom", "m/m",
        "monthly change", "compared to last month", "vs last month",
        "vs previous month", "prior month",
    ],
    "latest_available_period": [
        "latest", "most recent", "newest", "up to date",
        "as of today", "updated",
    ],
    "sum": [
        "total", "aggregate", "cumulative", "sum", "overall",
        "combined", "across all", "for the year", "annual total",
        "year wise total", "yearly total",
    ],
    "point_in_time": [
        "as of", "as on", "as at", "at the end of",
        "end of month", "end of year",
    ],
}

# ── Granularity keywords ──────────────────────────────────────────────────────

_GRANULARITY_MAP: dict[str, list[str]] = {
    "monthly": [
        "monthly", "month by month", "each month", "per month",
        "month wise", "month-wise",
    ],
    "quarterly": [
        "quarterly", "quarter by quarter", "each quarter", "per quarter",
        "quarter wise", "quarter-wise", "qoq", "quarter-on-quarter",
    ],
    "annual": [
        "annual", "yearly", "year by year", "year wise", "year-wise",
        "per year", "annually", "each year", "every year",
    ],
    "daily": [
        "daily", "day by day", "per day", "day wise",
    ],
}

# ── AMC / fund-house names ────────────────────────────────────────────────────

_AMC_NAMES: list[tuple[str, str]] = [
    ("icici prudential", "ICICI Prudential"),
    ("aditya birla sun life", "Aditya Birla Sun Life"),
    ("aditya birla", "Aditya Birla Sun Life"),
    ("canara robeco", "Canara Robeco"),
    ("baroda bnp paribas", "Baroda BNP Paribas"),
    ("baroda bnp", "Baroda BNP Paribas"),
    ("bank of india", "Bank of India"),
    ("bajaj finserv", "Bajaj Finserv"),
    ("motilal oswal", "Motilal Oswal"),
    ("mirae asset", "Mirae Asset"),
    ("franklin templeton", "Franklin Templeton"),
    ("whiteoak", "WhiteOak"),
    ("360 one", "360 ONE"),
    ("nippon india", "Nippon India"),
    ("hdfc", "HDFC"),
    ("icici", "ICICI"),
    ("kotak", "Kotak"),
    ("nippon", "Nippon India"),
    ("franklin", "Franklin Templeton"),
    ("mirae", "Mirae Asset"),
    ("pgim", "PGIM India"),
    ("invesco", "Invesco"),
    ("quantum", "Quantum"),
    ("ppfas", "PPFAS"),
    ("edelweiss", "Edelweiss"),
    ("sundaram", "Sundaram"),
    ("samco", "Samco"),
    ("groww", "Groww"),
    ("tata", "Tata"),
    ("axis", "Axis"),
    ("navi", "Navi"),
    ("dsp", "DSP"),
    ("sbi", "SBI"),
    ("uti", "UTI"),
    ("l&t", "L&T"),
]

# ── Label taxonomy ────────────────────────────────────────────────────────────

VALID_LABELS = frozenset({
    "definition_query",
    "numeric_aggregation",
    "snapshot_query",
    "trend_query",
    "comparison_query",
    "document_summary",
    "table_lookup",
    "investment_advice",
    "out_of_scope",
})

# Maps intent_type → default labels[] for stage-1 population
_INTENT_TO_LABELS: dict[str, list[str]] = {
    "factual":    ["snapshot_query"],
    "analytical": ["numeric_aggregation"],
    "trend":      ["trend_query"],
    "comparison": ["comparison_query"],
    "definition": ["definition_query"],
    "tabular":    ["table_lookup"],
    "lookup":     ["document_summary"],
    "regulatory": ["definition_query", "snapshot_query"],
}

# Maps primary_label (from LLM) → intent_type for routing
_LABEL_TO_INTENT: dict[str, str] = {
    "definition_query":  "definition",
    "numeric_aggregation": "analytical",
    "snapshot_query":    "factual",
    "trend_query":       "trend",
    "comparison_query":  "comparison",
    "document_summary":  "lookup",
    "table_lookup":      "tabular",
    "investment_advice": "factual",   # guardrail catches this downstream
    "out_of_scope":      "factual",   # guardrail catches this downstream
}

# ── Stage 2: Labelled example corpus ─────────────────────────────────────────

_LABELED_EXAMPLES: list[tuple[str, dict]] = [
    # ── Advice / performance-ranking boundary ─────────────────────────────
    ("what is the best performing fund for the last 5 years with good cagr around 7%",
     {"intent_type": "factual", "labels": ["investment_advice"], "metric": None,
      "aggregation": "cagr", "time_mode": "range", "needs_analytical": False}),
    ("which mutual funds have the best returns for a five year investment",
     {"intent_type": "factual", "labels": ["investment_advice"], "metric": None,
      "aggregation": "cagr", "time_mode": "range", "needs_analytical": False}),
    ("top funds with strong cagr that i should invest in",
     {"intent_type": "factual", "labels": ["investment_advice"], "metric": None,
      "aggregation": "cagr", "time_mode": "range", "needs_analytical": False}),

    # ── Snapshot / point-in-time ───────────────────────────────────────────
    ("what is the total aum of large cap funds",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("what is the nav of hdfc large cap fund",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "nav",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("how many folios does the large cap category have as of may 2026",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "folios",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("what is the total number of folios in large cap funds",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "folios",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("what was the funds mobilized in large cap in march 2026",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "funds_mobilized",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("what is the net inflow for equity funds",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "funds_mobilized",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("what is the redemption amount for small cap funds in 2026",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "redemption",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("total assets under management of mid cap category",
     {"intent_type": "factual", "labels": ["snapshot_query", "numeric_aggregation"],
      "metric": "aum", "aggregation": "sum", "time_mode": "snapshot"}),
    ("aum of equity funds as on december 2025",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),

    # ── Latest / current ───────────────────────────────────────────────────
    ("what is the latest aum of equity mutual funds",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "aum",
      "aggregation": "latest_available_period", "time_mode": "latest"}),
    ("current folio count for large cap funds",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "folios",
      "aggregation": "latest_available_period", "time_mode": "latest"}),
    ("most recent sip contribution data for equity funds",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "sip",
      "aggregation": "latest_available_period", "time_mode": "latest"}),
    ("what is the sip contribution for large cap funds this month",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "sip",
      "aggregation": "latest_available_period", "time_mode": "latest"}),
    ("give me the latest redemption figures for debt funds",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "redemption",
      "aggregation": "latest_available_period", "time_mode": "latest"}),

    # ── Table / ranking ────────────────────────────────────────────────────
    ("give me the category wise aum breakdown",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("show me the number of schemes in each category",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "schemes",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("how many schemes are there in the debt category",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "schemes",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("list all equity subcategories with their aum",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("folio count distribution across all fund categories",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "folios",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("what is the aum distribution across equity subcategories",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("which large cap fund has the highest aum",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "aum",
      "aggregation": "rank", "time_mode": "snapshot"}),
    ("top 5 equity funds by assets under management",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "aum",
      "aggregation": "rank", "time_mode": "snapshot"}),
    ("which category saw the highest inflow in march 2026",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "funds_mobilized",
      "aggregation": "rank", "time_mode": "snapshot"}),
    ("what is the smallest mid cap fund by aum",
     {"intent_type": "tabular", "labels": ["table_lookup"], "metric": "aum",
      "aggregation": "rank", "time_mode": "snapshot"}),

    # ── Time-series / trend ────────────────────────────────────────────────
    ("how has aum of large cap funds changed over the years",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "annual"}),
    ("show me the growth in folio count over the last 3 years",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "folios",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "annual"}),
    ("how did large cap fund aum evolve from 2022 to 2024",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "annual"}),
    ("trajectory of mid cap fund mobilization over years",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "funds_mobilized",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "annual"}),
    ("monthly trend of sip contributions in 2025",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "sip",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "monthly"}),
    ("quarterly aum growth for equity funds",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "quarterly"}),
    ("historical trend of large cap fund inflows",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "funds_mobilized",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "annual"}),

    # ── YoY (year-over-year) ───────────────────────────────────────────────
    ("what is the year on year change in large cap aum",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "year_over_year", "time_mode": "yoy", "granularity": "annual",
      "comparison_period": "previous_year"}),
    ("yoy growth in equity fund inflows for march 2026",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "funds_mobilized",
      "aggregation": "year_over_year", "time_mode": "yoy", "granularity": "annual",
      "comparison_period": "previous_year"}),
    ("how did sip contributions change compared to last year",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "sip",
      "aggregation": "year_over_year", "time_mode": "yoy", "granularity": "annual",
      "comparison_period": "previous_year"}),
    ("large cap aum in march 2026 versus march 2025",
     {"intent_type": "comparison", "labels": ["comparison_query", "trend_query"],
      "metric": "aum", "aggregation": "year_over_year", "time_mode": "yoy",
      "comparison_period": "same_period_last_year"}),
    ("equity inflows this month vs same period last year",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "funds_mobilized",
      "aggregation": "year_over_year", "time_mode": "yoy",
      "comparison_period": "same_period_last_year"}),
    ("y/y change in mid cap fund aum",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "year_over_year", "time_mode": "yoy", "granularity": "annual",
      "comparison_period": "previous_year"}),

    # ── MoM (month-over-month) ─────────────────────────────────────────────
    ("what is the month on month change in large cap aum",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "month_over_month", "time_mode": "mom", "granularity": "monthly",
      "comparison_period": "previous_month"}),
    ("mom growth in sip contributions for march 2026",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "sip",
      "aggregation": "month_over_month", "time_mode": "mom", "granularity": "monthly",
      "comparison_period": "previous_month"}),
    ("how did equity inflows change compared to last month",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "funds_mobilized",
      "aggregation": "month_over_month", "time_mode": "mom", "granularity": "monthly",
      "comparison_period": "previous_month"}),
    ("redemption in february 2026 vs january 2026",
     {"intent_type": "comparison", "labels": ["comparison_query", "trend_query"],
      "metric": "redemption", "aggregation": "month_over_month", "time_mode": "mom",
      "granularity": "monthly", "comparison_period": "previous_month"}),
    ("m/m change in folio count for large cap funds",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "folios",
      "aggregation": "month_over_month", "time_mode": "mom", "granularity": "monthly",
      "comparison_period": "previous_month"}),

    # ── Numeric aggregation (multi-year range) ─────────────────────────────
    # "analytical" is not a ground-truth intent_type (see _fix_year_range_intent):
    # plain "total/annual/cumulative X from Y to Z" is a tabular data request that
    # query_sql (Vanna GROUP BY / UNION ALL) already serves directly — needs_analytical
    # is reserved for queries carrying an explicit trend-language signal (_TREND_TYPE_KWS).
    ("total investment in large cap funds from 2020 to 2026",
     {"intent_type": "tabular", "labels": ["numeric_aggregation"], "metric": "funds_mobilized",
      "aggregation": "sum", "time_mode": "range", "granularity": "annual",
      "needs_analytical": False}),
    ("year wise funds mobilized in large cap from 2021 to 2025",
     {"intent_type": "trend", "labels": ["numeric_aggregation", "trend_query"],
      "metric": "funds_mobilized", "aggregation": "sum", "time_mode": "range",
      "granularity": "annual", "needs_analytical": True}),
    ("what is the annual total mobilization of mid cap funds from 2022 to 2026",
     {"intent_type": "tabular", "labels": ["numeric_aggregation"], "metric": "funds_mobilized",
      "aggregation": "sum", "time_mode": "range", "granularity": "annual",
      "needs_analytical": False}),
    ("cumulative folio count of equity funds between 2019 and 2024",
     {"intent_type": "tabular", "labels": ["numeric_aggregation"], "metric": "folios",
      "aggregation": "sum", "time_mode": "range", "granularity": "annual",
      "needs_analytical": False}),
    ("year by year aum of large cap from 2020 to 2026",
     {"intent_type": "trend", "labels": ["numeric_aggregation", "trend_query"],
      "metric": "aum", "aggregation": "sum", "time_mode": "range",
      "granularity": "annual", "needs_analytical": True}),
    ("how much was mobilized annually in large cap from 2018 to 2024",
     {"intent_type": "tabular", "labels": ["numeric_aggregation"], "metric": "funds_mobilized",
      "aggregation": "sum", "time_mode": "range", "granularity": "annual",
      "needs_analytical": False}),
    ("total funds mobilized from 2020 to 2023",
     {"intent_type": "tabular", "labels": ["numeric_aggregation"], "metric": "funds_mobilized",
      "aggregation": "sum", "time_mode": "range", "granularity": "annual",
      "needs_analytical": False}),
    ("search for the year wise investment done in large cap funds from 2020 to 2026",
     {"intent_type": "trend", "labels": ["numeric_aggregation", "trend_query"],
      "metric": "funds_mobilized", "aggregation": "sum", "time_mode": "range",
      "granularity": "annual", "needs_analytical": True}),

    # ── Comparison ─────────────────────────────────────────────────────────
    ("compare aum of equity vs debt funds",
     {"intent_type": "comparison", "labels": ["comparison_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("which performed better large cap or mid cap in 2024",
     {"intent_type": "comparison", "labels": ["comparison_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("difference in folio count between equity and debt categories",
     {"intent_type": "comparison", "labels": ["comparison_query"], "metric": "folios",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("sbi vs hdfc large cap fund aum comparison",
     {"intent_type": "comparison", "labels": ["comparison_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),

    # ── Compound / multi-intent ────────────────────────────────────────────
    ("compare large cap inflows from 2020 to 2026 and summarize the trend",
     {"intent_type": "comparison", "labels": ["comparison_query", "trend_query", "numeric_aggregation"],
      "metric": "funds_mobilized", "aggregation": "sum", "time_mode": "range",
      "granularity": "annual", "needs_analytical": True}),
    ("show me the yoy trend in equity fund aum and compare to debt funds",
     {"intent_type": "comparison", "labels": ["comparison_query", "trend_query"],
      "metric": "aum", "aggregation": "year_over_year", "time_mode": "yoy",
      "granularity": "annual", "comparison_period": "previous_year"}),
    ("what is the monthly sip trend for 2025 and how does it compare to 2024",
     {"intent_type": "comparison", "labels": ["comparison_query", "trend_query"],
      "metric": "sip", "aggregation": "year_over_year", "time_mode": "time_series",
      "granularity": "monthly", "comparison_period": "previous_year"}),

    # ── Definition ─────────────────────────────────────────────────────────
    ("what is a large cap fund",
     {"intent_type": "definition", "labels": ["definition_query"], "metric": None,
      "aggregation": None, "time_mode": None}),
    ("explain the difference between aum and nav",
     {"intent_type": "definition", "labels": ["definition_query"], "metric": None,
      "aggregation": None, "time_mode": None}),
    ("what does folio mean in mutual funds",
     {"intent_type": "definition", "labels": ["definition_query"], "metric": "folios",
      "aggregation": None, "time_mode": None}),
    ("what is elss and how does it work",
     {"intent_type": "definition", "labels": ["definition_query"], "metric": None,
      "aggregation": None, "time_mode": None}),
    ("explain what sip means",
     {"intent_type": "definition", "labels": ["definition_query"], "metric": "sip",
      "aggregation": None, "time_mode": None}),
    ("what is the difference between equity and debt mutual funds",
     {"intent_type": "definition", "labels": ["definition_query"], "metric": None,
      "aggregation": None, "time_mode": None}),
    ("how does nav get calculated",
     {"intent_type": "definition", "labels": ["definition_query"], "metric": "nav",
      "aggregation": None, "time_mode": None}),

    # ── Document summary ───────────────────────────────────────────────────
    ("summarize the march 2026 amfi report",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("give me a summary of the monthly report for january 2026",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("what are the key highlights from the latest amfi report",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": "latest_available_period", "time_mode": "latest"}),
    ("brief me on what happened in the mutual fund industry in february 2026",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("key takeaways from the amfi data for december 2025",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("overview of the mutual fund landscape in q4 2025",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("what does the amfi report say about large cap funds in 2026",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),

    # ── Document lookup ────────────────────────────────────────────────────
    ("get the monthly report for march 2026",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("fetch the amfi data for january 2025",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("show me the repository for may 2026",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),

    # ── AMFI report content queries (lookup) ───────────────────────────────
    ("what did amfi report about the mutual fund industry in march 2023",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("what were the highlights of amfi monthly data for june 2022",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("what information was published in the amfi report for october 2023",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("what were the notable data points in amfi monthly data for september 2024",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),
    ("what were the key trends in the amfi monthly report for january 2024",
     {"intent_type": "lookup", "labels": ["document_summary"], "metric": None,
      "aggregation": None, "time_mode": "snapshot"}),

    # ── Monthly trend / pattern queries ────────────────────────────────────
    ("show monthly redemption pattern for small cap fund in 2023",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "redemption",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "monthly"}),
    ("track month-over-month aum of mid cap fund from january to december 2022",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "month_over_month", "time_mode": "time_series", "granularity": "monthly"}),
    ("show month-by-month aum of large cap fund in 2024",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "monthly"}),
    ("how has the number of folios in small cap fund grown from 2021 to 2025",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "folios",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "annual",
      "needs_analytical": True}),
    ("how did flexi cap fund aum evolve from 2021 to 2026",
     {"intent_type": "trend", "labels": ["trend_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "time_series", "granularity": "annual",
      "needs_analytical": True}),

    # ── Single-answer ranking queries (factual, not tabular list) ──────────
    ("which fund category had the most number of schemes in june 2024",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "schemes",
      "aggregation": "rank", "time_mode": "snapshot"}),
    ("which month in 2024 saw the highest large cap fund aum",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "aum",
      "aggregation": "rank", "time_mode": "snapshot"}),
    ("which quarter had the lowest redemption for debt funds in 2023",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "redemption",
      "aggregation": "rank", "time_mode": "snapshot"}),

    # ── Fiscal year ────────────────────────────────────────────────────────
    ("what is the total aum in fy2024",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("how much was mobilized in large cap in fy25",
     {"intent_type": "factual", "labels": ["snapshot_query", "numeric_aggregation"],
      "metric": "funds_mobilized", "aggregation": "sum", "time_mode": "snapshot"}),
    ("folio count for large cap funds in fy2026",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "folios",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),

    # ── AMC-specific ───────────────────────────────────────────────────────
    ("what is the aum of hdfc large cap fund as of march 2026",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "aum",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("sbi bluechip fund folio count",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "folios",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
    ("mirae asset large cap fund nav",
     {"intent_type": "factual", "labels": ["snapshot_query"], "metric": "nav",
      "aggregation": "point_in_time", "time_mode": "snapshot"}),
]

# ── Stage 3: LLM prompt ───────────────────────────────────────────────────────

_INTENT_SYSTEM = """\
You are a financial query intent classifier for Indian mutual fund data (AMFI monthly reports).
Classify the user query into a structured JSON intent. Reply ONLY with valid JSON — no markdown, no explanation.\
"""

_INTENT_USER_TMPL = """\
Query: {query}

Classify the query as JSON with EXACTLY these keys:
{{
    "labels": ["<ALL applicable labels from the valid set>"],
    "primary_label": "<single primary routing label>",
    "metric": "<folios|nav|sip|redemption|aum|schemes|funds_mobilized|null>",
    "fund_category": "<large_cap|mid_cap|small_cap|flexi_cap|equity|debt|hybrid|liquid|null>",
    "aggregation": "<sum|point_in_time|rank|cagr|year_over_year|month_over_month|latest_available_period|null>",
    "time_mode": "<snapshot|range|latest|yoy|mom|time_series|null>",
    "granularity": "<monthly|quarterly|annual|null>",
    "start_period": "<YYYY-MM or null>",
    "end_period": "<YYYY-MM or null>",
    "comparison_period": "<previous_year|previous_month|same_period_last_year|null>",
    "amc_names": ["<AMC canonical name>"],
    "requires_clarification": <true|false>,
    "needs_analytical": <true|false>,
    "confidence": <0.0–1.0>
}}

Valid labels (use ALL that apply — this is multi-label):
  definition_query    — asks what something means or how it works
  numeric_aggregation — asks for sum/total/cumulative over a time range
  snapshot_query      — asks for a value at a specific point in time
  trend_query         — asks how a metric changed over time
  comparison_query    — compares two or more entities or time periods
  document_summary    — asks for a report summary, overview, or highlights
  table_lookup        — asks for a structured breakdown, list, or ranking
  investment_advice   — asks for investment recommendations (flag for guardrail)
  out_of_scope        — unrelated to Indian mutual fund data

Time/aggregation rules:
  time_mode=range + needs_analytical=true when BOTH a start year and end year appear
  time_mode=yoy   when "year-on-year", "YoY", "vs last year", "prior year" appear
  time_mode=mom   when "month-on-month", "MoM", "vs last month" appear
  time_mode=snapshot when "as of", "as on", "in [specific month]" appear
  time_mode=latest   when "latest", "current", "recent", "now" appear
  time_mode=time_series when "trend", "over the years", "historical" appear
  aggregation=year_over_year for YoY; aggregation=month_over_month for MoM
  aggregation=latest_available_period for "latest"/"current" queries
  start_period/end_period: ISO "YYYY-MM" — use month=01 for year-start, 12 for year-end
  granularity=annual for year-wise queries; granularity=monthly for month-wise queries
  comparison_period: set for YoY (→ previous_year), MoM (→ previous_month),
    or same-month-prior-year (→ same_period_last_year)
  requires_clarification=true when the query is ambiguous about time period or metric

Metric rules:
  metric=funds_mobilized for "investment", "mobilized", "inflow", "mobilisation"
  metric=folios for "folio count", "number of investors", "investor accounts"
  aggregation=sum for "total", "cumulative", "year wise", "annual total"
  aggregation=rank for "highest", "top N", "best", "largest"

AMFI report columns: Scheme Name | No. of Schemes | No. of Folios |
Funds Mobilized | Repurchase/Redemption | Net Inflow/Outflow | Net Assets (AUM) | Avg AUM\
"""


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class QueryIntent:
    """Structured representation of what the user is asking for."""

    raw_query: str

    # ── Temporal filters ────────────────────────────────────────────────────
    year:    int | None = None
    month:   int | None = None
    quarter: str | None = None        # "Q1" … "Q4"

    year_from: int | None = None      # multi-year range start
    year_to:   int | None = None      # multi-year range end

    start_period: str | None = None   # ISO "YYYY-MM", month-precision (from LLM)
    end_period:   str | None = None   # ISO "YYYY-MM", month-precision (from LLM)

    fiscal_year: int | None = None    # calendar year of FY-end: FY24 → 2024

    # ── Document filters ─────────────────────────────────────────────────────
    category: str | None = None       # "monthly" | "quarterly"

    # ── Primary intent ────────────────────────────────────────────────────────
    intent_type: str = "factual"
    # Values: factual | trend | tabular | comparison | definition | lookup | analytical

    # ── Multi-label classification ────────────────────────────────────────────
    labels: list[str] = field(default_factory=list)
    # Values from VALID_LABELS: definition_query, numeric_aggregation, snapshot_query,
    # trend_query, comparison_query, document_summary, table_lookup,
    # investment_advice, out_of_scope

    secondary_intent: str | None = None  # kept for backward compat; prefer labels[]

    # ── Metric ────────────────────────────────────────────────────────────────
    metric: str | None = None
    # Values: aum | nav | folios | funds_mobilized | redemption | sip | schemes

    fund_category: str | None = None
    # Values: large_cap | mid_cap | small_cap | equity | debt | hybrid | liquid | etc.

    # ── Aggregation ───────────────────────────────────────────────────────────
    aggregation: str | None = None
    # Values: sum | point_in_time | rank | cagr
    #         | year_over_year | month_over_month | latest_available_period

    # ── Time mode (Step 1 / slot filling) ────────────────────────────────────
    time_mode: str | None = None
    # Values: snapshot | range | latest | yoy | mom | time_series

    # ── Granularity (slot filling) ────────────────────────────────────────────
    granularity: str | None = None
    # Values: monthly | quarterly | annual | daily

    # ── Comparison period (slot filling) ─────────────────────────────────────
    comparison_period: str | None = None
    # Values: previous_year | previous_month | same_period_last_year

    # ── Financial entities ────────────────────────────────────────────────────
    scheme_types:    list[str] = field(default_factory=list)
    financial_terms: list[str] = field(default_factory=list)
    amc_names:       list[str] = field(default_factory=list)

    # ── Routing hints ─────────────────────────────────────────────────────────
    needs_chunks:     bool = True
    needs_tables:     bool = False
    needs_documents:  bool = False
    time_sensitive:   bool = False
    prefer_recent:    bool = False
    needs_analytical: bool = False        # True → route to analytical agent loop
    needs_reasoning:  bool = False        # True → route to reasoning_engine (causal "why" queries)
    requires_clarification: bool = False  # True → ask user for clarification

    # ── Pipeline metadata ─────────────────────────────────────────────────────
    search_query: str  = ""
    confidence:   float = 0.0
    ambiguous:    bool  = False
    stage_used:   str   = "rule"   # "rule" | "embedding" | "llm"

    # ── Ontology resolver output (fallback signal, see ontology_resolver.py) ──
    canonical: "CanonicalQuery | None" = None

    def describe(self) -> str:
        parts = [f"intent={self.intent_type}"]
        if self.labels:
            parts.append(f"labels={self.labels}")
        if self.secondary_intent:
            parts.append(f"secondary={self.secondary_intent}")
        if self.metric:
            parts.append(f"metric={self.metric}")
        if self.fund_category:
            parts.append(f"fund_category={self.fund_category}")
        if self.aggregation:
            parts.append(f"agg={self.aggregation}")
        if self.time_mode:
            parts.append(f"time_mode={self.time_mode}")
        if self.granularity:
            parts.append(f"granularity={self.granularity}")
        if self.comparison_period:
            parts.append(f"cmp_period={self.comparison_period}")
        if self.start_period and self.end_period:
            parts.append(f"period={self.start_period}→{self.end_period}")
        elif self.year_from:
            parts.append(f"range={self.year_from}-{self.year_to}")
        elif self.year:
            parts.append(f"year={self.year}")
        if self.month:
            parts.append(f"month={self.month}")
        if self.fiscal_year:
            parts.append(f"fy={self.fiscal_year}")
        if self.scheme_types:
            parts.append(f"schemes={self.scheme_types}")
        if self.amc_names:
            parts.append(f"amc={self.amc_names}")
        if self.requires_clarification:
            parts.append("needs_clarification=true")
        parts.append(f"stage={self.stage_used}({self.confidence:.2f})")
        return "  ".join(parts)


# ── Stage 2: Embedding similarity classifier ─────────────────────────────────

class _EmbeddingClassifier:
    """Top-K cosine-similarity classifier against pre-embedded labelled examples.

    Uses the same SentenceTransformer already loaded by the Retriever — no
    additional model download or memory cost.

    Top-3 weighted voting reduces noise from single nearest-neighbor misses.
    Labels are merged as a union across all top-K hits.
    """

    def __init__(self, embed_fn: Callable[[str], list[float]]) -> None:
        import numpy as np
        self._embed_fn = embed_fn
        texts  = [ex[0] for ex in _LABELED_EXAMPLES]
        labels = [ex[1] for ex in _LABELED_EXAMPLES]
        self._labels = labels
        vecs = [np.array(embed_fn(t), dtype=np.float32) for t in texts]
        self._matrix = np.stack(vecs)
        log.info("intent.embedding_classifier.ready", n_examples=len(texts))

    def classify(self, query: str, k: int = 3) -> tuple[dict, float]:
        """Return (merged_label_dict, top_similarity) using top-K weighted voting."""
        import numpy as np
        q_vec = np.array(self._embed_fn(query), dtype=np.float32)
        sims  = self._matrix @ q_vec

        top_k       = min(k, len(self._labels))
        top_indices = list(sims.argsort()[-top_k:][::-1])
        top_sims    = [float(sims[i]) for i in top_indices]

        # Weighted vote on intent_type
        intent_votes: dict[str, float] = {}
        for idx, sim in zip(top_indices, top_sims):
            it = self._labels[idx].get("intent_type", "factual")
            intent_votes[it] = intent_votes.get(it, 0.0) + sim

        winner_intent = max(intent_votes, key=intent_votes.__getitem__)

        # Merge fields from all top-K hits; labels[] is a union
        merged: dict = {
            "intent_type":     winner_intent,
            "labels":          [],
            "needs_analytical": False,
        }
        for idx in top_indices:
            lbl = self._labels[idx]
            # Scalar fields: first winner-intent hit wins
            if lbl.get("intent_type") == winner_intent:
                for key in ("metric", "aggregation", "time_mode",
                            "granularity", "comparison_period"):
                    if not merged.get(key) and lbl.get(key):
                        merged[key] = lbl[key]
            # Labels: union across all top-K
            for label in (lbl.get("labels") or []):
                if label not in merged["labels"]:
                    merged["labels"].append(label)
            if lbl.get("needs_analytical"):
                merged["needs_analytical"] = True

        return merged, top_sims[0]


# ── Stage 3: LLM structured extractor ────────────────────────────────────────

class _LLMExtractor:
    """Single Claude Haiku call returning structured JSON intent."""

    def __init__(self, generator) -> None:
        self._generator = generator

    def extract(self, query: str) -> tuple[dict, float]:
        """Return (parsed_dict, confidence). Returns ({}, 0.0) on failure."""
        try:
            messages = [
                {"role": "system", "content": _INTENT_SYSTEM},
                {"role": "user",   "content": _INTENT_USER_TMPL.format(query=query)},
            ]
            result = self._generator.generate(
                messages,
                intent_type="factual",
                max_tokens=200,
            )
            raw    = re.sub(r"```(?:json)?\s*|\s*```", "",
                            result.answer.strip()).strip()
            parsed = json.loads(raw)
            conf   = float(parsed.pop("confidence", 0.70))
            return parsed, conf
        except Exception as exc:
            log.warning("intent.llm_extractor.failed", error=str(exc))
            return {}, 0.0


# ── Main QueryAnalyzer ────────────────────────────────────────────────────────

class QueryAnalyzer:
    """Converts a raw query string into a structured QueryIntent.

    Instantiate once at startup with optional embed_fn and generator to
    enable stages 2 and 3.  Without them the analyzer falls back to pure
    rule-based classification (stage 1 only).
    """

    def __init__(
        self,
        embed_fn:  Callable[[str], list[float]] | None = None,
        generator = None,
    ) -> None:
        self._emb_cls = _EmbeddingClassifier(embed_fn) if embed_fn else None
        self._llm_ext = _LLMExtractor(generator)       if generator else None

    # ── Public API ────────────────────────────────────────────────────────────

    @staticmethod
    def _enforce_temporal_invariant(intent: "QueryIntent") -> "QueryIntent":
        """needs_analytical routes to the plan_years month-by-month loop, which
        has nothing to iterate over without an actual period. Any stage
        (rule/embedding/LLM) can set needs_analytical=True independently —
        this is a single backstop applied at every exit point of analyze()
        rather than re-guarding each stage's merge logic separately."""
        if intent.needs_analytical and not (intent.year or intent.year_from or intent.month):
            intent.needs_analytical = False
        return intent

    def analyze(self, query: str) -> QueryIntent:
        """Run the 5-stage pipeline and return a QueryIntent."""

        intent = self._stage1(query)

        log.debug("intent.stage1",
                  intent_type=intent.intent_type, metric=intent.metric,
                  aggregation=intent.aggregation, time_mode=intent.time_mode,
                  granularity=intent.granularity, confidence=intent.confidence)

        if intent.confidence >= _CONF_THRESHOLD_RULE:
            intent.stage_used = "rule"
            log.info("intent.resolved", stage="rule",
                     intent_type=intent.intent_type, confidence=intent.confidence)
            return self._enforce_temporal_invariant(intent)

        # ── Stage 2: Embedding similarity classifier ──────────────────────
        emb_result, emb_conf = {}, 0.0
        if self._emb_cls:
            emb_result, emb_conf = self._emb_cls.classify(
                intent.search_query or query
            )
            intent = self._merge_embedding(intent, emb_result, emb_conf)
            log.debug("intent.stage2",
                      emb_intent=emb_result.get("intent_type"),
                      emb_conf=round(emb_conf, 3))

            stages_agree = (
                not emb_result
                or emb_result.get("intent_type") == intent.intent_type
            )
            if emb_conf >= _CONF_THRESHOLD_EMB and stages_agree:
                self._apply_policy_overrides(query.lower().strip(), intent)
                intent.stage_used = "embedding"
                log.info("intent.resolved", stage="embedding",
                         intent_type=intent.intent_type, confidence=intent.confidence)
                return self._enforce_temporal_invariant(intent)

        # ── Stage 3: LLM structured extractor ────────────────────────────
        if self._llm_ext:
            llm_result, llm_conf = self._llm_ext.extract(query)
            if llm_result:
                intent = self._merge_llm(intent, llm_result, llm_conf)
                self._apply_policy_overrides(query.lower().strip(), intent)
                intent.stage_used = "llm"
                log.info("intent.resolved", stage="llm",
                         intent_type=intent.intent_type, confidence=intent.confidence)
                return self._enforce_temporal_invariant(intent)

        # ── Stage 4 / 5: Fallback ─────────────────────────────────────────
        intent.ambiguous  = intent.confidence < _CONF_THRESHOLD_RULE
        intent.stage_used = "rule"
        log.info("intent.resolved", stage="rule_fallback",
                 intent_type=intent.intent_type,
                 confidence=intent.confidence,
                 ambiguous=intent.ambiguous)
        return self._enforce_temporal_invariant(intent)

    # ── Stage 1 internals ────────────────────────────────────────────────────

    def _stage1(self, query: str) -> QueryIntent:
        intent = QueryIntent(raw_query=query, search_query=query)
        intent.canonical = resolve_ontology(query)
        q = query.lower().strip()

        self._detect_policy_labels(q, intent)
        self._detect_causal_reasoning(q, intent)
        self._extract_year_range(q, intent)   # must run before _extract_temporal
        self._extract_temporal(q, intent)
        self._extract_fiscal_year(q, intent)
        self._extract_metric(q, intent)
        self._classify_intent(q, intent)
        self._extract_aggregation(q, intent)
        self._extract_time_mode(q, intent)
        self._extract_granularity(q, intent)
        self._extract_entities(q, intent)
        self._set_routing_hints(q, intent)
        self._build_search_query(query, q, intent)
        self._build_labels(q, intent)         # must run after _classify_intent
        self._fix_year_range_intent(q, intent) # override analytical/comparison for ranges
        self._fix_comparison_db_metric(intent) # comparison + DB metric → tabular
        # Single-year "month-by-month" trend queries never hit the year_from/year_to
        # guard in _fix_year_range_intent, so needs_analytical is synced here too —
        # any final "trend" classification needs the per-month analytical loop.
        # But the loop needs an actual period to iterate over (plan_months ->
        # retrieve_month -> extract_month_metric) — a "trend" classification
        # with zero temporal signal (e.g. causal "why did X change" phrasing
        # with no year/month mentioned) has nothing to analyze and previously
        # routed to plan_years anyway, producing an abstention every time.
        has_temporal_signal = bool(intent.year or intent.year_from or intent.month)
        if intent.intent_type == "trend" and "investment_advice" not in intent.labels:
            intent.needs_analytical = has_temporal_signal
        self._apply_policy_overrides(q, intent)
        intent.confidence = self._compute_confidence(q, intent)
        return intent

    def _detect_policy_labels(self, q: str, intent: QueryIntent) -> None:
        """Detect policy-sensitive query families before semantic slot filling.

        Performance-ranking language is especially easy to misroute as a
        numeric aggregation. Mark it early so downstream guardrails and merges
        keep it out of the analytical path unless a future product explicitly
        supports factual fund-return rankings.
        """
        performance_advice = (
            any(sig in q for sig in PERFORMANCE_ADVICE_SIGNALS)
            or bool(re.search(r"\bbest\s+(performing\s+)?funds?\b", q))
            or bool(re.search(r"\bfunds?\b.{0,60}\b(cagr|returns?)\b.{0,60}\b(best|good|high|strong)\b", q))
            or bool(re.search(r"\b(best|good|high|strong)\b.{0,60}\b(cagr|returns?)\b.{0,60}\bfunds?\b", q))
        )
        if performance_advice:
            intent.labels.append("investment_advice")
            intent.aggregation = "cagr"
            intent.requires_clarification = True
            intent.needs_analytical = False
            intent.needs_tables = False
            intent.needs_chunks = True

    def _detect_causal_reasoning(self, q: str, intent: QueryIntent) -> None:
        """"Why did X change" queries need the reasoning engine (real
        computed metric directions matched against semantic/reasoning_rules.yaml),
        not RAG or SQL alone — AMFI source documents contain no explanatory
        prose (nothing ever states market appreciation caused an AUM move),
        and a raw SQL query can't produce a causal "because" statement by
        itself. Requires >=1 resolved metric (via the ontology resolver,
        already run before this) so a generic "why is this happening" with
        no financial-metric anchor doesn't get misrouted into a reasoning
        path with nothing to reason about.
        """
        if "investment_advice" in intent.labels:
            return
        if not any(sig in q for sig in CAUSAL_SIGNALS):
            return
        if not (intent.canonical and intent.canonical.metrics):
            return
        from financial_pipeline.config import settings
        if settings.enable_reasoning_engine:
            intent.needs_reasoning = True

    def _extract_year_range(self, q: str, intent: QueryIntent) -> None:
        if "investment_advice" in intent.labels:
            return
        # Matches: "2020 to 2026", "2020-2026", "between 2020 and 2026".
        # 20\d\d (2000-2099) rather than 20[0-2]\d (2000-2029) — the narrower
        # bound silently failed to extract genuinely-future years like 2030/
        # 2050 used by out-of-range guardrail test queries.
        m = (re.search(r'\b(20\d\d)\s*(?:to|through|–|-|versus|vs\.?)\s*(20\d\d)\b', q)
             or re.search(r'\bbetween\s+(20\d\d)\s+and\s+(20\d\d)\b', q))
        if m:
            y1, y2 = int(m.group(1)), int(m.group(2))
            if y1 != y2:
                intent.year_from        = min(y1, y2)
                intent.year_to          = max(y1, y2)
                intent.year             = None
                intent.needs_analytical = True
                intent.prefer_recent    = False
                intent.time_sensitive   = True
                # intent_type set later in _classify_intent or _stage1 post-step

    def _extract_temporal(self, q: str, intent: QueryIntent) -> None:
        if not (intent.year_from and intent.year_to):
            ym = re.search(r"\b(20\d\d)\b", q)
            if ym:
                intent.year = int(ym.group(1))
                intent.time_sensitive = True

        for name, num in MONTH_MAP.items():
            if re.search(r"\b" + re.escape(name) + r"\b", q):
                intent.month = num
                intent.time_sensitive = True
                break

        qm = re.search(r"\b(q[1-4])\b", q)
        if qm:
            intent.quarter = qm.group(1).upper()
            intent.time_sensitive = True

        # Ordinal quarter words: "third quarter" → Q3
        if not intent.quarter:
            _ORDINAL_Q = {"first": 1, "second": 2, "third": 3, "fourth": 4,
                          "1st": 1, "2nd": 2, "3rd": 3, "4th": 4}
            oq = re.search(r"\b(first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\b", q)
            if oq:
                n = _ORDINAL_Q[oq.group(1)]
                intent.quarter = f"Q{n}"
                intent.time_sensitive = True

        # When quarter is set but month is not, resolve to the first month of that quarter
        # so DB metadata/retrieval queries can use a concrete period_month filter.
        if intent.quarter and not intent.month:
            _Q_FIRST_MONTH = {"Q1": 1, "Q2": 4, "Q3": 7, "Q4": 10}
            first = _Q_FIRST_MONTH.get(intent.quarter)
            if first:
                intent.month = first

        if re.search(r"\b(latest|recent|current|today|now)\b", q) and not intent.year:
            intent.prefer_recent = True

        if (intent.year and not intent.month and not intent.quarter
                and not intent.needs_analytical):
            intent.prefer_recent = True

        if "monthly" in q or "month of" in q:
            intent.category = "monthly"
        elif "quarterly" in q or "quarter" in q:
            intent.category = "quarterly"

    def _extract_fiscal_year(self, q: str, intent: QueryIntent) -> None:
        m = re.search(r'\bfy\s*(\d{2}(?:\d{2})?)\b', q)
        if m:
            raw = m.group(1)
            fy  = int(raw) if len(raw) == 4 else (2000 + int(raw))
            intent.fiscal_year    = fy
            intent.time_sensitive = True
            if not intent.year:
                intent.year = fy

    # Metrics that live in the structured DB tables and should route to SQL
    _DB_METRICS = frozenset({"aum", "avg_aum", "net_inflow", "funds_mobilized", "redemption", "folios", "schemes"})

    # Keywords that confirm "show how X evolved" — strong enough to keep trend intent
    # even for year-range queries. "trend" alone qualifies UNLESS "by year/month" negates it.
    # Use "grew" not "grow" to avoid substring-matching "growth" (tabular queries often say
    # "AUM growth from X to Y" meaning list the values, not show the pattern).
    _TREND_TYPE_KWS = frozenset({
        "pattern", "track",
        "month-by-month", "month by month", "year by year", "year-by-year",
        "year wise", "year-wise",
        "year on year", "year over year", "yoy",
        "evolved", "evolution", "trajectory",
        "grew", "grown", "shrink", "declined",
        "evolve", "evolving",
    })

    def _classify_intent(self, q: str, intent: QueryIntent) -> None:
        if intent.needs_analytical:
            intent.needs_chunks = True
            # Don't return early — let trend/tabular signals determine intent_type below

        # Pre-compute before signal_counts so exclusions work correctly
        is_explain_diff = bool(
            (re.search(r"\b(explain|describe)\b", q) and "difference between" in q)
            or re.search(r"\bwhat is the difference between\b", q)
        )

        # "number of" is a TABLE_SIGNAL (for "show number of folios by category"
        # style listing requests), but it's also a literal substring of the
        # count-metric canonical names ("Number of Folios"/"Number of Schemes").
        # That makes it spuriously fire for "What is Number of Folios?"
        # (definitional) or "Which report contains Number of Schemes..."
        # (document lookup) — both pre-empted by the table branch since it's
        # checked before DEFINITION_SIGNALS/LOOKUP_SIGNALS below. Drop it from
        # the table signal specifically when one of those stronger shapes
        # (bare "what is X?", or a "which report/document" lookup) is present.
        _is_definition_or_lookup_shape = (
            bool(re.search(r"^\s*what is\b", q)) or any(kw in q for kw in LOOKUP_SIGNALS)
        )
        _table_kws = (TABLE_SIGNALS - {"number of"}) if _is_definition_or_lookup_shape else TABLE_SIGNALS

        signal_counts = {
            # Exclude "explain the difference between" from comparison
            "comparison":  any(kw in q for kw in COMPARISON_SIGNALS) and not is_explain_diff,
            "trend":       any(kw in q for kw in TREND_SIGNALS),
            "yoy":         any(kw in q for kw in YOY_SIGNALS),
            "mom":         any(kw in q for kw in MOM_SIGNALS),
            "ranking":     any(kw in q for kw in RANKING_SIGNALS),
            "table":       any(kw in q for kw in _table_kws),
            "doc_summary": any(kw in q for kw in DOCUMENT_SUMMARY_SIGNALS),
            "regulatory":  any(kw in q for kw in REGULATORY_SIGNALS),
        }

        if signal_counts["comparison"]:
            intent.intent_type  = "comparison"
            intent.needs_chunks = True
            if signal_counts["trend"] or signal_counts["yoy"] or signal_counts["mom"]:
                intent.secondary_intent = "trend"
            elif signal_counts["table"]:
                intent.secondary_intent = "tabular"

        elif signal_counts["yoy"] or signal_counts["mom"]:
            intent.intent_type  = "trend"
            intent.needs_chunks = True
            intent.needs_tables = True

        # Doc summary checked before trend: "key trends in the AMFI report" is a report
        # content query, not a trend analysis — doc signal must win over trend keywords.
        elif signal_counts["doc_summary"]:
            intent.intent_type     = "lookup"
            intent.needs_documents = True
            intent.needs_chunks    = True

        elif signal_counts["trend"]:
            intent.intent_type  = "trend"
            intent.needs_chunks = True
            intent.needs_tables = True
            if signal_counts["table"]:
                intent.secondary_intent = "tabular"

        elif signal_counts["ranking"]:
            # "top N" / "rank all" → tabular list; "which X had the most/highest" → factual
            is_list_ranking = bool(re.search(r'\btop\s+\d+\b', q) or re.search(r'\b(rank all|all .{0,20} ranked)\b', q))
            intent.intent_type  = "tabular" if is_list_ranking else "factual"
            intent.needs_tables = True
            intent.needs_chunks = True
            intent.aggregation  = intent.aggregation or "rank"

        elif signal_counts["table"]:
            intent.intent_type  = "tabular"
            intent.needs_tables = True
            intent.needs_chunks = True

        elif signal_counts["regulatory"]:
            intent.intent_type  = "regulatory"
            intent.needs_chunks = True

        elif any(re.search(r"\b" + re.escape(kw) + r"\b", q)
                 for kw in DEFINITION_SIGNALS):
            is_db_metric = intent.metric in self._DB_METRICS
            is_value_query = re.search(r"\bwhat is the\b", q) and intent.metric is not None
            # Conceptual comparisons ("difference between X and Y") are definitional,
            # not numeric lookups — gate is_value_query so they stay as definition.
            is_comparison_def = bool(re.search(r"\bdifference between\b", q))
            # Derived metrics (ratio, percentage, per-folio, per-scheme) are answered
            # via RAG context, not a direct DB column lookup → factual, not tabular
            is_derived = bool(re.search(
                r"\b(ratio|percentage|per folio|per scheme|average per|rate of"
                r"|proportion|share of total)\b", q))
            is_latest = any(kw in q for kw in LATEST_SIGNALS)
            if is_db_metric and (intent.year or intent.year_from or is_latest) and not is_derived:
                # Specific numeric lookup from structured DB — use tabular, not factual.
                # "latest"/"current" phrasing (no explicit year) is just as much a
                # point-in-time DB read as naming a year outright.
                intent.intent_type  = "tabular"
                intent.needs_tables = True
                intent.needs_chunks = False
            elif (is_value_query or is_derived) and not is_comparison_def:
                intent.intent_type  = "factual"
                intent.needs_chunks = True
            else:
                intent.intent_type  = "definition"
                intent.needs_chunks = True

        elif any(re.search(r"\b" + re.escape(kw), q) for kw in LOOKUP_SIGNALS):
            intent.intent_type     = "lookup"
            intent.needs_documents = True
            intent.needs_chunks    = True

        elif (intent.year or any(kw in q for kw in LATEST_SIGNALS)) and intent.metric:
            # Structured point-in-time DB lookup: a specific year OR "current"/
            # "latest" phrasing, plus a known metric. Both resolve to a single
            # row/most-recent-row DB read — pure numeric retrieval, SQL always
            # outperforms document search here. (intent.time_mode isn't set
            # yet at this point in the pipeline — _extract_time_mode runs
            # after _classify_intent — so LATEST_SIGNALS is checked directly
            # against the query text rather than intent.time_mode.)
            # Exception: derived/ratio queries ("percentage of", "share of") need
            # RAG context rather than a raw DB column, so classify factual.
            is_derived = bool(re.search(
                r"\b(ratio|percentage|per folio|per scheme|average per|rate of"
                r"|proportion|share of total)\b", q))
            if is_derived:
                intent.intent_type  = "factual"
                intent.needs_chunks = True
            else:
                intent.intent_type  = "tabular"
                intent.needs_tables = True
                intent.needs_chunks = False

        elif intent.year and re.search(r"\bhow many\b", q) and re.search(r"\bschemes?\b", q):
            # "How many schemes in category X in year Y" → SQL COUNT on amfi_fund_stats.
            # Only fires when "scheme" is present; NFO/folio counts go through RAG.
            intent.intent_type  = "tabular"
            intent.needs_tables = True
            intent.needs_chunks = False

        else:
            intent.intent_type  = "factual"
            intent.needs_chunks = True

    def _fix_comparison_db_metric(self, intent: QueryIntent) -> None:
        """Comparison intent + known DB metric → tabular.

        The ground truth corpus has no 'comparison' class. Queries that compare
        DB metric values side by side (e.g. 'compare equity vs income AUM in 2017')
        are tabular data requests, not narrative comparisons.
        Only applies when year/year_range is set (ensures it's a DB query).
        """
        if intent.intent_type != "comparison":
            return
        if intent.metric not in self._DB_METRICS:
            return
        if not (intent.year or intent.year_from):
            return
        intent.intent_type  = "tabular"
        intent.needs_tables = True
        intent.needs_chunks = False

    def _fix_year_range_intent(self, q: str, intent: QueryIntent) -> None:
        """Post-step: year-range queries must be tabular or trend, never analytical/comparison.

        The ground truth labels have no 'analytical' or 'comparison' class.
        All DB year-range queries should be 'trend' if showing change/growth pattern,
        or 'tabular' if just listing values organized by time period.

        Trend qualifier: "trend" keyword (unless "by year/month" overrides it as a
        format specifier) OR any strong trend-type keyword (_TREND_TYPE_KWS).
        TREND_SIGNALS in _classify_intent fires too broadly (includes "each year",
        "from 20", etc.), so we re-adjudicate here for all year-range queries.
        """
        if not (intent.year_from and intent.year_to):
            return
        if "investment_advice" in intent.labels:
            intent.needs_analytical = False
            return
        # "factual" is included here because _classify_intent's final catch-all
        # (no other signal matched) defaults to "factual" — for a query that
        # already has a resolved year range, that's an unclassified tabular/
        # trend request, not a deliberate factual judgment. (is_derived
        # ratio/percentage queries also produce "factual", but those are rare
        # in combination with an explicit year range and get correctly
        # reclassified as tabular below anyway, which is the safer of the two
        # outcomes for a derived-metric-over-a-range query.)
        if intent.intent_type == "tabular":
            # Already correctly classified (e.g. via a direct TABLE_SIGNAL
            # match, not the ambiguous catch-all) — no reclassification
            # needed, but needs_analytical still defaults to True from
            # _extract_year_range's blanket year-span check and must be
            # corrected here, same as the comparison/tabular branches below.
            intent.needs_analytical = False
            return
        if intent.intent_type not in ("analytical", "comparison", "trend", "factual"):
            return

        # Comparison-type year-range queries (e.g. "compare X vs Y from 2022 to 2024")
        # are side-by-side data requests — always tabular, never trend.
        if intent.intent_type == "comparison":
            intent.intent_type      = "tabular"
            intent.needs_tables     = True
            intent.needs_chunks     = False
            # Side-by-side data request — the DB (via query_sql) already covers
            # the full range in one call; the month-by-month analytical loop
            # was left on from _extract_year_range's blanket year-span check.
            intent.needs_analytical = False
            return

        # "trend" word qualifies as a strong signal UNLESS "by year" or "by month"
        # immediately specifies a tabular format (e.g. "AUM trend by year")
        trend_word = ("trend" in q) and not ("by year" in q or "by month" in q)
        strong_trend = trend_word or any(kw in q for kw in self._TREND_TYPE_KWS)

        if strong_trend:
            intent.intent_type      = "trend"
            intent.needs_chunks     = True
            intent.needs_tables     = True
            intent.needs_analytical = True
        else:
            intent.intent_type      = "tabular"
            intent.needs_tables     = True
            intent.needs_chunks     = False
            # Plain "list the values" year-range request — query_sql (Vanna)
            # already handles this via GROUP BY / UNION ALL; the analytical
            # month-by-month loop is unnecessary and 10-50x more expensive.
            intent.needs_analytical = False

    def _extract_metric(self, q: str, intent: QueryIntent) -> None:
        # "how many <category words> schemes" ("how many mutual fund schemes
        # were there") isn't caught by the fixed "how many schemes" phrase —
        # category words can sit between "how many" and "schemes".
        if re.search(r"how many\b.{0,25}\bschemes?\b", q):
            intent.metric = "schemes"
            return

        for metric, keywords in _METRIC_MAP.items():
            if any(kw in q for kw in keywords):
                # avg_aum is a real per-row DB column (avg daily AUM within one
                # month), distinct from the aum column, but only when the query
                # targets a single period — "average AUM across 2024" or "...from
                # 2020 to 2026" is aggregation=average applied over the aum column
                # across many rows, not a request for the avg_aum column itself.
                # "current"/"latest" (no explicit month) is just as single-period
                # as naming a month outright.
                is_single_period = (
                    (bool(intent.month) or any(kw in q for kw in LATEST_SIGNALS))
                    and not (intent.year_from and intent.year_to)
                )
                if metric == "aum" and is_single_period and ("average aum" in q or "avg aum" in q):
                    intent.metric = "avg_aum"
                else:
                    intent.metric = metric
                return

        # Ontology fallback: only reached when _METRIC_MAP found nothing.
        if intent.canonical and intent.canonical.metrics:
            intent.metric = intent.canonical.metrics[0]

    def _extract_aggregation(self, q: str, intent: QueryIntent) -> None:
        if intent.aggregation:
            return
        for agg, keywords in _AGGREGATION_MAP.items():
            if any(kw in q for kw in keywords):
                intent.aggregation = agg
                return

    def _extract_time_mode(self, q: str, intent: QueryIntent) -> None:
        """Set intent.time_mode from keyword signals."""
        if intent.needs_analytical:
            intent.time_mode = "range"
            return

        if any(kw in q for kw in YOY_SIGNALS):
            intent.time_mode        = "yoy"
            intent.comparison_period = intent.comparison_period or "previous_year"
            return

        if any(kw in q for kw in MOM_SIGNALS):
            intent.time_mode        = "mom"
            intent.comparison_period = intent.comparison_period or "previous_month"
            return

        if any(kw in q for kw in LATEST_SIGNALS) and not intent.year:
            intent.time_mode = "latest"
            return

        if any(kw in q for kw in SNAPSHOT_SIGNALS) or (intent.year and intent.month):
            intent.time_mode = "snapshot"
            return

        if any(kw in q for kw in TREND_SIGNALS):
            intent.time_mode = "time_series"
            return

        if intent.year:
            intent.time_mode = "snapshot"

    def _extract_granularity(self, q: str, intent: QueryIntent) -> None:
        """Set intent.granularity from keyword signals or infer from time_mode."""
        for gran, keywords in _GRANULARITY_MAP.items():
            if any(kw in q for kw in keywords):
                intent.granularity = gran
                return

        # Infer from time_mode and aggregation
        if intent.time_mode == "mom" or intent.aggregation == "month_over_month":
            intent.granularity = "monthly"
        elif intent.time_mode in ("yoy", "range", "time_series") or intent.aggregation == "year_over_year":
            intent.granularity = "annual"
        elif intent.category == "quarterly" or intent.quarter:
            intent.granularity = "quarterly"
        elif intent.category == "monthly":
            intent.granularity = "monthly"

    def _build_labels(self, q: str, intent: QueryIntent) -> None:
        """Populate intent.labels from intent_type and detected signals."""
        existing = [lbl for lbl in intent.labels if lbl in VALID_LABELS]
        base = _INTENT_TO_LABELS.get(intent.intent_type, ["snapshot_query"])
        intent.labels = existing + [lbl for lbl in base if lbl not in existing]

        if intent.secondary_intent:
            for lbl in _INTENT_TO_LABELS.get(intent.secondary_intent, []):
                if lbl not in intent.labels:
                    intent.labels.append(lbl)

        if intent.needs_analytical and "numeric_aggregation" not in intent.labels:
            intent.labels.append("numeric_aggregation")

        if intent.aggregation in ("year_over_year", "month_over_month"):
            if "trend_query" not in intent.labels:
                intent.labels.append("trend_query")

        # Document summary gets priority label placement
        if any(kw in q for kw in DOCUMENT_SUMMARY_SIGNALS):
            if "document_summary" not in intent.labels:
                intent.labels.insert(0, "document_summary")

    def _apply_policy_overrides(self, q: str, intent: QueryIntent) -> None:
        """Keep policy-sensitive queries from being over-normalized."""
        if "investment_advice" not in intent.labels:
            return
        if any(sig in q for sig in PERFORMANCE_ADVICE_SIGNALS) or "cagr" in q or "return" in q:
            intent.metric = None
            intent.aggregation = "cagr"
        intent.intent_type = "factual"
        intent.needs_analytical = False
        intent.needs_tables = False
        intent.needs_chunks = True
        intent.requires_clarification = True

    def _extract_entities(self, q: str, intent: QueryIntent) -> None:
        # Several SCHEME_TYPES entries overlap the same mention by design
        # (e.g. "gold etf" text also contains bare "gold" and "etf"; "index
        # funds" contains "index fund" and "index"). Keep only the longest
        # match per mention rather than appending every substring hit.
        matches = [scheme for scheme in SCHEME_TYPES if scheme in q]
        kept = [m for m in matches if not any(m != other and m in other for other in matches)]
        intent.scheme_types.extend(kept)

        # Ontology fallback: only reached when SCHEME_TYPES found nothing.
        if not kept and intent.canonical and intent.canonical.scheme_types:
            intent.scheme_types.extend(intent.canonical.scheme_types)

        for term in FINANCIAL_TERMS:
            if re.search(r"\b" + re.escape(term) + r"\b", q):
                intent.financial_terms.append(term)

        seen: set[str] = set()
        for pattern, canonical in _AMC_NAMES:
            if pattern in q and canonical not in seen:
                intent.amc_names.append(canonical)
                seen.add(canonical)

        # Ontology fallback: only reached when _AMC_NAMES found nothing.
        if not intent.amc_names and intent.canonical and intent.canonical.amcs:
            intent.amc_names.extend(intent.canonical.amcs)

    def _set_routing_hints(self, q: str, intent: QueryIntent) -> None:
        if re.search(r"\b(how many|number of|count of|total)\b.*(scheme|fund|folio)", q):
            intent.needs_tables = True

        if re.search(r"\b(report for|repository for|data for|document for)\b", q):
            intent.needs_documents = True

        if "aum" in q and any(kw in q for kw in ("total", "breakdown", "category")):
            intent.needs_tables = True

    def _compute_confidence(self, q: str, intent: QueryIntent) -> float:
        """Score stage-1 confidence from 0.0 to 0.90."""
        if "investment_advice" in intent.labels:
            return 0.90
        if intent.intent_type in ("definition", "lookup"):
            return 0.90
        if intent.needs_analytical and intent.intent_type in ("tabular", "trend"):
            # Year-range DB query: deterministic classification — skip embedding stage
            return min(0.88 + (0.02 if intent.metric else 0.0), 0.90)
        if intent.needs_analytical:
            return min(0.88 + (0.02 if intent.metric else 0.0), 0.90)

        # Structural tabular: year + metric is a deterministic DB lookup — skip embedding
        if intent.intent_type == "tabular" and intent.year and intent.metric:
            return 0.88

        score = 0.50
        if intent.metric:
            score += 0.12
        if intent.aggregation and intent.aggregation != "rank":
            score += 0.08
        if intent.scheme_types:
            score += 0.05
        if intent.year or intent.month:
            score += 0.05
        if intent.amc_names:
            score += 0.04
        if intent.fiscal_year:
            score += 0.03
        if intent.time_mode:
            score += 0.04
        if intent.granularity:
            score += 0.03
        if intent.comparison_period:
            score += 0.03

        n_signals = sum([
            any(kw in q for kw in COMPARISON_SIGNALS),
            any(kw in q for kw in TREND_SIGNALS),
            any(kw in q for kw in TABLE_SIGNALS),
        ])
        if n_signals > 1:
            score -= 0.12
        if len(q.split()) < 4:
            score -= 0.10

        return min(max(score, 0.0), 0.90)

    def _build_search_query(self, raw: str, q: str, intent: QueryIntent) -> None:
        """Strip temporal and meta-terms that add noise to embedding queries."""
        cleaned = raw

        if intent.year and intent.month:
            cleaned = re.sub(r"\b" + str(intent.year) + r"\b", "", cleaned)

        for name in MONTH_MAP:
            cleaned = re.sub(r"\b" + name + r"\b", "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(r"\bq[1-4]\b", "", cleaned, flags=re.IGNORECASE)

        cleaned = re.sub(
            r"^(tell me|show me|give me|what is|what are|how many|how much|"
            r"search for|find me|can you|please|i want to know|"
            r"provide me with|fetch me)\s+",
            "", cleaned, flags=re.IGNORECASE,
        )

        cleaned = re.sub(r"\s*[?!.]\s*$", "", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        intent.search_query = cleaned or raw

    # ── Stage 4: Merge helpers ────────────────────────────────────────────────

    def _merge_embedding(
        self,
        intent: QueryIntent,
        emb: dict,
        emb_conf: float,
    ) -> QueryIntent:
        """Overlay embedding predictions on the rule-based intent."""
        if not emb or emb_conf < 0.60:
            return intent

        # Once stage 1 has explicitly resolved a year-range query (_fix_year_range_intent
        # decided tabular vs. trend from precise keyword matches), fuzzy embedding
        # similarity must not override that call — nearest-neighbor voting can't
        # reliably tell "AUM by year" (tabular) from "AUM year by year" (trend), and
        # flipping needs_analytical here would misroute a cheap query_sql query into
        # the 10-50x more expensive analytical month-by-month loop.
        has_resolved_year_range = bool(intent.year_from and intent.year_to)

        emb_intent_type = emb.get("intent_type")
        if emb_intent_type and emb_intent_type != "factual" and not has_resolved_year_range:
            if not intent.needs_analytical:
                intent.intent_type = emb_intent_type

        if emb.get("metric") and not intent.metric:
            intent.metric = emb["metric"]

        if emb.get("aggregation") and not intent.aggregation:
            intent.aggregation = emb["aggregation"]

        if emb.get("time_mode") and not intent.time_mode:
            intent.time_mode = emb["time_mode"]

        if emb.get("granularity") and not intent.granularity:
            intent.granularity = emb["granularity"]

        if emb.get("comparison_period") and not intent.comparison_period:
            intent.comparison_period = emb["comparison_period"]

        if emb.get("needs_analytical") and not has_resolved_year_range:
            intent.needs_analytical = True

        # Union of labels
        for lbl in (emb.get("labels") or []):
            if lbl in VALID_LABELS and lbl not in intent.labels:
                intent.labels.append(lbl)

        intent.confidence = min(
            intent.confidence * 0.4 + emb_conf * 0.6,
            0.95,
        )
        return intent

    def _merge_llm(
        self,
        intent: QueryIntent,
        llm: dict,
        llm_conf: float,
    ) -> QueryIntent:
        """Apply LLM classification to the intent.

        LLM wins on semantic fields; rule-based wins on temporal + entity fields.
        """
        def _notnull(v) -> bool:
            return v is not None and v != "null" and v != ""

        # Same reasoning as _merge_embedding: once stage 1 has explicitly resolved
        # a year-range query via precise keyword matching, a free-form LLM guess
        # must not override tabular-vs-trend / needs_analytical — it has no
        # visibility into _TREND_TYPE_KWS and can misroute a cheap query_sql
        # query into the far more expensive analytical month-by-month loop.
        has_resolved_year_range = bool(intent.year_from and intent.year_to)

        # primary_label maps to intent_type for graph routing
        primary = llm.get("primary_label") or llm.get("intent_type")
        if _notnull(primary) and not has_resolved_year_range:
            mapped = _LABEL_TO_INTENT.get(primary, primary)
            if mapped in ("factual", "trend", "tabular", "comparison",
                          "definition", "lookup", "analytical"):
                # Never let LLM degrade a structural tabular signal (year + metric) to factual
                structural_tabular = (intent.intent_type == "tabular"
                                      and intent.year and intent.metric)
                if not intent.needs_analytical and not (structural_tabular and mapped == "factual"):
                    intent.intent_type = mapped

        if _notnull(llm.get("secondary_intent")):
            intent.secondary_intent = llm["secondary_intent"]

        if _notnull(llm.get("metric")):
            intent.metric = llm["metric"]

        if _notnull(llm.get("fund_category")):
            intent.fund_category = llm["fund_category"]

        if _notnull(llm.get("aggregation")):
            intent.aggregation = llm["aggregation"]

        if _notnull(llm.get("time_mode")):
            intent.time_mode = llm["time_mode"]

        if _notnull(llm.get("granularity")):
            intent.granularity = llm["granularity"]

        if _notnull(llm.get("comparison_period")):
            intent.comparison_period = llm["comparison_period"]

        # ISO period → also back-fill year_from/year_to for analytical agent
        if _notnull(llm.get("start_period")):
            intent.start_period = llm["start_period"]
            try:
                y = int(llm["start_period"].split("-")[0])
                intent.year_from = intent.year_from or y
            except (ValueError, AttributeError):
                pass

        if _notnull(llm.get("end_period")):
            intent.end_period = llm["end_period"]
            try:
                y = int(llm["end_period"].split("-")[0])
                intent.year_to = intent.year_to or y
            except (ValueError, AttributeError):
                pass

        if llm.get("needs_analytical") and not has_resolved_year_range:
            intent.needs_analytical = True

        if llm.get("requires_clarification"):
            intent.requires_clarification = True

        # Multi-label: union with existing
        for lbl in (llm.get("labels") or []):
            if lbl in VALID_LABELS and lbl not in intent.labels:
                intent.labels.append(lbl)

        for name in (llm.get("amc_names") or []):
            if name and name not in intent.amc_names:
                intent.amc_names.append(name)

        # Update routing hints for new intent_type
        if intent.intent_type in ("tabular", "analytical"):
            intent.needs_tables = True
        if intent.intent_type == "lookup":
            intent.needs_documents = True

        intent.confidence = llm_conf
        intent.ambiguous  = llm_conf < 0.60
        return intent
