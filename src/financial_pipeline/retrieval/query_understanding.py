"""QueryAnalyzer — Step 1 of the retrieval pipeline.

Parses a natural-language query into a structured QueryIntent without
calling an LLM, using regex, keyword matching, and simple heuristics.
Fast (<1 ms) and always available regardless of LLM configuration.

What it extracts:
  - Temporal context  : year, month, quarter
  - Document category : monthly | quarterly
  - Query intent      : factual | trend | tabular | comparison | definition | lookup
  - Financial entities: scheme types, AUM/NAV/folio keywords
  - Routing hints     : which sources to search (chunks / tables / metadata)
  - Clean search text : stripped of meta-terms for better semantic embedding
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# ── Vocabulary tables ─────────────────────────────────────────────────────────

MONTH_MAP: dict[str, int] = {
    "january": 1,
    "jan": 1,
    "february": 2,
    "feb": 2,
    "march": 3,
    "mar": 3,
    "april": 4,
    "apr": 4,
    "may": 5,
    "june": 6,
    "jun": 6,
    "july": 7,
    "jul": 7,
    "august": 8,
    "aug": 8,
    "september": 9,
    "sep": 9,
    "sept": 9,
    "october": 10,
    "oct": 10,
    "november": 11,
    "nov": 11,
    "december": 12,
    "dec": 12,
}

QUARTER_MAP: dict[str, tuple[int, int]] = {
    # Maps Q-label to (start_month, end_month) for display
    "q1": (1, 3),
    "q2": (4, 6),
    "q3": (7, 9),
    "q4": (10, 12),
}

TABLE_SIGNALS = frozenset(
    {
        "table",
        "breakdown",
        "distribution",
        "count",
        "number of",
        "how many",
        "total number",
        "scheme count",
        "fund count",
        "list of",
        "show me",
        "give me the list",
        "aum breakdown",
        "folio count",
        "category-wise",
        "type-wise",
        "wise break",
    }
)

TREND_SIGNALS = frozenset(
    {
        "growth",
        "trend",
        "over the years",
        "historical",
        "changed",
        "increase",
        "decrease",
        "decline",
        "rise",
        "fell",
        "grew",
        "year-on-year",
        "yoy",
        "month-on-month",
        "mom",
        "quarter-on-quarter",
        "performance over",
        "how has",
        "evolution",
        "trajectory",
        # multi-year range signals
        "year wise",
        "year-wise",
        "yearwise",
        "year by year",
        "annually",
        "each year",
        "every year",
        "from 20",          # catches "from 2020 to 2026" style queries
    }
)

COMPARISON_SIGNALS = frozenset(
    {
        "compare",
        "versus",
        "vs",
        "vs.",
        "difference between",
        "equity vs",
        "debt vs",
        "better than",
        "higher than",
        "lower than",
        "relative to",
        "against",
        "benchmark",
    }
)

DEFINITION_SIGNALS = frozenset(
    {
        "what is",
        "what are",
        "define",
        "explain",
        "meaning of",
        "what does",
        "how does",
        "description of",
        "definition",
    }
)

LOOKUP_SIGNALS = frozenset(
    {
        "get",
        "fetch",
        "download",
        "show me the report",
        "give me the",
        "monthly report for",
        "quarterly report for",
        "repository for",
    }
)

SCHEME_TYPES = [
    "equity",
    "debt",
    "hybrid",
    "liquid",
    "gilt",
    "balanced",
    "etf",
    "index fund",
    "elss",
    "fund of funds",
    "balanced advantage",
    "arbitrage",
    "overnight",
    "money market",
    "credit risk",
    "dynamic bond",
    "large cap",
    "mid cap",
    "small cap",
    "flexi cap",
    "multi cap",
    "thematic",
    "sectoral",
    "conservative hybrid",
    "aggressive hybrid",
    "fixed maturity",
    "fmp",
    "interval fund",
]

FINANCIAL_TERMS = frozenset(
    {
        "aum",
        "nav",
        "folio",
        "sip",
        "lumpsum",
        "redemption",
        "mobilisation",
        "inflow",
        "outflow",
        "net",
        "gross",
        "scheme",
        "fund",
        "category",
        "subcategory",
    }
)


# ── Output dataclass ──────────────────────────────────────────────────────────


@dataclass
class QueryIntent:
    """Structured representation of what the user is asking."""

    # Original
    raw_query: str

    # Temporal filters (extracted)
    year: int | None = None
    month: int | None = None
    quarter: str | None = None  # "Q1" … "Q4"

    # Year range (multi-year trend queries: "from 2020 to 2026")
    # When set, year is cleared to None so the retriever uses the range instead.
    year_from: int | None = None
    year_to:   int | None = None

    # Document filters
    category: str | None = None  # "monthly" | "quarterly"

    # Intent
    intent_type: str = "factual"
    # Values: factual | trend | tabular | comparison | definition | lookup

    # Financial entities
    scheme_types: list[str] = field(default_factory=list)
    financial_terms: list[str] = field(default_factory=list)

    # Routing hints (set by analyzer, consumed by SearchRouter)
    needs_chunks: bool = True
    needs_tables: bool = False
    needs_documents: bool = False
    time_sensitive: bool = False  # query has an explicit time reference
    prefer_recent: bool = False  # no year given, prefer recent data

    # Cleaned query for semantic embedding (meta-terms removed)
    search_query: str = ""

    def describe(self) -> str:
        """Human-readable summary — logged at debug level."""
        parts = [f"intent={self.intent_type}"]
        if self.year:
            parts.append(f"year={self.year}")
        if self.month:
            parts.append(f"month={self.month}")
        if self.quarter:
            parts.append(f"quarter={self.quarter}")
        if self.category:
            parts.append(f"category={self.category}")
        if self.scheme_types:
            parts.append(f"schemes={self.scheme_types}")
        parts.append(
            f"sources={'|'.join([s for s, v in [('chunks', self.needs_chunks), ('tables', self.needs_tables), ('docs', self.needs_documents)] if v])}"  # noqa: E501
        )
        return "  ".join(parts)


# ── Analyzer ──────────────────────────────────────────────────────────────────


class QueryAnalyzer:
    """Converts a raw query string into a structured QueryIntent.

    Fully rule-based — no network calls, no LLM dependency.
    """

    def analyze(self, query: str) -> QueryIntent:
        intent = QueryIntent(raw_query=query, search_query=query)
        q = query.lower().strip()

        self._extract_year_range(q, intent)   # must run before _extract_temporal
        self._extract_temporal(q, intent)
        self._classify_intent(q, intent)
        self._extract_entities(q, intent)
        self._set_routing_hints(q, intent)
        self._build_search_query(query, q, intent)

        return intent

    # ------------------------------------------------------------------

    def _extract_year_range(self, q: str, intent: QueryIntent) -> None:
        """Detect multi-year range patterns before single-year extraction runs.

        Patterns matched:
          "from 2020 to 2026"
          "2020 to 2026"
          "2020 through 2026"
          "2020-2026"   (hyphen — not a month range)
          "2020 – 2026" (en-dash)

        When a range is found:
          - year_from / year_to are set
          - year is cleared (don't pin to a single year)
          - intent_type is forced to "trend"
          - prefer_recent is cleared (avoid resolving to one month)
        """
        range_m = re.search(
            r'\b(20[0-2]\d)\s*(?:to|through|–|-)\s*(20[0-2]\d)\b',
            q,
        )
        if range_m:
            y1, y2 = int(range_m.group(1)), int(range_m.group(2))
            if y1 != y2:                   # "2026-2026" is not a range
                intent.year_from     = min(y1, y2)
                intent.year_to       = max(y1, y2)
                intent.year          = None    # cleared — range replaces single year
                intent.intent_type   = "trend"
                intent.prefer_recent = False
                intent.time_sensitive = True

    def _extract_temporal(self, q: str, intent: QueryIntent) -> None:
        # Year: 2009–2029
        # Skip single-year extraction when a range was already found — the
        # regex would pick up year_from (the earlier year) and overwrite None.
        if not (intent.year_from and intent.year_to):
            year_m = re.search(r"\b(20[0-2]\d)\b", q)
            if year_m:
                intent.year = int(year_m.group(1))
                intent.time_sensitive = True

        # Month by name
        for name, num in MONTH_MAP.items():
            if re.search(r"\b" + re.escape(name) + r"\b", q):
                intent.month = num
                intent.time_sensitive = True
                break

        # Quarter  (Q1 2024 / 2024 Q3)
        qtr_m = re.search(r"\b(q[1-4])\b", q)
        if qtr_m:
            intent.quarter = qtr_m.group(1).upper()
            intent.time_sensitive = True

        # "latest", "recent", "current" → prefer_recent without explicit year
        if re.search(r"\b(latest|recent|current|today|now)\b", q) and not intent.year:
            intent.prefer_recent = True

        # Year given but no month → user wants the latest available data for that year.
        # Mark prefer_recent so the ContextOptimizer and pipeline can resolve the month.
        # Exceptions: trend queries want all months; range queries already set prefer_recent=False.
        if (intent.year and not intent.month and not intent.quarter
                and not (intent.year_from and intent.year_to)):
            intent.prefer_recent = True

        # Category hints from temporal language
        if "monthly" in q or "month of" in q:
            intent.category = "monthly"
        elif "quarterly" in q or "quarter" in q:
            intent.category = "quarterly"

    def _classify_intent(self, q: str, intent: QueryIntent) -> None:
        # Order matters: check specific signals first
        if any(kw in q for kw in COMPARISON_SIGNALS):
            intent.intent_type = "comparison"
            intent.needs_chunks = True

        elif any(kw in q for kw in TREND_SIGNALS):
            intent.intent_type = "trend"
            intent.needs_chunks = True
            intent.needs_tables = True  # trend queries often want numbers

        elif any(kw in q for kw in TABLE_SIGNALS):
            intent.intent_type = "tabular"
            intent.needs_tables = True
            intent.needs_chunks = True  # still need text for context

        elif any(re.search(r"\b" + re.escape(kw) + r"\b", q) for kw in DEFINITION_SIGNALS):
            intent.intent_type = "definition"
            intent.needs_chunks = True

        elif any(re.search(r"\b" + re.escape(kw), q) for kw in LOOKUP_SIGNALS):
            intent.intent_type = "lookup"
            intent.needs_documents = True
            intent.needs_chunks = True

        else:
            intent.intent_type = "factual"
            intent.needs_chunks = True

    def _extract_entities(self, q: str, intent: QueryIntent) -> None:
        for scheme in SCHEME_TYPES:
            if scheme in q:
                intent.scheme_types.append(scheme)

        for term in FINANCIAL_TERMS:
            if re.search(r"\b" + re.escape(term) + r"\b", q):
                intent.financial_terms.append(term)

    def _set_routing_hints(self, q: str, intent: QueryIntent) -> None:
        # Table lookup for "how many schemes" / "number of funds" / explicit table mentions
        if re.search(r"\b(how many|number of|count of|total)\b.*(scheme|fund|folio)", q):
            intent.needs_tables = True

        # Document lookup for "get the report" style
        if re.search(r"\b(report for|repository for|data for|document for)\b", q):
            intent.needs_documents = True

        # Tabular data questions need real numbers — tables are more reliable than prose
        if "aum" in q and any(kw in q for kw in ("total", "breakdown", "category")):
            intent.needs_tables = True

    def _build_search_query(self, raw: str, q: str, intent: QueryIntent) -> None:
        """Remove meta-terms that hurt semantic search quality."""
        cleaned = raw
        # Remove year only when a specific month is also present — the month
        # already scopes the search and the year adds noise.
        # When no month is given, keep the year so column headers like
        # "No. of Folios as on May 31, 2026" are matched correctly.
        if intent.year and intent.month:
            cleaned = re.sub(r"\b" + str(intent.year) + r"\b", "", cleaned)
        # Remove bare month names (kept in intent.month already)
        for name in MONTH_MAP:
            cleaned = re.sub(r"\b" + name + r"\b", "", cleaned, flags=re.IGNORECASE)
        # Remove quarter labels
        cleaned = re.sub(r"\bq[1-4]\b", "", cleaned, flags=re.IGNORECASE)
        # Collapse whitespace
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        intent.search_query = cleaned or raw
