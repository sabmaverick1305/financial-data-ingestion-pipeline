"""Vanna.ai text-to-SQL agent for amfi_fund_stats queries.

Uses Anthropic Claude for SQL generation and ChromaDB for persisting
training examples (DDL, documentation, question–SQL pairs).

Typical lifecycle
-----------------
1. At app startup: ``agent = build_vanna_agent(settings)``
2. After first deploy (or schema changes): run ``scripts/train_vanna.py``
3. At query time: ``sql, df, answer = agent.ask(query)``
"""
from __future__ import annotations

import concurrent.futures
import re
import time
from pathlib import Path
from typing import TYPE_CHECKING

import structlog
from vanna.anthropic import Anthropic_Chat
from vanna.chromadb import ChromaDB_VectorStore

if TYPE_CHECKING:
    import pandas as pd

log = structlog.get_logger()

# Default path for persisted ChromaDB training store
_DEFAULT_CHROMA_PATH = str(
    Path(__file__).parent.parent.parent.parent / ".vanna_chromadb"
)

# ── SQL safety policy ─────────────────────────────────────────────────────────

APPROVED_TABLES: frozenset[str] = frozenset({
    "amfi_fund_stats", "amfi_amc_stats",
    # Per-scheme mutual fund dataset (mfapi.in) — see mf_ingestion/, mf_performance/.
    "mf_scheme_master", "mf_nav_history", "mf_scheme_performance",
})
MAX_ROWS: int = 500
QUERY_TIMEOUT_SEC: int = 30

_SELECT_RE  = re.compile(r"^\s*(?:--[^\n]*\n\s*)*SELECT\b", re.IGNORECASE | re.DOTALL)
_TABLE_RE   = re.compile(r"\b(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)", re.IGNORECASE)
_LIMIT_RE   = re.compile(r"\bLIMIT\s+\d+", re.IGNORECASE)
_DATE_RE    = re.compile(r"\bperiod_(?:year|month)\b", re.IGNORECASE)
# mf_scheme_master/mf_nav_history/mf_scheme_performance have no period_year/
# period_month columns — scheme_code (or nav_date) is their bounding filter.
_SCHEME_BOUND_RE = re.compile(r"\b(?:scheme_code|nav_date)\b", re.IGNORECASE)

# Question-level pre-checks: catch SQL injection before calling the LLM
_QUESTION_DML_RE = re.compile(
    r"^\s*(UPDATE|DELETE|INSERT|DROP|TRUNCATE|ALTER|GRANT|REVOKE|EXEC(?:UTE)?)\b",
    re.IGNORECASE,
)
_QUESTION_SELECT_RE = re.compile(r"^\s*SELECT\b", re.IGNORECASE)

# Detect LLM refusals (natural-language decline, not actual SQL) so we can
# return "no results" instead of a hard security block for data-absent queries.
_LLM_REFUSAL_RE = re.compile(
    r"^\s*(i\s+(cannot|can\s+only|am\s+unable)|the\s+llm\s+is\s+not"
    r"|forecasting\s+requires|unfortunately[,\s]"
    r"|the\s+provided\s+context\s+is\s+insufficient)",
    re.IGNORECASE,
)


def _validate_and_prepare(sql: str) -> tuple[str, list[str]]:
    """Apply safety policy to generated SQL.

    Returns (possibly-modified sql, list of warning strings).
    Raises ValueError for hard violations (non-SELECT, unapproved table).
    """
    warnings: list[str] = []

    # 1. SELECT only — hard block
    if not _SELECT_RE.match(sql):
        first = sql.strip()[:60]
        raise ValueError(f"Only SELECT statements are permitted. Got: {first!r}")

    # 2. Approved tables only — hard block
    tables_found = set(_TABLE_RE.findall(sql))
    bad_tables = {t.lower() for t in tables_found} - {t.lower() for t in APPROVED_TABLES}
    if bad_tables:
        raise ValueError(
            f"Query references unapproved table(s): {bad_tables}. "
            f"Allowed: {set(APPROVED_TABLES)}"
        )

    # 3. Date filter check — warn (LIMIT already caps blast radius)
    if tables_found and not _DATE_RE.search(sql) and not _SCHEME_BOUND_RE.search(sql):
        msg = "No period_year/period_month/scheme_code/nav_date filter detected; query may scan full table"
        warnings.append(msg)
        log.warning("vanna.no_date_filter", tables=list(tables_found), hint=msg)

    # 4. Inject LIMIT if absent — auto-fix
    if not _LIMIT_RE.search(sql):
        sql = sql.rstrip().rstrip(";")
        sql = f"{sql} LIMIT {MAX_ROWS}"
        log.info("vanna.limit_injected", max_rows=MAX_ROWS)
        warnings.append(f"LIMIT {MAX_ROWS} automatically applied")

    return sql, warnings


class FinancialVanna(ChromaDB_VectorStore, Anthropic_Chat):
    """Vanna subclass wiring Anthropic Claude SQL generation with ChromaDB vector store."""

    def __init__(self, config: dict | None = None) -> None:
        ChromaDB_VectorStore.__init__(self, config=config)
        Anthropic_Chat.__init__(self, config=config)


def build_vanna_agent(
    anthropic_api_key: str,
    postgres_url: str,
    chroma_path: str = _DEFAULT_CHROMA_PATH,
    model: str = "claude-haiku-4-5-20251001",
) -> FinancialVanna:
    """Instantiate and connect the Vanna agent.

    Parameters
    ----------
    anthropic_api_key:
        Anthropic API key (``sk-ant-...``) for SQL generation.
    postgres_url:
        SQLAlchemy-style URL for the RDS database
        (``postgresql+psycopg2://user:pass@host:port/db``).
    chroma_path:
        Local filesystem path where ChromaDB persists training examples.
    model:
        Anthropic model to use; defaults to ``claude-haiku-4-5-20251001``.
    """
    Path(chroma_path).mkdir(parents=True, exist_ok=True)

    vn = FinancialVanna(config={
        "api_key": anthropic_api_key,
        "model":   model,
        "path":    chroma_path,
    })

    # Parse postgres_url → individual connection params
    _connect_postgres(vn, postgres_url)

    # connect_to_postgres() sets vn.run_sql to a raw, unfiltered executor.
    # ask() below applies _validate_and_prepare() before calling it — but
    # Vanna's own generate_sql(allow_llm_to_see_data=True) also calls
    # self.run_sql() directly and internally, for the "intermediate_sql"
    # disambiguation pattern (e.g. "find distinct scheme_name values first").
    # Wrap run_sql itself so that path is policy-checked too, not just the
    # explicit one in ask().
    raw_run_sql = vn.run_sql

    def _policy_checked_run_sql(sql: str, **kwargs):
        checked_sql, _ = _validate_and_prepare(sql)
        return raw_run_sql(checked_sql, **kwargs)

    vn.run_sql = _policy_checked_run_sql

    log.info("vanna.agent_ready", model=model, chroma_path=chroma_path)
    return vn


def _connect_postgres(vn: FinancialVanna, url: str) -> None:
    """Parse a SQLAlchemy URL and connect Vanna to Postgres."""
    # Strip driver prefix
    raw = re.sub(r"^postgresql\+\w+://", "", url)
    userpass, hostdb = raw.split("@", 1)
    user, password   = userpass.split(":", 1)
    hostport, dbname = hostdb.split("/", 1)
    parts = hostport.split(":")
    host = parts[0]
    port = int(parts[1]) if len(parts) > 1 else 5432
    password = password.replace("%40", "@")

    vn.connect_to_postgres(
        host=host, dbname=dbname, user=user,
        password=password, port=port,
    )
    log.info("vanna.postgres_connected", host=host, db=dbname)


# ── Query helper ──────────────────────────────────────────────────────────────

def ask(
    vn: FinancialVanna,
    question: str,
    *,
    auto_train: bool = False,
) -> tuple[str | None, "pd.DataFrame | None", str, list[str]]:
    """Generate SQL, apply safety policy, execute with timeout, return markdown answer.

    Returns
    -------
    sql : str | None
        The final (possibly policy-modified) SQL, or None on failure.
    df : DataFrame | None
        Query result rows (None on failure).
    answer : str
        Human-readable markdown answer, or an error message.
    warnings : list[str]
        Policy notices (e.g. auto-injected LIMIT, missing date filter).
    """
    t0 = time.time()
    try:
        q = question.strip()

        # ── Question-level pre-checks (before calling LLM) ────────────────
        # Block DML statements directly in the question text
        if _QUESTION_DML_RE.match(q):
            log.warning("vanna.question_is_dml", question=q[:80])
            return None, None, "SQL blocked by safety policy: Only SELECT statements are permitted.", []

        # Block SELECT-style injection against unapproved tables
        if _QUESTION_SELECT_RE.match(q):
            tables = {t.lower() for t in _TABLE_RE.findall(q)}
            bad = tables - {t.lower() for t in APPROVED_TABLES}
            if bad:
                log.warning("vanna.question_unapproved_table", tables=list(bad), question=q[:80])
                return None, None, f"SQL blocked by safety policy: Unapproved table(s) {bad}.", []

        # allow_llm_to_see_data=True lets Vanna resolve its own documented
        # "intermediate_sql" pattern automatically (e.g. running a `SELECT
        # DISTINCT scheme_name ...` disambiguation query and feeding the
        # results back into a second LLM call) instead of returning that
        # intermediate query as if it were the final answer. Safe to allow
        # here since run_sql itself is now policy-checked (see build_vanna_agent).
        raw_sql = vn.generate_sql(question, allow_llm_to_see_data=True)
        if not raw_sql:
            return None, None, "Could not generate SQL for this question.", []

        # ── Safety policy ──────────────────────────────────────────────────
        try:
            sql, policy_warnings = _validate_and_prepare(raw_sql)
        except ValueError as exc:
            # Distinguish LLM refusals (semantic: no data for the query) from
            # actual policy violations (e.g. generated malformed SQL).
            # Refusals → empty result; policy violations → hard block.
            if _LLM_REFUSAL_RE.match(raw_sql.strip()):
                log.info("vanna.llm_refused", question=q[:80])
                return None, None, "The query returned no results for the given filters.", []
            log.warning("vanna.sql_blocked", reason=str(exc), raw_sql=raw_sql)
            return raw_sql, None, f"SQL blocked by safety policy: {exc}", []

        # ── Full SQL log (not truncated) ───────────────────────────────────
        log.info("vanna.sql_generated", question=question[:80], sql=sql)

        # ── Timeout-guarded execution ──────────────────────────────────────
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(vn.run_sql, sql)
            try:
                df = fut.result(timeout=QUERY_TIMEOUT_SEC)
            except concurrent.futures.TimeoutError:
                log.warning("vanna.query_timeout", timeout_sec=QUERY_TIMEOUT_SEC)
                return sql, None, f"SQL query exceeded {QUERY_TIMEOUT_SEC}s timeout.", policy_warnings

        latency_ms = int((time.time() - t0) * 1000)

        if df is None or df.empty:
            return sql, df, "The query returned no results for the given filters.", policy_warnings

        answer = _format_dataframe(df, question, sql)
        log.info("vanna.query_ok", rows=len(df), latency_ms=latency_ms,
                 policy_warnings=policy_warnings)
        return sql, df, answer, policy_warnings

    except Exception as exc:
        log.warning("vanna.ask_failed", error=str(exc))
        return None, None, f"SQL query failed: {exc}", []


def _format_dataframe(df: "pd.DataFrame", question: str, sql: str) -> str:
    """Render a DataFrame as a clean markdown table with a short header."""
    # Round all numeric columns to 2 decimal places for readability
    display = df.copy()
    for col in display.select_dtypes(include="number").columns:
        display[col] = display[col].round(2)

    md_table = display.to_markdown(index=False)

    lines = [
        f"**Query:** {question}",
        "",
        md_table,
        "",
        f"*{len(df)} row(s) returned.*",
    ]
    return "\n".join(lines)
