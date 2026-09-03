"""query_log — persists each /api/ask request/answer for feedback and eval
linkage, keyed by query_id.

Distinct from the LangGraph checkpoint tables (checkpoint_*, owned by
PostgresSaver — see checkpointer.py): those persist node-by-node graph
STATE for replay/resumability, keyed by thread_id. This table persists a
flat, queryable SUMMARY of each request/answer, keyed by query_id, so a
feedback UI or an eval pipeline can look up "what did we answer for this
query_id" without deserializing checkpoint blobs.
"""
from __future__ import annotations

import structlog
from sqlalchemy import text
from sqlalchemy.engine import Engine

log = structlog.get_logger()


class QueryLogRepository:
    """Owns the query_log table."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def create_tables(self) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS query_log (
                    query_id             UUID PRIMARY KEY,
                    thread_id            UUID NOT NULL,
                    user_name            TEXT,
                    user_email           TEXT,
                    question             TEXT NOT NULL,
                    answer               TEXT,
                    intent_type          TEXT,
                    route                TEXT,
                    citations_count      INT,
                    latency_ms           INT,
                    blocked              BOOLEAN DEFAULT FALSE,
                    block_reason         TEXT,
                    hallucination_risk   TEXT,
                    faithfulness_score   FLOAT,
                    citation_valid       BOOLEAN,
                    number_consistent    BOOLEAN,
                    abstention_detected  BOOLEAN,
                    created_at           TIMESTAMPTZ DEFAULT NOW(),
                    feedback_rating      INT,
                    feedback_comment     TEXT,
                    feedback_at          TIMESTAMPTZ,
                    triage_category      TEXT,
                    triage_priority      TEXT,
                    triage_reason        TEXT,
                    triaged_at           TIMESTAMPTZ
                )
            """))
            # Idempotent migrations for tables created before these columns existed.
            for migration in [
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS block_reason TEXT",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS hallucination_risk TEXT",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS faithfulness_score FLOAT",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS citation_valid BOOLEAN",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS number_consistent BOOLEAN",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS abstention_detected BOOLEAN",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS triage_category TEXT",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS triage_priority TEXT",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS triage_reason TEXT",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS triaged_at TIMESTAMPTZ",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS user_name TEXT",
                "ALTER TABLE query_log ADD COLUMN IF NOT EXISTS user_email TEXT",
            ]:
                conn.execute(text(migration))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_query_log_thread_id
                ON query_log (thread_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_query_log_created_at
                ON query_log (created_at)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_query_log_triage_priority
                ON query_log (triage_priority, created_at)
            """))
        log.info("query_log.tables_ready")

    def log_query(
        self,
        *,
        query_id: str,
        thread_id: str,
        question: str,
        answer: str,
        intent_type: str | None,
        route: str | None,
        citations_count: int,
        latency_ms: int,
        blocked: bool,
        block_reason: str | None = None,
        hallucination_risk: str | None = None,
        faithfulness_score: float | None = None,
        citation_valid: bool | None = None,
        number_consistent: bool | None = None,
        abstention_detected: bool | None = None,
        user_name: str | None = None,
        user_email: str | None = None,
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO query_log (
                    query_id, thread_id, question, answer, intent_type, route,
                    citations_count, latency_ms, blocked, block_reason,
                    hallucination_risk, faithfulness_score, citation_valid,
                    number_consistent, abstention_detected, user_name, user_email
                ) VALUES (
                    :query_id, :thread_id, :question, :answer, :intent_type, :route,
                    :citations_count, :latency_ms, :blocked, :block_reason,
                    :hallucination_risk, :faithfulness_score, :citation_valid,
                    :number_consistent, :abstention_detected, :user_name, :user_email
                )
            """), {
                "query_id": query_id, "thread_id": thread_id, "question": question,
                "answer": answer, "intent_type": intent_type, "route": route,
                "citations_count": citations_count, "latency_ms": latency_ms,
                "blocked": blocked, "block_reason": block_reason,
                "hallucination_risk": hallucination_risk,
                "faithfulness_score": faithfulness_score,
                "citation_valid": citation_valid,
                "number_consistent": number_consistent,
                "abstention_detected": abstention_detected,
                "user_name": user_name,
                "user_email": user_email,
            })

    def record_feedback(self, *, query_id: str, rating: int, comment: str | None) -> bool:
        """Returns True if a query_log row was found and updated, False if
        query_id doesn't exist — the caller (the /api/feedback endpoint)
        turns False into a 404 rather than silently no-op'ing."""
        with self._engine.begin() as conn:
            result = conn.execute(text("""
                UPDATE query_log
                SET feedback_rating  = :rating,
                    feedback_comment = :comment,
                    feedback_at      = NOW()
                WHERE query_id = :query_id
            """), {"query_id": query_id, "rating": rating, "comment": comment})
            return result.rowcount > 0

    def fetch_row(self, query_id: str) -> dict | None:
        """Fetch a single query_log row as a plain dict (for triage input)."""
        with self._engine.begin() as conn:
            result = conn.execute(
                text("SELECT * FROM query_log WHERE query_id = :query_id"),
                {"query_id": query_id},
            )
            row = result.mappings().first()
            return dict(row) if row else None

    def save_triage(self, *, query_id: str, category: str, priority: str, reason: str) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                UPDATE query_log
                SET triage_category = :category,
                    triage_priority = :priority,
                    triage_reason   = :reason,
                    triaged_at      = NOW()
                WHERE query_id = :query_id
            """), {
                "query_id": query_id, "category": category,
                "priority": priority, "reason": reason,
            })

    def fetch_untriaged(self, limit: int = 200) -> list[dict]:
        """Rows never triaged yet — either just logged (guardrail signals
        already present) or feedback arrived after the last triage pass."""
        with self._engine.begin() as conn:
            result = conn.execute(text("""
                SELECT * FROM query_log
                WHERE triaged_at IS NULL
                ORDER BY created_at ASC
                LIMIT :limit
            """), {"limit": limit})
            return [dict(row) for row in result.mappings().all()]

    def fetch_for_review(
        self, *, priority: str | None = None, category: str | None = None, limit: int = 50,
    ) -> list[dict]:
        """The human-review queue: triaged rows, most urgent first."""
        clauses = ["triage_category IS NOT NULL"]
        params: dict = {"limit": limit}
        if priority:
            clauses.append("triage_priority = :priority")
            params["priority"] = priority
        if category:
            clauses.append("triage_category = :category")
            params["category"] = category
        where = " AND ".join(clauses)
        with self._engine.begin() as conn:
            result = conn.execute(text(f"""
                SELECT * FROM query_log
                WHERE {where}
                ORDER BY
                    CASE triage_priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                    created_at DESC
                LIMIT :limit
            """), params)
            return [dict(row) for row in result.mappings().all()]
