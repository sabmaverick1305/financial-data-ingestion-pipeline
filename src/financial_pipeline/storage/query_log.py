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
                    query_id          UUID PRIMARY KEY,
                    thread_id         UUID NOT NULL,
                    question          TEXT NOT NULL,
                    answer            TEXT,
                    intent_type       TEXT,
                    route             TEXT,
                    citations_count   INT,
                    latency_ms        INT,
                    blocked           BOOLEAN DEFAULT FALSE,
                    created_at        TIMESTAMPTZ DEFAULT NOW(),
                    feedback_rating   INT,
                    feedback_comment  TEXT,
                    feedback_at       TIMESTAMPTZ
                )
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_query_log_thread_id
                ON query_log (thread_id)
            """))
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_query_log_created_at
                ON query_log (created_at)
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
    ) -> None:
        with self._engine.begin() as conn:
            conn.execute(text("""
                INSERT INTO query_log (
                    query_id, thread_id, question, answer, intent_type, route,
                    citations_count, latency_ms, blocked
                ) VALUES (
                    :query_id, :thread_id, :question, :answer, :intent_type, :route,
                    :citations_count, :latency_ms, :blocked
                )
            """), {
                "query_id": query_id, "thread_id": thread_id, "question": question,
                "answer": answer, "intent_type": intent_type, "route": route,
                "citations_count": citations_count, "latency_ms": latency_ms,
                "blocked": blocked,
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
