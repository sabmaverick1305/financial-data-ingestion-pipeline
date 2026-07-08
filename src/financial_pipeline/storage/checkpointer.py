"""Postgres-backed LangGraph checkpointer.

Uses a psycopg v3 ConnectionPool rather than PostgresSaver.from_conn_string()'s
single-Connection context manager: the FastAPI app runs sync route handlers
in a threadpool, so the checkpointer must be thread-safe and outlive any one
request. A pool gives both; a single raw connection (the from_conn_string
default) does not — and its context manager would close the connection the
moment the `with` block exits, which happens once at startup, not once per
request.
"""
from __future__ import annotations

import re

import structlog
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

log = structlog.get_logger()

_SQLALCHEMY_DRIVER_RE = re.compile(r"^postgresql\+\w+://")


def to_psycopg_dsn(sqlalchemy_url: str) -> str:
    """Strip a SQLAlchemy driver suffix (e.g. postgresql+psycopg2://...) down
    to the plain postgresql://... DSN psycopg v3 expects. settings.postgres_url
    is a SQLAlchemy engine URL (used by DocumentRepository); the checkpointer
    talks to psycopg directly and doesn't understand the +driver segment."""
    return _SQLALCHEMY_DRIVER_RE.sub("postgresql://", sqlalchemy_url)


def build_checkpointer(postgres_url: str) -> PostgresSaver:
    """Build and initialize a Postgres-backed checkpointer.

    Call once at app startup; the returned PostgresSaver (backed by a
    connection pool) lives for the process lifetime — every graph.invoke()
    call reuses it. .setup() idempotently creates the checkpoint_* tables,
    safe to call on every boot.
    """
    dsn = to_psycopg_dsn(postgres_url)
    pool = ConnectionPool(
        conninfo=dsn,
        kwargs={"autocommit": True, "row_factory": dict_row},
        min_size=1,
        max_size=10,
        open=True,
    )
    checkpointer = PostgresSaver(pool)
    checkpointer.setup()
    log.info("checkpointer.ready", backend="postgres")
    return checkpointer
