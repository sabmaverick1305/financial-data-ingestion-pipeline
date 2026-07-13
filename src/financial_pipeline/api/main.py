"""AMFI Financial Document Retrieval API.

Endpoints:
  POST /api/search    — intelligent 5-stage retrieval pipeline
  POST /api/ask       — RAG: retrieve + generate grounded answer with citations
  GET  /api/documents — list / filter ingested documents
  GET  /api/stats     — pipeline health and index statistics
  GET  /healthz       — liveness probe (ECS / ALB)

Retrieval pipeline (5 stages):
  1. QueryAnalyzer    — understand intent, extract year/month/entities
  2. SearchRouter     — decide which sources to query (chunks/tables/docs)
  3. MultiSourceFetcher — fetch from pgvector + document_table_assets + metadata
  4. ResultRanker     — deduplicate, MMR diversity filter
  5. ContextAssembler — numbered citations, LLM-ready context block

Start locally:
    python scripts/serve.py
    # or
    uvicorn financial_pipeline.api.main:app --reload --port 8080
"""

from __future__ import annotations

import asyncio
import os
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from langchain_core.runnables import RunnableConfig
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from sqlalchemy import text

from financial_pipeline.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkResult,
    DocumentListResponse,
    DocumentSummary,
    FeedbackRequest,
    FeedbackResponse,
    GuardrailReport,
    PipelineStats,
    SearchRequest,
    SearchResponse,
    Source,
)
from financial_pipeline.evaluation.observability import RequestTrace, emitter, ls_feedback
from financial_pipeline.graph import build_graph, NodeFactory, AnalyticalNodeFactory
from financial_pipeline.config import settings
from financial_pipeline.retrieval.pipeline import RetrievalPipeline
from financial_pipeline.retrieval.rag import RAGPipeline
from financial_pipeline.retrieval.retriever import Retriever
from financial_pipeline.storage.checkpointer import build_checkpointer
from financial_pipeline.storage.document_repo import DocumentRepository
from financial_pipeline.storage.query_log import QueryLogRepository

log = structlog.get_logger()

# ── Prometheus metrics ────────────────────────────────────────────────────────

_LATENCY_BUCKETS = (100, 250, 500, 1000, 2000, 4000, 8000, 15000, 30000)

QUERY_LATENCY = Histogram(
    "amfi_query_latency_ms",
    "End-to-end /api/ask latency in milliseconds",
    ["intent_type"],
    buckets=_LATENCY_BUCKETS,
)
QUERIES_TOTAL = Counter(
    "amfi_queries_total",
    "Total /api/ask requests",
    ["intent_type"],
)
INTENT_STAGE = Counter(
    "amfi_intent_stage_total",
    "Intent classification stage that resolved the query",
    ["stage"],          # rule | embedding | llm
)
LLM_TOKENS = Counter(
    "amfi_llm_tokens_total",
    "LLM tokens consumed",
    ["token_type"],     # prompt | completion
)
GUARDRAIL_BLOCKS = Counter(
    "amfi_guardrail_blocks_total",
    "Pre-guardrail hard blocks (investment advice / unsafe)",
)
ANALYTICAL_QUERIES = Counter(
    "amfi_analytical_queries_total",
    "Year-range analytical agent invocations",
)

# ── Application state (initialised once at startup) ───────────────────────────


class _AppState:
    repo: DocumentRepository | None = None
    retriever: Retriever | None = None
    pipeline: RetrievalPipeline | None = None
    rag: RAGPipeline | None = None
    graph = None  # compiled LangGraph — set in lifespan
    checkpointer = None  # Postgres-backed BaseCheckpointSaver — set in lifespan, None if disabled
    query_log: QueryLogRepository | None = None


_state = _AppState()


async def _periodic_metrics_flush(interval_seconds: int = 120) -> None:
    """Safety-net flush for MetricsEmitter's CloudWatch buffer — it only
    auto-flushes once BATCH_SIZE (20) requests have accumulated (see
    evaluation/observability.py), so a quiet period could otherwise leave
    metrics sitting in memory indefinitely. Runs for the lifetime of the
    app; cancelled in lifespan()'s shutdown block."""
    while True:
        await asyncio.sleep(interval_seconds)
        try:
            emitter.flush()
        except Exception as exc:
            log.warning("api.metrics_periodic_flush_failed", error=str(exc))


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup")

    # ── LangSmith tracing — must be set before build_graph() ──────────────
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGSMITH_API_KEY",      settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_API_KEY",      settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT",      settings.langchain_project)
        os.environ.setdefault("LANGCHAIN_TRACING_V2",   "true" if settings.langchain_tracing_v2 else "false")
        log.info("langsmith.configured", project=settings.langchain_project,
                 tracing=settings.langchain_tracing_v2)

    _state.repo = DocumentRepository(settings.postgres_url)
    _state.retriever = Retriever(_state.repo, settings.embed_model)
    _state.pipeline = RetrievalPipeline(_state.repo, _state.retriever)
    _state.rag = RAGPipeline(_state.retriever)
    factory      = NodeFactory(repo=_state.repo, retriever=_state.retriever)
    analytical   = AnalyticalNodeFactory(repo=_state.repo)

    sql_factory = None
    if settings.llm_provider == "anthropic" and settings.openai_api_key:
        try:
            from financial_pipeline.graph.nodes_sql import SQLNodeFactory
            from financial_pipeline.text_to_sql.vanna_agent import build_vanna_agent

            vanna = build_vanna_agent(
                anthropic_api_key=settings.openai_api_key,
                postgres_url=settings.postgres_url,
            )
            sql_factory = SQLNodeFactory(vanna=vanna)
            log.info("api.sql_agent_ready")
        except ModuleNotFoundError as exc:
            log.info("api.sql_agent_unavailable", error=str(exc))
        except Exception as exc:
            log.warning("api.sql_agent_failed", error=str(exc))

    # ── Postgres checkpointer — every node's state persisted per thread_id ──
    if settings.enable_checkpointing:
        try:
            _state.checkpointer = build_checkpointer(settings.postgres_url)
        except Exception as exc:
            log.warning("api.checkpointer_failed", error=str(exc))
            _state.checkpointer = None
    else:
        log.info("api.checkpointer_disabled")

    # ── query_log — flat request/answer record, keyed by query_id, for
    # feedback (POST /api/feedback) and eval linkage. Shares repo's engine —
    # this table lives in the same Postgres instance, no reason for a
    # second connection pool.
    try:
        _state.query_log = QueryLogRepository(_state.repo._engine)
        _state.query_log.create_tables()
    except Exception as exc:
        log.warning("api.query_log_unavailable", error=str(exc))
        _state.query_log = None

    _state.graph = build_graph(
        factory, analytical=analytical, sql=sql_factory, checkpointer=_state.checkpointer,
    )
    log.info(
        "api.ready",
        embed_model=settings.embed_model,
        llm=f"{settings.llm_provider}/{settings.active_llm_model}",
        graph="langgraph",
        sql_agent=sql_factory is not None,
        checkpointing=_state.checkpointer is not None,
        query_log=_state.query_log is not None,
    )
    flush_task = asyncio.create_task(_periodic_metrics_flush())
    yield

    flush_task.cancel()
    try:
        await flush_task
    except asyncio.CancelledError:
        pass
    emitter.flush()  # final flush — don't lose whatever's still buffered

    if _state.checkpointer is not None:
        # PostgresSaver.conn is the ConnectionPool build_checkpointer() created —
        # close it explicitly so its worker threads shut down cleanly instead
        # of racing the process exit.
        try:
            _state.checkpointer.conn.close()
        except Exception as exc:
            log.warning("api.checkpointer_close_failed", error=str(exc))

    log.info("api.shutdown")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="AMFI Financial Document Retrieval API",
    description=(
        "Semantic + keyword search over 440 AMFI documents (2009–2025) "
        "with optional RAG-powered Q&A using any OpenAI-compatible LLM."
    ),
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api_cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_repo() -> DocumentRepository:
    if _state.repo is None:
        raise HTTPException(503, "Repository not initialised")
    return _state.repo


def _get_retriever() -> Retriever:
    if _state.retriever is None:
        raise HTTPException(503, "Retriever not initialised")
    return _state.retriever


def _get_pipeline() -> RetrievalPipeline:
    if _state.pipeline is None:
        raise HTTPException(503, "Retrieval pipeline not initialised")
    return _state.pipeline


def _get_rag() -> RAGPipeline:
    if _state.rag is None:
        raise HTTPException(503, "RAG pipeline not initialised")
    return _state.rag


# ── Routes ────────────────────────────────────────────────────────────────────


@app.get("/metrics", tags=["ops"], include_in_schema=False)
def metrics() -> Response:
    """Prometheus metrics endpoint — scrape with Prometheus or view in browser."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/healthz", tags=["ops"])
def healthz() -> dict[str, str]:
    """Liveness probe — always returns 200 if the process is alive."""
    return {"status": "ok"}


@app.get("/readyz", tags=["ops"])
def readyz() -> dict[str, Any]:
    """Readiness probe — confirms DB is reachable."""
    try:
        repo = _get_repo()
        with repo._engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return {"status": "ready", "db": "ok"}
    except Exception as exc:
        raise HTTPException(503, f"DB not ready: {exc}")


@app.post("/api/search", response_model=SearchResponse, tags=["search"])
def search(req: SearchRequest) -> SearchResponse:
    """5-stage intelligent retrieval pipeline.

    **Stage 1 — Query Understanding**: extracts year, month, intent type
    (factual / trend / tabular / comparison / definition / lookup), scheme
    types, and financial entities from the query without calling an LLM.

    **Stage 2 — Search Routing**: selects which sources to query (chunks,
    table assets, document metadata) and which mode (hybrid/semantic/keyword)
    based on the detected intent.

    **Stage 3 — Multi-Source Fetch**: queries pgvector chunks + optionally
    `document_table_assets` and `document_metadata`.

    **Stage 4 — Rank / Filter**: deduplicates by chunk_id, applies MMR
    diversity to avoid over-weighting one document, filters by score.

    **Stage 5 — Context Assembly**: numbers results as citations [1]…[N]
    with source metadata for grounded downstream use.
    """
    pipeline = _get_pipeline()
    try:
        ctx = pipeline.retrieve(
            query=req.query,
            top_k=req.limit,
            year=req.year,
            month=req.month,
            category=req.category,
            mode=req.mode if req.mode != "hybrid" else None,
        )
    except Exception as exc:
        log.exception("api.search_error", error=str(exc))
        raise HTTPException(500, str(exc))

    results = [
        ChunkResult(
            chunk_id=str(r.get("chunk_id", "")),
            chunk_index=r.get("chunk_index", 0) or 0,
            text=r.get("text") or r.get("table_name") or "",
            file_name=r.get("file_name"),
            period_year=r.get("period_year"),
            period_month=r.get("period_month"),
            category=r.get("category"),
            similarity=r.get("similarity"),
            rrf_score=r.get("rrf_score") or r.get("_mmr_score"),
            search_mode=r.get("_source", r.get("search_mode")),
        )
        for r in ctx.raw_results
    ]
    return SearchResponse(
        query=req.query,
        mode=ctx.intent.intent_type,  # return detected intent, not just mode
        count=len(results),
        results=results,
    )


def _record_observability(trace: RequestTrace, run_id: str, guardrail: dict) -> None:
    """Runs as a FastAPI background task, after the response has already
    been sent — emitter.record()/flush() and ls_feedback.record_case() make
    real network calls (CloudWatch PutMetricData, LangSmith create_feedback
    respectively), so this must never run inline in the request path.
    Mirrors evaluation/runner.py's exact call sequence (same RequestTrace
    shape, same emitter/ls_feedback calls) — this is what makes live /api/ask
    traffic produce the same traces/costs/latency/retrieval diagnostics the
    offline eval suite already did, instead of only the eval suite.
    """
    try:
        emitter.record(trace)
        # Only the production-meaningful feedback keys are populated — the
        # eval-only ground-truth keys (citation_coverage, keyword_recall,
        # numeric_precision, expected_num_hit, abstention_correct) have no
        # live equivalent for a real user question with no expected answer,
        # and record_case already skips None values for the rest (see
        # observability.py's _LS_FEEDBACK_KEYS loop).
        ls_feedback.record_case(run_id, {
            "pre_blocked": guardrail.get("blocked"),
            "post_passed": guardrail.get("post_passed"),
        })
    except Exception as exc:  # noqa: BLE001 — best-effort, never affects an already-sent response
        log.warning("api.observability_failed", error=str(exc), run_id=run_id)


@app.post("/api/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest, background_tasks: BackgroundTasks) -> AskResponse:
    """6-stage Augmentation Pipeline.

    1. Retrieve  — intent-aware hybrid search (pgvector + BM25 + MMR)
    2. Re-rank   — CrossEncoder (ms-marco-MiniLM-L-12-v2) joint scoring
    3. Cite      — numbered citations [1]…[N] with confidence scores
    4. Prompt    — intent-aware template (regulatory / factual / trend / …)
    5. Generate  — Claude / OpenAI with temperature tuned per intent
    6. Guardrails— citation validity, number consistency, faithfulness score
    """
    if _state.graph is None:
        raise HTTPException(503, "LangGraph not initialised.")

    # thread_id — the LangGraph checkpoint thread this invocation's node-by-node
    # state is persisted under (single-turn today: fresh thread per request; see
    # graph/graph.py's build_graph docstring for the multi-turn follow-up path).
    # query_id — stable id for this specific answer, independent of thread_id,
    # returned to the client for POST /api/feedback and eval linkage.
    # run_id — ties this invocation to its LangSmith trace (same RunnableConfig
    # pattern evaluation/runner.py uses) so _record_observability's feedback
    # attaches to the right run.
    thread_id = str(uuid.uuid4())
    query_id  = str(uuid.uuid4())
    run_id    = uuid.uuid4()
    t0 = time.time()

    try:
        result = _state.graph.invoke(
            {"query": req.question, "retry_count": 0, "repair_count": 0},
            config=RunnableConfig(run_id=run_id, configurable={"thread_id": thread_id}),
        )
    except Exception as exc:
        log.exception("api.ask_error", error=str(exc), thread_id=thread_id, query_id=query_id)
        raise HTTPException(500, str(exc))

    resp_dict = result.get("response", {})
    g         = resp_dict.get("guardrail", {})
    intent    = result.get("intent")
    intent_type = getattr(intent, "intent_type", "unknown") or "unknown"
    stage_used  = getattr(intent, "stage_used",  "rule")    or "rule"

    # ── Prometheus metrics ─────────────────────────────────────────────
    latency = resp_dict.get("latency_ms", 0) or 0
    QUERIES_TOTAL.labels(intent_type=intent_type).inc()
    QUERY_LATENCY.labels(intent_type=intent_type).observe(latency)
    INTENT_STAGE.labels(stage=stage_used).inc()
    if resp_dict.get("prompt_tokens"):
        LLM_TOKENS.labels(token_type="prompt").inc(resp_dict["prompt_tokens"])
    if resp_dict.get("completion_tokens"):
        LLM_TOKENS.labels(token_type="completion").inc(resp_dict["completion_tokens"])
    if g.get("blocked"):
        GUARDRAIL_BLOCKS.inc()
    if getattr(intent, "needs_analytical", False):
        ANALYTICAL_QUERIES.inc()

    # ── Live observability — same RequestTrace shape evaluation/runner.py
    # builds for the offline eval suite, now scheduled for real /api/ask
    # traffic too. Gated on prompt_tokens/model like runner.py's own gate
    # (runner.py:101) — a blocked/abstained query with no generation call
    # has nothing meaningful to trace here.
    if resp_dict.get("prompt_tokens") and resp_dict.get("model"):
        trace = RequestTrace(
            request_id=str(run_id),
            question=req.question[:200],
            intent=intent_type,
            model=resp_dict.get("model", ""),
            provider=resp_dict.get("provider", ""),
            prompt_tokens=resp_dict.get("prompt_tokens", 0),
            completion_tokens=resp_dict.get("completion_tokens", 0),
            total_ms=latency,
            pre_blocked=bool(g.get("blocked", False)),
            post_passed=bool(g.get("post_passed", True)),
            answer_safe=bool(g.get("answer_safe", True)),
            abstention_detected=bool(g.get("abstention_detected", False)),
            faithfulness_score=(fs if (fs := g.get("faithfulness_score")) is not None else -1.0),
            citation_count=resp_dict.get("retrieval_count", 0),
        )
        background_tasks.add_task(_record_observability, trace, str(run_id), g)

    log.info("api.ask_done",
             grade=result.get("grade"),
             retry_count=result.get("retry_count"),
             citations=resp_dict.get("retrieval_count", 0),
             intent=intent_type,
             stage=stage_used,
             latency_ms=latency,
             thread_id=thread_id,
             query_id=query_id)

    # ── query_log — best-effort; a logging failure must never fail the
    # request that already has a good answer for the user.
    if _state.query_log is not None:
        # Mirrors is_range_query's branch priority (edges_analytical.py) —
        # not read from graph state, since routing is a conditional-edge
        # decision, not a stored field.
        if getattr(intent, "needs_reasoning", False):
            route = "reasoning"
        elif getattr(intent, "needs_analytical", False):
            route = "plan_years"
        elif intent_type == "tabular":
            route = "query_sql"
        else:
            route = "route"
        try:
            _state.query_log.log_query(
                query_id=query_id,
                thread_id=thread_id,
                question=req.question,
                answer=resp_dict.get("answer", ""),
                intent_type=intent_type,
                route=route,
                citations_count=resp_dict.get("retrieval_count", 0),
                latency_ms=latency or int((time.time() - t0) * 1000),
                blocked=bool(g.get("blocked", False)),
            )
        except Exception as exc:
            log.warning("api.query_log_write_failed", error=str(exc), query_id=query_id)

    return AskResponse(
        question=req.question,
        answer=resp_dict.get("answer", ""),
        sources=[
            Source(
                citation=s["citation"],
                file_name=s.get("file_name"),
                period_year=s.get("period_year"),
                period_month=s.get("period_month"),
                category=s.get("category"),
                preview=s.get("preview", ""),
            )
            for s in resp_dict.get("sources", [])
        ],
        model=resp_dict.get("model", ""),
        latency_ms=resp_dict.get("latency_ms", 0),
        prompt_tokens=resp_dict.get("prompt_tokens", 0),
        completion_tokens=resp_dict.get("completion_tokens", 0),
        retrieval_count=resp_dict.get("retrieval_count", 0),
        guardrail=GuardrailReport(
            pre_passed=g.get("pre_passed", True),
            post_passed=g.get("post_passed", True),
            blocked=g.get("blocked", False),
            block_reason=g.get("block_reason"),
            is_investment_advice=g.get("is_investment_advice", False),
            answer_safe=g.get("answer_safe", True),
            hallucination_risk=g.get("hallucination_risk", "unknown"),
            # post_guardrail sets this key to None (not absent) for extractive/
            # deterministic answers where faithfulness scoring doesn't apply —
            # .get()'s default only fires on a *missing* key, so None must be
            # coalesced explicitly or pydantic's float validation rejects it.
            faithfulness_score=(fs if (fs := g.get("faithfulness_score")) is not None else -1.0),
            citation_valid=g.get("citation_valid", True),
            number_consistent=g.get("number_consistent", True),
            abstention_detected=g.get("abstention_detected", False),
            warnings=g.get("warnings", []),
        ),
        thread_id=thread_id,
        query_id=query_id,
    )


@app.post("/api/feedback", response_model=FeedbackResponse, tags=["rag"])
def feedback(req: FeedbackRequest) -> FeedbackResponse:
    """Attach a rating (1-5) and optional comment to a prior /api/ask answer,
    keyed by the query_id that endpoint returned."""
    if _state.query_log is None:
        raise HTTPException(503, "Query log not initialised.")

    query_id = str(req.query_id)
    recorded = _state.query_log.record_feedback(
        query_id=query_id, rating=req.rating, comment=req.comment,
    )
    if not recorded:
        raise HTTPException(404, f"query_id {query_id!r} not found.")

    log.info("api.feedback_recorded", query_id=query_id, rating=req.rating)
    return FeedbackResponse(query_id=query_id, recorded=True)


@app.get("/api/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(
    category: str | None = Query(None, description="monthly | quarterly | unknown"),
    year: int | None = Query(None),
    status: str | None = Query(None, description="Filter by processing_status"),
    limit: int = Query(default=50, le=500),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    """List ingested documents with optional filters."""
    repo = _get_repo()

    conditions = ["1=1"]
    params: dict = {"limit": limit, "offset": offset}

    if category:
        conditions.append("category = :category")
        params["category"] = category
    if year:
        conditions.append("period_year = :year")
        params["year"] = year
    if status:
        conditions.append("processing_status = :status")
        params["status"] = status

    where = " AND ".join(conditions)
    with repo._engine.connect() as conn:
        total = (
            conn.execute(
                text(f"SELECT count(*) FROM document_metadata WHERE {where}"),
                {k: v for k, v in params.items() if k not in ("limit", "offset")},
            ).scalar()
            or 0
        )

        rows = (
            conn.execute(
                text(f"""
                SELECT document_id, file_name, file_type, category,
                       period_year, period_month, processing_status, schema_version
                  FROM document_metadata
                 WHERE {where}
                 ORDER BY period_year DESC NULLS LAST, period_month DESC NULLS LAST
                 LIMIT :limit OFFSET :offset
            """),
                params,
            )
            .mappings()
            .all()
        )

    return DocumentListResponse(
        total=total,
        documents=[
            DocumentSummary(
                document_id=str(r["document_id"]),
                file_name=r["file_name"],
                file_type=r["file_type"],
                category=r["category"],
                period_year=r["period_year"],
                period_month=r["period_month"],
                processing_status=r["processing_status"],
                schema_version=r["schema_version"],
            )
            for r in rows
        ],
    )


@app.get("/api/stats", response_model=PipelineStats, tags=["ops"])
def stats() -> PipelineStats:
    """Pipeline health: queue depths, chunk count, LLM readiness."""
    repo = _get_repo()
    rag = _get_rag()
    depths = repo.queue_depths()
    chunks = repo.chunk_count()
    total = sum(depths.values())

    return PipelineStats(
        total_documents=total,
        embedded=depths.get("embedded", 0),
        total_chunks=chunks,
        queue_depths=depths,
        embed_model=settings.embed_model,
        llm_model=f"{settings.llm_provider}/{settings.active_llm_model}",
        llm_configured=rag.is_llm_configured(),
    )
