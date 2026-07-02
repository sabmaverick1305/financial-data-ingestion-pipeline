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

from contextlib import asynccontextmanager
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from financial_pipeline.api.schemas import (
    AskRequest,
    AskResponse,
    ChunkResult,
    DocumentListResponse,
    DocumentSummary,
    GuardrailReport,
    PipelineStats,
    SearchRequest,
    SearchResponse,
    Source,
)
from financial_pipeline.augmentation.pipeline import AugmentationPipeline
from financial_pipeline.config import settings
from financial_pipeline.evaluation.observability import RequestTrace, emitter
from financial_pipeline.retrieval.pipeline import RetrievalPipeline
from financial_pipeline.retrieval.rag import RAGPipeline
from financial_pipeline.retrieval.retriever import Retriever
from financial_pipeline.storage.document_repo import DocumentRepository

log = structlog.get_logger()

# ── Application state (initialised once at startup) ───────────────────────────


class _AppState:
    repo: DocumentRepository | None = None
    retriever: Retriever | None = None
    pipeline: RetrievalPipeline | None = None
    augment: AugmentationPipeline | None = None
    rag: RAGPipeline | None = None  # kept for fallback


_state = _AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("api.startup")
    _state.repo = DocumentRepository(settings.postgres_url)
    _state.retriever = Retriever(_state.repo, settings.embed_model)
    _state.pipeline = RetrievalPipeline(_state.repo, _state.retriever)
    _state.augment = AugmentationPipeline(
        _state.pipeline,
        use_cross_encoder=True,
        top_k_retrieve=12,
        top_k_augment=6,
    )
    _state.rag = RAGPipeline(_state.retriever)  # fallback only
    log.info(
        "api.ready",
        embed_model=settings.embed_model,
        llm=f"{settings.llm_provider}/{settings.active_llm_model}",
    )
    yield
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


def _get_augment() -> AugmentationPipeline:
    if _state.augment is None:
        raise HTTPException(503, "Augmentation pipeline not initialised")
    return _state.augment


def _get_rag() -> RAGPipeline:
    if _state.rag is None:
        raise HTTPException(503, "RAG pipeline not initialised")
    return _state.rag


# ── Routes ────────────────────────────────────────────────────────────────────


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


@app.post("/api/ask", response_model=AskResponse, tags=["rag"])
def ask(req: AskRequest) -> AskResponse:
    """6-stage Augmentation Pipeline.

    1. Retrieve  — intent-aware hybrid search (pgvector + BM25 + MMR)
    2. Re-rank   — CrossEncoder (ms-marco-MiniLM-L-6-v2) joint scoring
    3. Cite      — numbered citations [1]…[N] with confidence scores
    4. Prompt    — intent-aware template (regulatory / factual / trend / …)
    5. Generate  — Claude / OpenAI with temperature tuned per intent
    6. Guardrails— citation validity, number consistency, faithfulness score
    """
    augment = _get_augment()
    if not augment.is_configured():
        raise HTTPException(501, "LLM not configured. Set OPENAI_API_KEY (or Anthropic sk-ant-...) in .env.")

    try:
        resp = augment.run(
            question=req.question,
            year=req.year,
            month=req.month,
            category=req.category,
            model=req.model,
            top_k=req.top_k,
        )
    except ValueError as exc:
        raise HTTPException(501, str(exc))
    except Exception as exc:
        log.exception("api.ask_error", error=str(exc))
        raise HTTPException(500, str(exc))

    # Emit per-request observability trace to CloudWatch
    g = resp.guardrail
    trace = RequestTrace.from_augmented_response(resp, request_id=str(id(resp)), question=req.question)
    emitter.record(trace)
    log.info("api.ask_trace", **trace.to_log_dict())

    return AskResponse(
        question=req.question,
        answer=resp.answer,
        sources=[
            Source(
                citation=c.number,
                file_name=c.file_name,
                period_year=c.period_year,
                period_month=c.period_month,
                category=c.category,
                preview=c.excerpt[:200],
            )
            for c in resp.citations
        ],
        model=resp.generation.model if resp.generation else "",
        latency_ms=resp.augment_stats.get("total_latency_ms", 0),
        prompt_tokens=resp.generation.prompt_tokens if resp.generation else 0,
        completion_tokens=resp.generation.completion_tokens if resp.generation else 0,
        retrieval_count=resp.augment_stats.get("chunks_retrieved", 0),
        guardrail=GuardrailReport(
            pre_passed=g.pre_passed,
            post_passed=g.post_passed,
            blocked=g.blocked,
            block_reason=g.block_reason,
            is_investment_advice=g.is_investment_advice,
            answer_safe=g.answer_safe,
            hallucination_risk=g.hallucination_risk,
            faithfulness_score=g.faithfulness_score,
            citation_valid=g.citation_valid,
            number_consistent=g.number_consistent,
            abstention_detected=g.abstention_detected,
            warnings=g.warnings,
        ),
    )


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
