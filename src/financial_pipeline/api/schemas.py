"""Pydantic request / response schemas for the retrieval API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ── Search ────────────────────────────────────────────────────────────────────


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=500)
    mode: Literal["hybrid", "semantic", "keyword"] = "hybrid"
    limit: int = Field(default=8, ge=1, le=50)
    year: int | None = None
    month: int | None = Field(default=None, ge=1, le=12)
    category: Literal["monthly", "quarterly", "unknown"] | None = None
    min_sim: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum cosine similarity (semantic/hybrid only)")


class ChunkResult(BaseModel):
    chunk_id: str
    chunk_index: int
    text: str
    file_name: str | None
    period_year: int | None
    period_month: int | None
    category: str | None
    similarity: float | None = None
    rrf_score: float | None = None
    search_mode: str | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str
    count: int
    results: list[ChunkResult]


# ── RAG / Ask ─────────────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    year: int | None = None
    month: int | None = Field(default=None, ge=1, le=12)
    category: Literal["monthly", "quarterly", "unknown"] | None = None
    top_k: int = Field(default=8, ge=1, le=20)
    model: str | None = Field(default=None, description="Override LLM model for this request")


class Source(BaseModel):
    citation: int
    file_name: str | None
    period_year: int | None
    period_month: int | None
    category: str | None
    preview: str


class GuardrailReport(BaseModel):
    """Dual-layer guardrail result exposed in the API response."""

    pre_passed: bool
    post_passed: bool
    blocked: bool
    block_reason: str | None = None
    is_investment_advice: bool = False
    answer_safe: bool = True
    hallucination_risk: str = "unknown"  # low | medium | high
    faithfulness_score: float = -1.0
    citation_valid: bool = True
    number_consistent: bool = True
    abstention_detected: bool = False
    warnings: list[str] = []


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    model: str
    latency_ms: int
    prompt_tokens: int
    completion_tokens: int
    retrieval_count: int
    guardrail: GuardrailReport | None = None


# ── Documents ─────────────────────────────────────────────────────────────────


class DocumentSummary(BaseModel):
    document_id: str
    file_name: str | None
    file_type: str | None
    category: str | None
    period_year: int | None
    period_month: int | None
    processing_status: str
    schema_version: str | None


class DocumentListResponse(BaseModel):
    total: int
    documents: list[DocumentSummary]


# ── Stats ─────────────────────────────────────────────────────────────────────


class PipelineStats(BaseModel):
    total_documents: int
    embedded: int
    total_chunks: int
    queue_depths: dict[str, int]
    embed_model: str
    llm_model: str
    llm_configured: bool
