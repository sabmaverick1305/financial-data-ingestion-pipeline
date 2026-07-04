from __future__ import annotations

from typing import TypedDict

from financial_pipeline.augmentation.citations import Citation
from financial_pipeline.augmentation.guardrails import (
    PostGuardrailResult,
    PreGuardrailResult,
)
from financial_pipeline.retrieval.query_understanding import QueryIntent


class RAGState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────
    query:             str
    retry_count:       int         # CRAG loop guard  (0 → 1 → 2 → abstain)
    repair_count:      int         # repair loop guard (0 → 1 → abstain)

    # ── analyze_query ──────────────────────────────────────
    intent:            QueryIntent
    active_branches:   list[str]   # ["dense","sparse","table","metadata"]

    # ── sequential lookup path ────────────────────────────
    found_document_ids: list[str]   # set by retrieve_metadata_first;
                                    # empty = not found → fallback to parallel

    # ── parallel retrieval ─────────────────────────────────
    dense_results:     list[dict]
    sparse_results:    list[dict]
    table_results:     list[dict]
    metadata_results:  list[dict]

    # ── rrf_fusion ─────────────────────────────────────────
    fused_results:     list[dict]

    # ── rerank ─────────────────────────────────────────────
    reranked_results:  list[dict]

    # ── context_optimizer ──────────────────────────────────
    optimized_results: list[dict]
    optimizer_stats:   dict        # dedup/diversity/recency/fallback reason

    # ── grade_context ──────────────────────────────────────
    grade:             str         # "sufficient" | "retry" | "abstain"
    rewritten_query:   str | None

    # ── augment ────────────────────────────────────────────
    citations:         list[Citation]
    context_text:      str

    # ── pre_guardrail ──────────────────────────────────────
    pre_result:        PreGuardrailResult
    blocked:           bool

    # ── generate ───────────────────────────────────────────
    answer:            str
    generation_meta:   dict        # model, provider, prompt_tokens,
                                   # completion_tokens, latency_ms

    # ── post_guardrail / repair ────────────────────────────
    post_result:       PostGuardrailResult
    repair_attempted:  bool

    # ── format_response ────────────────────────────────────
    response:          dict        # serialisable AskResponse payload
