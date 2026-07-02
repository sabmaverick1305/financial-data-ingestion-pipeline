"""Retrieval + Reranker Quality Evaluator (Dimensions 1 & 2).

Runs each eval case through the retrieval pipeline twice:
  - Once with bi-encoder only  (pre cross-encoder)
  - Once with cross-encoder    (post cross-encoder)

Then computes Hit@k, MRR, NDCG@10, and rank-improvement metrics to measure
how much the cross-encoder re-ranker improves retrieval quality.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from financial_pipeline.evaluation.dataset import EvalCase
from financial_pipeline.evaluation.metrics import (
    hit_at_k,
    ndcg_at_k,
    reranker_promoted,
    reranker_rank_improvement,
    reciprocal_rank,
)

log = structlog.get_logger()


@dataclass
class RetrievalCaseResult:
    case_id:          str
    question:         str
    relevant_files:   list[str]

    # Pre cross-encoder (bi-encoder + RRF fusion only)
    pre_files:        list[str] = field(default_factory=list)
    pre_hit_5:        float     = 0.0
    pre_hit_10:       float     = 0.0
    pre_mrr:          float     = 0.0
    pre_ndcg_10:      float     = 0.0
    pre_latency_ms:   int       = 0

    # Post cross-encoder
    post_files:       list[str] = field(default_factory=list)
    post_hit_5:       float     = 0.0
    post_hit_10:      float     = 0.0
    post_mrr:         float     = 0.0
    post_ndcg_10:     float     = 0.0
    post_latency_ms:  int       = 0

    # Reranker delta
    rank_improvement: float     = 0.0
    ce_promoted:      bool      = False

    error: str | None = None


@dataclass
class RetrievalEvalSummary:
    cases: list[RetrievalCaseResult] = field(default_factory=list)

    # Pre CE averages
    pre_hit_5:    float = 0.0
    pre_hit_10:   float = 0.0
    pre_mrr:      float = 0.0
    pre_ndcg_10:  float = 0.0

    # Post CE averages
    post_hit_5:   float = 0.0
    post_hit_10:  float = 0.0
    post_mrr:     float = 0.0
    post_ndcg_10: float = 0.0

    # Reranker deltas
    ce_avg_improvement: float = 0.0
    ce_promotion_rate:  float = 0.0
    avg_latency_ms:     float = 0.0


class RetrievalEvaluator:
    """Evaluates retrieval quality (Dims 1 + 2) over the eval dataset.

    Parameters
    ----------
    retriever:
        The Retriever instance (bi-encoder search, pgvector).
    ranker:
        The ContextRanker instance (cross-encoder re-ranking).
    top_k:
        Number of results to retrieve and evaluate at.
    """

    def __init__(self, retriever, ranker, top_k: int = 10) -> None:
        self._retriever = retriever
        self._ranker    = ranker
        self._top_k     = top_k

    def run(self, cases: list[EvalCase]) -> RetrievalEvalSummary:
        results: list[RetrievalCaseResult] = []

        # Skip cases that don't have relevance judgements
        retrieval_cases = [c for c in cases if c.relevant_files and not c.is_investment_advice]
        log.info("retrieval_eval.start", cases=len(retrieval_cases))

        for case in retrieval_cases:
            result = self._eval_one(case)
            results.append(result)
            log.debug(
                "retrieval_eval.case",
                id=case.id,
                pre_hit5=result.pre_hit_5,
                post_hit5=result.post_hit_5,
                ce_gain=result.rank_improvement,
            )

        return self._summarise(results)

    def _eval_one(self, case: EvalCase) -> RetrievalCaseResult:
        result = RetrievalCaseResult(
            case_id       = case.id,
            question      = case.question,
            relevant_files= case.relevant_files,
        )

        try:
            # ── Pre CE: bi-encoder + RRF only ──────────────────────────
            t0  = time.perf_counter()
            raw = self._retriever.search(
                query    = case.question,
                mode     = "hybrid",
                limit    = self._top_k * 2,   # fetch more for pre-CE eval
                category = case.category,
            )
            result.pre_latency_ms = int((time.perf_counter() - t0) * 1000)
            result.pre_files      = [r.get("file_name", "") for r in raw]

            result.pre_hit_5   = hit_at_k(result.pre_files, case.relevant_files, k=5)
            result.pre_hit_10  = hit_at_k(result.pre_files, case.relevant_files, k=10)
            result.pre_mrr     = reciprocal_rank(result.pre_files, case.relevant_files)
            result.pre_ndcg_10 = ndcg_at_k(result.pre_files, case.relevant_files, k=10)

            # ── Post CE: cross-encoder re-ranking ──────────────────────
            t1        = time.perf_counter()
            reranked  = self._ranker.rerank(case.question, raw[: self._top_k])
            result.post_latency_ms = int((time.perf_counter() - t1) * 1000)
            result.post_files      = [r.get("file_name", "") for r in reranked]

            result.post_hit_5   = hit_at_k(result.post_files, case.relevant_files, k=5)
            result.post_hit_10  = hit_at_k(result.post_files, case.relevant_files, k=10)
            result.post_mrr     = reciprocal_rank(result.post_files, case.relevant_files)
            result.post_ndcg_10 = ndcg_at_k(result.post_files, case.relevant_files, k=10)

            # ── Reranker delta ─────────────────────────────────────────
            result.rank_improvement = reranker_rank_improvement(
                result.pre_files, result.post_files, case.relevant_files
            )
            result.ce_promoted = reranker_promoted(
                result.pre_files, result.post_files, case.relevant_files, top_k=5
            )

        except Exception as exc:
            result.error = str(exc)
            log.warning("retrieval_eval.case_failed", id=case.id, error=str(exc))

        return result

    def _summarise(self, results: list[RetrievalCaseResult]) -> RetrievalEvalSummary:
        ok = [r for r in results if not r.error]
        if not ok:
            return RetrievalEvalSummary(cases=results)

        def avg(vals: list[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0

        summary = RetrievalEvalSummary(
            cases          = results,
            pre_hit_5      = avg([r.pre_hit_5    for r in ok]),
            pre_hit_10     = avg([r.pre_hit_10   for r in ok]),
            pre_mrr        = avg([r.pre_mrr      for r in ok]),
            pre_ndcg_10    = avg([r.pre_ndcg_10  for r in ok]),
            post_hit_5     = avg([r.post_hit_5   for r in ok]),
            post_hit_10    = avg([r.post_hit_10  for r in ok]),
            post_mrr       = avg([r.post_mrr     for r in ok]),
            post_ndcg_10   = avg([r.post_ndcg_10 for r in ok]),
            ce_avg_improvement = avg([r.rank_improvement for r in ok]),
            ce_promotion_rate  = avg([float(r.ce_promoted) for r in ok]),
            avg_latency_ms     = avg([r.pre_latency_ms + r.post_latency_ms for r in ok]),
        )
        return summary
