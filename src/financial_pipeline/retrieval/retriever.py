"""Retriever — single entry point for all search modes.

Wraps the three search paths in document_repo and handles:
  - Lazy model loading (sentence-transformers loaded once, reused across requests)
  - Embedding caching (same query string → same vector, no re-encode)
  - Unified result schema regardless of search mode
  - Hybrid RRF fusion across semantic + keyword results
"""

from __future__ import annotations

import functools
from typing import Any

import structlog

from financial_pipeline.config import settings
from financial_pipeline.storage.document_repo import DocumentRepository

log = structlog.get_logger()

_model_cache: dict[str, Any] = {}


def _get_model(model_name: str):
    if model_name not in _model_cache:
        from sentence_transformers import SentenceTransformer

        log.info("retriever.model_loading", model=model_name)
        _model_cache[model_name] = SentenceTransformer(
            model_name,
            token=settings.hf_token or None,
        )
        log.info("retriever.model_ready", model=model_name)
    return _model_cache[model_name]


def _rrf_merge(
    semantic: list[dict],
    keyword: list[dict],
    limit: int,
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion (Cormack et al. 2009)."""
    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}

    for rank, r in enumerate(semantic, 1):
        cid = str(r["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        meta[cid] = r

    for rank, r in enumerate(keyword, 1):
        cid = str(r["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in meta:
            meta[cid] = r

    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)[:limit]
    return [meta[c] | {"rrf_score": round(scores[c], 6), "search_mode": "hybrid"} for c in ranked]


class Retriever:
    """High-level search interface for the AMFI knowledge base.

    Usage::

        retriever = Retriever(repo)
        results   = retriever.search("balanced advantage fund returns 2024",
                                      mode="hybrid", limit=8, year=2024)
    """

    def __init__(
        self,
        repo: DocumentRepository,
        model_name: str | None = None,
    ) -> None:
        self._repo = repo
        self._model_name = model_name or settings.embed_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        mode: str = "hybrid",  # "hybrid" | "semantic" | "keyword"
        limit: int = 10,
        year: int | None = None,
        month: int | None = None,
        category: str | None = None,
        min_sim: float = 0.0,
    ) -> list[dict]:
        """Return ranked chunks for *query* using the requested mode."""
        if not query.strip():
            return []

        semantic: list[dict] = []
        keyword: list[dict] = []

        if mode in ("semantic", "hybrid"):
            vec = self._encode(query)
            semantic = self._repo.search_similar(
                query_embedding=vec,
                limit=limit * 2 if mode == "hybrid" else limit,
                period_year=year,
                period_month=month,
                category=category,
                min_similarity=min_sim,
            )
            for r in semantic:
                r["search_mode"] = "semantic"

        if mode in ("keyword", "hybrid"):
            keyword = self._repo.search_fulltext(
                query=query,
                limit=limit * 2 if mode == "hybrid" else limit,
                period_year=year,
                category=category,
            )
            for r in keyword:
                r["search_mode"] = "keyword"

        if mode == "hybrid":
            return _rrf_merge(semantic, keyword, limit)
        elif mode == "semantic":
            return semantic[:limit]
        else:
            return keyword[:limit]

    def get_context_chunks(
        self,
        query: str,
        limit: int = 8,
        **filters,
    ) -> list[dict]:
        """Convenience wrapper used by RAGPipeline — hybrid search, top-k."""
        return self.search(query, mode="hybrid", limit=limit, **filters)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=512)
    def _encode(self, text: str) -> list[float]:
        """Encode *text* to a normalised embedding vector (cached by text)."""
        model = _get_model(self._model_name)
        vector = model.encode(text, normalize_embeddings=True)
        return vector.tolist()
