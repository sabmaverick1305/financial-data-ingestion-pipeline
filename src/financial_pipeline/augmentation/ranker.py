"""Context Ranker — Stage 1 of the augmentation layer.

Re-ranks retrieved chunks using a cross-encoder model that jointly encodes
the query and each chunk (unlike the bi-encoder used in retrieval, which
encodes them independently). Cross-encoders are slower but significantly
more accurate at judging relevance.

Model: cross-encoder/ms-marco-MiniLM-L-12-v2
  - 120 MB, CPU-friendly
  - Trained on MS MARCO passage retrieval
  - Outputs a relevance score per (query, passage) pair

Fallback: If the cross-encoder is not available or disabled, falls back to
the original retrieval scores (similarity / rrf_score / _mmr_score).
"""

from __future__ import annotations

import structlog

log = structlog.get_logger()

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"
_ce_cache: dict = {}


def _get_cross_encoder(model_name: str = CROSS_ENCODER_MODEL):
    if model_name not in _ce_cache:
        from sentence_transformers import CrossEncoder

        log.info("ranker.cross_encoder_loading", model=model_name)
        _ce_cache[model_name] = CrossEncoder(model_name)
        log.info("ranker.cross_encoder_ready", model=model_name)
    return _ce_cache[model_name]


class ContextRanker:
    """Cross-encoder re-ranker — Stage 3 of the hybrid retrieval pipeline.

    Operates on a shortlist of candidates (produced by Stage 1 + Stage 2)
    rather than the full corpus. This keeps latency constant regardless of
    corpus size:

        24 K chunks  → bi-encoder fetches 12  → CE re-ranks 12  (~50 ms)
        15 M chunks  → dual retrieval fetches 50-75 → CE re-ranks 75 (~300 ms)

    The cross-encoder jointly encodes [query, chunk] and outputs a relevance
    logit — it can see the interaction between query terms and passage content
    in a way the bi-encoder (which encodes them independently) cannot.

    Parameters
    ----------
    use_cross_encoder:
        True (default) = accurate. False = fast fallback using retrieval scores.
    top_k:
        Chunks to return after re-ranking.
    min_score:
        Drop chunks below this cross-encoder logit. Typical range: -10 to +10.
        Empirically, scores > 0 are highly relevant; < -3 are not relevant.
    model_name:
        Any sentence-transformers cross-encoder. Alternatives:
          - cross-encoder/ms-marco-MiniLM-L-12-v2  (larger, more accurate)
          - cross-encoder/ms-marco-electra-base     (best accuracy, slowest)
    """

    def __init__(
        self,
        use_cross_encoder: bool = True,
        top_k: int = 8,
        min_score: float = -5.0,
        model_name: str = CROSS_ENCODER_MODEL,
    ) -> None:
        self._use_ce = use_cross_encoder
        self._top_k = top_k
        self._min_score = min_score
        self._model = model_name

    def rerank(self, query: str, chunks: list[dict]) -> list[dict]:
        """Return at most *top_k* chunks sorted by cross-encoder relevance."""
        if not chunks:
            return []

        if not self._use_ce:
            return self._fallback_rank(chunks)

        try:
            return self._cross_encoder_rank(query, chunks)
        except Exception as exc:
            log.warning("ranker.cross_encoder_failed_fallback", error=str(exc))
            return self._fallback_rank(chunks)

    # ------------------------------------------------------------------

    def _cross_encoder_rank(self, query: str, chunks: list[dict]) -> list[dict]:
        ce = _get_cross_encoder(self._model)
        # Truncate each passage to 512 chars — CE tokenises internally but
        # very long passages slow batch inference significantly.
        pairs = [(query, (c.get("text") or "")[:512]) for c in chunks]
        scores = ce.predict(pairs).tolist()

        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)

        result = []
        for score, chunk in ranked[: self._top_k]:
            if score < self._min_score:
                continue
            chunk["_ce_score"] = round(float(score), 4)
            chunk["_rank_method"] = "cross_encoder"
            result.append(chunk)

        # Safety net: if every chunk scored below min_score (common for
        # financial table chunks scored by an English-QA cross-encoder),
        # return the top-3 anyway so downstream nodes always have context.
        if not result and ranked:
            for score, chunk in ranked[: min(3, self._top_k)]:
                chunk["_ce_score"] = round(float(score), 4)
                chunk["_rank_method"] = "cross_encoder"
                result.append(chunk)
            log.debug("ranker.cross_encoder_fallback",
                      reason="all_below_min_score",
                      min_score=self._min_score,
                      top_score=round(ranked[0][0], 4))

        log.debug(
            "ranker.cross_encoder_done",
            input=len(chunks),
            output=len(result),
            top_score=round(ranked[0][0], 4) if ranked else None,
            bottom_score=round(ranked[-1][0], 4) if ranked else None,
        )
        return result

    def _fallback_rank(self, chunks: list[dict]) -> list[dict]:
        """Rank by existing retrieval scores when CE is disabled."""

        def score(c: dict) -> float:
            return c.get("_ce_score") or c.get("similarity") or c.get("rrf_score") or c.get("_mmr_score") or 0.0

        ranked = sorted(chunks, key=score, reverse=True)[: self._top_k]
        for c in ranked:
            c.setdefault("_rank_method", "fallback")
        return ranked
