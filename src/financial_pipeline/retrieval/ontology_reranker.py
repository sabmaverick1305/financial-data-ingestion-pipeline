"""Ontology-aware reranking — post-processes ContextRanker's cross-encoder
output with a lexical ontology-match bonus, then re-sorts.

    Final score = cross_encoder_score
                + ontology_metric_match
                + ontology_scheme_type_match
                + canonical_term_match
                + related_concept_match

Reuses the same CanonicalQuery/ExpandedQuery machinery as retrieval query
expansion (ontology_resolver.py / ontology_expansion.py): the bonus rewards
chunks whose TEXT contains the resolved metric/scheme_type vocabulary,
canonical terms, and related-concept terms for the query.

Why this is worth a second, independent scoring pass rather than folding
into query expansion alone: ContextRanker's cross-encoder
(ms-marco-MiniLM-L-12-v2) is a general MS MARCO QA model with no financial
vocabulary — its own docstring notes financial table chunks routinely score
below its relevance threshold. Query expansion helps retrieval *find* the
right chunks; this bonus helps reranking *not discard* them once found,
by directly rewarding domain-term presence the CE model has no basis to
recognize as relevant.

Additive, gated by settings.enable_ontology_reranking. Off, or nothing
resolves for the query, -> chunk order and scores pass through unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from financial_pipeline.semantic.semantic_engine import SemanticEngine, get_engine
from financial_pipeline.services.ontology_resolver import CanonicalQuery, resolve

from .ontology_expansion import ExpandedQuery, expand

# Per-component weights. Kept modest relative to the cross-encoder's
# documented -10..+10 range (ranker.py) so the bonus nudges/breaks ties and
# can rescue domain-relevant chunks the CE underrates, without letting a
# handful of keyword hits override a chunk the CE scores as clearly on-topic.
_METRIC_WEIGHT           = 1.0
_SCHEME_TYPE_WEIGHT      = 0.75
_CANONICAL_TERM_WEIGHT   = 1.0
_RELATED_CONCEPT_WEIGHT  = 0.5
_MAX_HITS_PER_COMPONENT  = 3  # caps one chunk repeating a term from dominating


@dataclass
class OntologyScoreBreakdown:
    metric_match:          float = 0.0
    scheme_type_match:     float = 0.0
    canonical_term_match:  float = 0.0
    related_concept_match: float = 0.0

    @property
    def total(self) -> float:
        return (self.metric_match + self.scheme_type_match
                + self.canonical_term_match + self.related_concept_match)


def _word_match(term: str, text: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", text) is not None


def _count_hits(terms: list[str], text: str) -> int:
    return min(sum(1 for t in terms if t and _word_match(t.lower(), text)), _MAX_HITS_PER_COMPONENT)


def score_chunk(text: str, canonical: CanonicalQuery, expanded: ExpandedQuery) -> OntologyScoreBreakdown:
    text_l = (text or "").lower()
    eng: SemanticEngine = get_engine()

    metric_hits = sum(
        _count_hits(
            [s.lower() for s in eng.metric_synonyms(mid)] + [mid.replace("_", " ")], text_l
        )
        for mid in canonical.metrics
    )
    scheme_hits = sum(
        _count_hits(
            [s.lower() for s in eng.scheme_type_synonyms(sid)] + [sid.replace("_", " ")], text_l
        )
        for sid in canonical.scheme_type_ids
    )
    canonical_hits = _count_hits(expanded.canonical_terms, text_l)
    related_hits   = _count_hits(expanded.related_terms, text_l)

    return OntologyScoreBreakdown(
        metric_match=metric_hits * _METRIC_WEIGHT,
        scheme_type_match=scheme_hits * _SCHEME_TYPE_WEIGHT,
        canonical_term_match=canonical_hits * _CANONICAL_TERM_WEIGHT,
        related_concept_match=related_hits * _RELATED_CONCEPT_WEIGHT,
    )


class OntologyAwareReranker:
    """Adds an ontology-match bonus to each chunk's cross-encoder score and
    re-sorts. Call after ContextRanker.rerank() — operates on chunks that
    already carry `_ce_score`.
    """

    def __init__(self, enabled: bool | None = None):
        self._enabled_override = enabled

    def _is_enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        from financial_pipeline.config import settings
        return settings.enable_ontology_reranking

    def rerank(self, query: str, intent, chunks: list[dict], final_k: int | None = None) -> list[dict]:
        """chunks is expected to already be CE-scored (carries `_ce_score`)
        and NOT yet truncated to the graph's final output size — see
        nodes.py's _RERANK_CANDIDATE_POOL / _RERANK_FINAL_K. Truncation to
        final_k always happens here, after the ontology bonus, so disabled
        and no-op paths return exactly the same slice a caller who never
        widened the candidate pool would have gotten.
        """
        if not chunks:
            return chunks
        if not self._is_enabled():
            return chunks[:final_k] if final_k is not None else chunks

        canonical = getattr(intent, "canonical", None) or resolve(query)
        expanded = expand(query, canonical)
        if not (canonical.metrics or canonical.scheme_type_ids or expanded.is_expanded):
            return chunks[:final_k] if final_k is not None else chunks

        for chunk in chunks:
            breakdown = score_chunk(chunk.get("text", ""), canonical, expanded)
            ce_score = chunk.get("_ce_score") or 0.0
            chunk["_ontology_bonus"] = round(breakdown.total, 4)
            chunk["_ontology_breakdown"] = {
                "metric_match": breakdown.metric_match,
                "scheme_type_match": breakdown.scheme_type_match,
                "canonical_term_match": breakdown.canonical_term_match,
                "related_concept_match": breakdown.related_concept_match,
            }
            chunk["_final_score"] = round(ce_score + breakdown.total, 4)

        chunks.sort(key=lambda c: c.get("_final_score", c.get("_ce_score") or 0.0), reverse=True)
        return chunks[:final_k] if final_k is not None else chunks
