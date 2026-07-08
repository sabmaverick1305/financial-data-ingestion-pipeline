"""Ontology-driven query expansion for retrieval — Stage 0.5 of the RAG path.

    User Query
      v
    Ontology Resolver
      v
    Query Expansion
      v
    Hybrid Retrieval
      |-- original query
      `-- ontology-expanded query (canonical metric terms + synonyms + related concepts)
      v
    RRF + rerank

Expansion terms come from the semantic/ stack, via SemanticEngine:
  - thesaurus.yaml's full metric/scheme_type synonym lists -> synonym terms
  - financial_relationships.yaml (formulas/influenced_by/related_to/indicates)
    -> related-concept terms, e.g. "net inflow" pulls in "funds mobilized" +
    "redemption" via net_inflow's formula, which is exactly the vocabulary
    AMFI's definitional documents use to explain net inflow.

This targets the RAG-path retrieval queries specifically (definitional/lookup
intents — "What is ELSS?", "What does net inflow mean?") where the eval
suite's keyword_coverage was weak: the raw query's wording ("ELSS") often
isn't the wording the source document uses ("Equity Linked Savings Scheme"),
so retrieval under-recalls even when the right chunk exists.

OntologyExpandedRetriever wraps dense/sparse search with a second query arm
built from ExpandedQuery, RRF-merged with the original-query results — never
a replacement. Gated by settings.enable_ontology_retrieval; when off, or
when nothing resolves to expand, behavior is identical to unexpanded search.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from financial_pipeline.semantic.semantic_engine import SemanticEngine, get_engine

from .ontology_resolver import CanonicalQuery, resolve


@dataclass
class ExpandedQuery:
    original: str
    canonical_terms: list[str] = field(default_factory=list)
    synonym_terms:   list[str] = field(default_factory=list)
    related_terms:   list[str] = field(default_factory=list)

    @property
    def is_expanded(self) -> bool:
        return bool(self.canonical_terms or self.synonym_terms or self.related_terms)

    @property
    def expansion_text(self) -> str:
        """De-duplicated, order-preserving join of every expansion term."""
        seen: dict[str, None] = {}
        for term in (*self.canonical_terms, *self.synonym_terms, *self.related_terms):
            seen.setdefault(term, None)
        return " ".join(seen)

    def as_query(self) -> str:
        return f"{self.original} {self.expansion_text}".strip() if self.is_expanded else self.original


def expand(query: str, canonical: CanonicalQuery | None = None, max_terms: int = 6) -> ExpandedQuery:
    """Build an ExpandedQuery from a raw query (+ optional pre-resolved CanonicalQuery,
    e.g. intent.canonical, to avoid resolving twice)."""
    canonical = canonical or resolve(query)
    eng: SemanticEngine = get_engine()

    canonical_terms: list[str] = []
    synonym_terms: list[str] = []
    related: list[str] = []

    for metric_id in canonical.metrics:
        name = eng.concept_canonical_name(metric_id)
        if name and name not in canonical_terms:
            canonical_terms.append(name)
        for term in eng.metric_synonyms(metric_id):
            if term not in synonym_terms:
                synonym_terms.append(term)
        for term in eng.related_concept_terms(metric_id):
            if term not in related:
                related.append(term)

    for entity_id in canonical.scheme_type_ids:
        for term in eng.scheme_type_synonyms(entity_id):
            if term not in synonym_terms:
                synonym_terms.append(term)

    return ExpandedQuery(
        original=query,
        canonical_terms=canonical_terms[:max_terms],
        synonym_terms=synonym_terms[:max_terms],
        related_terms=related[:max_terms],
    )


class OntologyExpandedRetriever:
    """Wraps a repo's dense/sparse search with an ontology-expanded second
    query arm, RRF-merged with the original-query results.

    `enabled=None` (default) reads settings.enable_ontology_retrieval at call
    time, so flipping the env var takes effect without re-instantiation.
    """

    def __init__(self, repo, encode_fn, rrf_merge_fn, enabled: bool | None = None):
        self._repo = repo
        self._encode = encode_fn
        self._rrf_merge = rrf_merge_fn
        self._enabled_override = enabled

    def _is_enabled(self) -> bool:
        if self._enabled_override is not None:
            return self._enabled_override
        from financial_pipeline.config import settings
        return settings.enable_ontology_retrieval

    def expand_query(self, query: str, intent) -> ExpandedQuery:
        canonical = getattr(intent, "canonical", None)
        return expand(query, canonical)

    def dense(self, query: str, intent, *, limit: int = 20, **search_kwargs) -> list[dict]:
        base = self._repo.search_similar(query_embedding=self._encode(query), limit=limit, **search_kwargs)
        for r in base:
            r["_source"], r["search_mode"] = "chunk", "dense"
        if not self._is_enabled():
            return base

        expanded = self.expand_query(query, intent)
        if not expanded.is_expanded:
            return base

        exp_results = self._repo.search_similar(
            query_embedding=self._encode(expanded.as_query()), limit=limit, **search_kwargs
        )
        for r in exp_results:
            r["_source"], r["search_mode"] = "chunk", "dense_ontology"
        return self._rrf_merge(base, exp_results, limit=limit)

    def sparse(self, query: str, intent, *, limit: int = 20, **search_kwargs) -> list[dict]:
        base = self._repo.search_fulltext(query=query, limit=limit, **search_kwargs)
        for r in base:
            r["_source"], r["search_mode"] = "chunk", "sparse"
        if not self._is_enabled():
            return base

        expanded = self.expand_query(query, intent)
        if not expanded.is_expanded:
            return base

        exp_results = self._repo.search_fulltext(query=expanded.as_query(), limit=limit, **search_kwargs)
        for r in exp_results:
            r["_source"], r["search_mode"] = "chunk", "sparse_ontology"
        return self._rrf_merge(base, exp_results, limit=limit)
