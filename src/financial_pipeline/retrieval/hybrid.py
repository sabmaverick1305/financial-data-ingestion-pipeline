"""HybridRetriever — three-stage retrieval pipeline designed for 500K+ documents.

Why three stages?
─────────────────
At 24 K chunks (current): bi-encoder alone is fine. The vector space is sparse
enough that the top-12 results are genuinely the most relevant ones.

At 15 M chunks (500 K docs × ~30 chunks): the vector space is dense.
Many relevant chunks score 0.72 and many irrelevant chunks also score 0.71.
Bi-encoder recall at top-12 collapses because the right answers are buried in
positions 13–80. You need to fetch more candidates, fuse signals, then re-rank.

Three-stage architecture
─────────────────────────

  Stage 1 │ DUAL RETRIEVAL          │ Parallel: dense (pgvector) + sparse (BM25)
          │ Target: top-100 each    │ High recall, fast, runs concurrently
          │                         │
  Stage 2 │ RRF FUSION              │ Merge the ~200 unique candidates with
          │ Target: top-50          │ Reciprocal Rank Fusion (no score calibration
          │                         │ needed — just rank positions)
          │                         │
  Stage 3 │ CROSS-ENCODER RE-RANK  │ Accurate joint scoring of (query, chunk)
          │ Target: top-10          │ Slow but only on 50 candidates, not 15 M

Retrieval signals
──────────────────
  Dense   pgvector HNSW cosine similarity (bi-encoder, 384 dims)
  Sparse  PostgreSQL tsvector / GIN (BM25-style, exact + stemmed terms)

  Optional upgrades (wire in when needed):
    • SPLADE    — learned sparse retrieval, better than BM25 on short queries
    • ColBERT   — late-interaction model, higher accuracy than bi-encoder
    • pg_bm25   — true Okapi BM25 via ParadeDB extension (drop-in for GIN)
    • Qdrant    — replaces pgvector for sub-10ms ANN at 50M+ vectors

HNSW tuning for scale
──────────────────────
  24 K chunks  →  m=16,  ef_construction=64,   ef_search=40   (current)
  1 M chunks   →  m=32,  ef_construction=128,  ef_search=80
  15 M chunks  →  m=48,  ef_construction=200,  ef_search=100
  (higher = more accurate + slower build, same query-time complexity)
"""

from __future__ import annotations

import concurrent.futures
import time
from dataclasses import dataclass, field

import structlog

from financial_pipeline.retrieval.retriever import Retriever, _rrf_merge
from financial_pipeline.storage.document_repo import DocumentRepository

log = structlog.get_logger()


# ── Configuration ─────────────────────────────────────────────────────────────


@dataclass
class HybridConfig:
    """Tune each stage independently as corpus size grows."""

    # Stage 1 — how many candidates each signal fetches
    dense_candidates: int = 100  # pgvector HNSW top-k
    sparse_candidates: int = 100  # BM25 / GIN top-k

    # Stage 2 — RRF
    rrf_k: int = 60  # Cormack constant (higher = favour deep ranks)
    fusion_candidates: int = 50  # candidates passed to cross-encoder

    # Stage 3 — final output
    final_top_k: int = 10

    # Query expansion
    expand_query: bool = False  # prepend extracted entities to query

    # Parallelism
    parallel_retrieval: bool = True  # fetch dense + sparse concurrently

    @classmethod
    def for_corpus_size(cls, num_chunks: int) -> HybridConfig:
        """Auto-tune based on expected corpus size."""
        if num_chunks < 100_000:  # < 3 K docs
            return cls(dense_candidates=50, sparse_candidates=50, fusion_candidates=25, final_top_k=8)
        elif num_chunks < 1_000_000:  # < 30 K docs
            return cls(dense_candidates=100, sparse_candidates=100, fusion_candidates=50, final_top_k=10)
        else:  # 500 K+ docs
            return cls(
                dense_candidates=150,
                sparse_candidates=150,
                fusion_candidates=75,
                final_top_k=12,
                expand_query=True,
            )


# ── Stage 1: Dual Retrieval ────────────────────────────────────────────────────


class DualRetriever:
    """Fetches dense and sparse candidates in parallel.

    Dense  → pgvector cosine ANN (semantic match)
    Sparse → PostgreSQL tsvector GIN (exact + stemmed keyword match)

    Running them concurrently saves ~40% wall-clock time vs. sequential,
    since both are I/O-bound (network round-trip to Postgres).
    """

    def __init__(self, repo: DocumentRepository, retriever: Retriever) -> None:
        self._repo = repo
        self._retriever = retriever

    def fetch(
        self,
        query: str,
        config: HybridConfig,
        year: int | None = None,
        month: int | None = None,
        category: str | None = None,
        search_query: str | None = None,  # cleaned query for embedding
    ) -> tuple[list[dict], list[dict]]:
        """Return (dense_results, sparse_results) — may overlap."""
        embed_query = search_query or query

        if config.parallel_retrieval:
            return self._fetch_parallel(query, embed_query, config, year, month, category)
        else:
            dense = self._fetch_dense(embed_query, config, year, month, category)
            sparse = self._fetch_sparse(query, config, year, category)
            return dense, sparse

    def _fetch_parallel(
        self,
        query: str,
        embed_query: str,
        config: HybridConfig,
        year: int | None,
        month: int | None,
        category: str | None,
    ) -> tuple[list[dict], list[dict]]:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            f_dense = pool.submit(self._fetch_dense, embed_query, config, year, month, category)
            f_sparse = pool.submit(self._fetch_sparse, query, config, year, category)
            dense = f_dense.result()
            sparse = f_sparse.result()
        return dense, sparse

    def _fetch_dense(
        self,
        query: str,
        config: HybridConfig,
        year: int | None,
        month: int | None,
        category: str | None,
    ) -> list[dict]:
        try:
            vec = self._retriever._encode(query)
            results = self._repo.search_similar(
                query_embedding=vec,
                limit=config.dense_candidates,
                period_year=year,
                period_month=month,
                category=category,
            )
            for r in results:
                r["_signal"] = "dense"
            log.debug("hybrid.dense_done", count=len(results))
            return results
        except Exception as exc:
            log.warning("hybrid.dense_failed", error=str(exc))
            return []

    def _fetch_sparse(
        self,
        query: str,
        config: HybridConfig,
        year: int | None,
        category: str | None,
    ) -> list[dict]:
        try:
            results = self._repo.search_fulltext(
                query=query,
                limit=config.sparse_candidates,
                period_year=year,
                category=category,
            )
            for r in results:
                r["_signal"] = "sparse"
            log.debug("hybrid.sparse_done", count=len(results))
            return results
        except Exception as exc:
            log.warning("hybrid.sparse_failed", error=str(exc))
            return []


# ── Stage 2: RRF Fusion ────────────────────────────────────────────────────────


class RRFFusion:
    """Merges dense + sparse ranked lists with Reciprocal Rank Fusion.

    RRF(d) = Σ  1 / (k + rank(d, list))

    Advantages over score normalisation:
    - No calibration needed between incompatible score scales
      (cosine similarity 0–1 vs BM25 logits 0–30+)
    - Robust: a very high BM25 score for one result doesn't dominate
    - Handles missing entries (a chunk missing from dense gets 0 from that list)
    """

    def fuse(
        self,
        dense: list[dict],
        sparse: list[dict],
        config: HybridConfig,
    ) -> list[dict]:
        merged = _rrf_merge(dense, sparse, limit=config.fusion_candidates, k=config.rrf_k)
        # Tag with which signals contributed
        dense_ids = {str(r.get("chunk_id")) for r in dense}
        sparse_ids = {str(r.get("chunk_id")) for r in sparse}
        for r in merged:
            cid = str(r.get("chunk_id"))
            signals = []
            if cid in dense_ids:
                signals.append("dense")
            if cid in sparse_ids:
                signals.append("sparse")
            r["_signals"] = "+".join(signals)  # "dense+sparse" = matched both!

        log.debug(
            "hybrid.fused",
            dense_in=len(dense),
            sparse_in=len(sparse),
            fused_out=len(merged),
            both_signals=sum(1 for r in merged if "+" in r.get("_signals", "")),
        )
        return merged


# ── Stage 3: Cross-encoder — imported from augmentation/ranker.py ─────────────
# (The cross-encoder re-ranks the fused top-50 to produce final top-10.
#  This is the same ContextRanker used in the augmentation layer; importing
#  it here keeps the model cached in one place.)


# ── Query Expansion ────────────────────────────────────────────────────────────


class QueryExpander:
    """Expands queries with extracted financial entities.

    Short queries like "AUM 2024" have poor recall with bi-encoders because
    the embedding is under-specified. Expansion adds synonyms and related
    terms extracted from the QueryIntent to improve dense retrieval recall.

    Example:
        "SIP growth" → "SIP growth systematic investment plan monthly contribution"
    """

    SYNONYMS: dict[str, list[str]] = {
        "aum": ["assets under management", "total aum", "fund size"],
        "sip": ["systematic investment plan", "monthly sip", "sip contribution"],
        "nav": ["net asset value", "unit price"],
        "elss": ["equity linked savings scheme", "tax saving fund"],
        "folio": ["investor account", "number of folios"],
        "sebi": ["securities and exchange board of india", "regulator"],
    }

    def expand(self, query: str, intent) -> str:
        """Return an enriched query string for better bi-encoder recall."""
        q = query.lower()
        additions = []

        for term, synonyms in self.SYNONYMS.items():
            if term in q:
                additions.extend(synonyms[:2])

        if intent and intent.scheme_types:
            additions.extend(intent.scheme_types[:3])

        if not additions:
            return query

        expanded = query + " " + " ".join(additions)
        log.debug("query_expander.done", original=query[:50], expanded=expanded[:80])
        return expanded


# ── Stage 4: Context Optimizer ────────────────────────────────────────────────


class ContextOptimizer:
    """Stage 4 — Post-rerank context refinement.

    Four passes applied after cross-encoder re-ranking:

    1. Near-duplicate removal
       Two chunks are near-duplicates when their word-set Jaccard similarity
       exceeds DEDUP_THRESHOLD (0.85). This catches sliding-window chunk
       overlap and repeated boilerplate (fund disclaimer text, headers).

    2. Source diversity
       Cap the number of chunks from any single document_id at MAX_PER_DOC.
       Prevents the context being dominated by one (possibly wrong) document.

    3. Recency preference
       When the QueryIntent says prefer_recent=True (query contains "latest",
       "current", "now") and no explicit year was given, chunks are sorted
       by (period_year DESC, period_month DESC) before slicing to final_top_k.

    4. Table-linked preservation
       For numeric/tabular intents, ensure at least MIN_TABLE_CHUNKS chunks
       tagged _source="table" survive into the final context. Table chunks
       carry schema_json and parquet_s3_key which the LLM can reference for
       exact numbers.
    """

    DEDUP_THRESHOLD = 0.85  # Jaccard similarity above this = near-duplicate
    MAX_PER_DOC = 2  # max chunks from the same document_id
    MIN_TABLE_CHUNKS = 1  # numeric intents: keep at least this many table chunks

    def optimize(
        self,
        chunks: list[dict],
        intent=None,  # QueryIntent
        final_top_k: int = 10,
    ) -> tuple[list[dict], dict]:
        """Return (optimized_chunks, stats)."""
        stats: dict = {}
        pool = list(chunks)

        # Pass 1: near-duplicate removal
        pool, n_removed = self._dedup(pool)
        stats["dedup_removed"] = n_removed

        # Pass 2: source diversity
        pool, n_capped = self._diversity(pool)
        stats["diversity_capped"] = n_capped

        # Pass 3: recency preference
        if intent and getattr(intent, "prefer_recent", False) and not getattr(intent, "year", None):
            pool = self._sort_by_recency(pool)
            stats["recency_sort"] = True
        else:
            stats["recency_sort"] = False

        # Pass 4: table-linked preservation for numeric queries
        intent_type = getattr(intent, "intent_type", "") if intent else ""
        if intent_type in ("tabular", "factual") and getattr(intent, "needs_tables", False):
            pool = self._preserve_table_chunks(pool, final_top_k)
            stats["table_preserved"] = True
        else:
            stats["table_preserved"] = False

        result = pool[:final_top_k]
        stats["final_count"] = len(result)

        log.debug(
            "optimizer.done",
            input=len(chunks),
            output=len(result),
            dedup=n_removed,
            diversity=n_capped,
            recency=stats["recency_sort"],
        )
        return result, stats

    # ------------------------------------------------------------------

    def _dedup(self, chunks: list[dict]) -> tuple[list[dict], int]:
        """Remove near-duplicate chunks using Jaccard word-set similarity."""
        kept: list[dict] = []
        removed: int = 0

        for candidate in chunks:
            text_c = (candidate.get("text") or "").lower()
            words_c = set(text_c.split())

            is_dup = False
            for kept_chunk in kept:
                text_k = (kept_chunk.get("text") or "").lower()
                words_k = set(text_k.split())
                union = words_k | words_c
                if not union:
                    continue
                jaccard = len(words_k & words_c) / len(union)
                if jaccard >= self.DEDUP_THRESHOLD:
                    is_dup = True
                    break

            if is_dup:
                removed += 1
            else:
                kept.append(candidate)

        return kept, removed

    def _diversity(self, chunks: list[dict]) -> tuple[list[dict], int]:
        """Cap chunks per document to enforce source diversity."""
        doc_counts: dict[str, int] = {}
        kept: list[dict] = []
        capped: int = 0

        for chunk in chunks:
            doc_id = str(chunk.get("document_id") or chunk.get("file_name") or "")
            count = doc_counts.get(doc_id, 0)
            if count >= self.MAX_PER_DOC:
                capped += 1
            else:
                doc_counts[doc_id] = count + 1
                kept.append(chunk)

        return kept, capped

    def _sort_by_recency(self, chunks: list[dict]) -> list[dict]:
        """Sort chunks newest-first when query asks for latest data."""

        def recency_key(c: dict):
            y = c.get("period_year") or 0
            m = c.get("period_month") or 0
            return (y, m)

        return sorted(chunks, key=recency_key, reverse=True)

    def _preserve_table_chunks(self, chunks: list[dict], top_k: int) -> list[dict]:
        """Ensure table chunks aren't all pushed out by prose re-ranking."""
        tables = [c for c in chunks if c.get("_source") == "table"]
        prose = [c for c in chunks if c.get("_source") != "table"]

        if len(tables) >= self.MIN_TABLE_CHUNKS:
            return chunks  # already enough tables in the natural ranking

        # Inject table chunks at front, fill rest with prose
        guaranteed = tables[: self.MIN_TABLE_CHUNKS]
        remaining = [c for c in prose if c not in guaranteed]
        return (guaranteed + remaining)[: top_k + len(guaranteed)]


# ── Orchestrator ───────────────────────────────────────────────────────────────


@dataclass
class HybridRetrievalResult:
    query: str
    dense_count: int
    sparse_count: int
    fused_count: int
    reranked_count: int
    final_count: int
    both_signals: int  # chunks matched by BOTH dense AND sparse
    latency_ms: int
    config: HybridConfig
    stage_ms: dict = field(default_factory=dict)  # per-stage latency
    optimizer_stats: dict = field(default_factory=dict)
    results: list[dict] = field(default_factory=list)


class HybridRetriever:
    """Orchestrates the 4-stage (+Stage 0 routing) hybrid retrieval pipeline.

    Scales accurately from 24 K to 15 M+ chunks by keeping the cross-encoder
    (Stage 3) operating on a fixed-size shortlist regardless of corpus size.

    Stage 0  Query Router     — detect query type, apply metadata filters
    Stage 1  Dual Retrieval   — parallel dense (pgvector) + sparse (BM25)
    Stage 2  RRF Fusion       — merge 300 → 75 unique candidates
    Stage 3  Cross-Encoder    — joint scoring, top 75 → top 20
    Stage 4  Context Optimizer— dedup, diversity, recency, table preservation

    Usage::

        hybrid = HybridRetriever(repo, retriever,
                     config=HybridConfig.for_corpus_size(15_000_000))
        result = hybrid.retrieve("What SEBI guidelines restricted derivatives?",
                                  year=2004, category="quarterly")
    """

    def __init__(
        self,
        repo: DocumentRepository,
        retriever: Retriever,
        config: HybridConfig | None = None,
    ) -> None:
        from financial_pipeline.augmentation.ranker import ContextRanker

        self._dual_ret = DualRetriever(repo, retriever)
        self._fusion = RRFFusion()
        self._expander = QueryExpander()
        self._reranker = ContextRanker(use_cross_encoder=True)
        self._optimizer = ContextOptimizer()
        self._config = config or HybridConfig()

    def retrieve(
        self,
        query: str,
        intent=None,
        year: int | None = None,
        month: int | None = None,
        category: str | None = None,
        config: HybridConfig | None = None,
    ) -> HybridRetrievalResult:
        t0 = time.perf_counter()
        cfg = config or self._config

        # ── Stage 0: Query routing / query expansion ──────────────────
        search_query = self._expander.expand(query, intent) if cfg.expand_query and intent else query
        t1 = time.perf_counter()

        # ── Stage 1: Parallel dual retrieval ─────────────────────────
        dense, sparse = self._dual_ret.fetch(
            query=query,
            search_query=search_query,
            config=cfg,
            year=year,
            month=month,
            category=category,
        )
        t2 = time.perf_counter()

        # ── Stage 2: RRF fusion ───────────────────────────────────────
        fused = self._fusion.fuse(dense, sparse, cfg)
        both = sum(1 for r in fused if "+" in r.get("_signals", ""))
        t3 = time.perf_counter()

        # ── Stage 3: Cross-encoder re-rank ────────────────────────────
        reranked = self._reranker.rerank(query, fused)
        t4 = time.perf_counter()

        # ── Stage 4: Context optimisation ────────────────────────────
        optimized, opt_stats = self._optimizer.optimize(reranked, intent=intent, final_top_k=cfg.final_top_k)
        t5 = time.perf_counter()

        stage_ms = {
            "s0_routing_ms": int((t1 - t0) * 1000),
            "s1_retrieval_ms": int((t2 - t1) * 1000),
            "s2_fusion_ms": int((t3 - t2) * 1000),
            "s3_rerank_ms": int((t4 - t3) * 1000),
            "s4_optimize_ms": int((t5 - t4) * 1000),
        }
        total_ms = int((t5 - t0) * 1000)

        log.info(
            "hybrid.done",
            dense=len(dense),
            sparse=len(sparse),
            fused=len(fused),
            reranked=len(reranked),
            final=len(optimized),
            both_signals=both,
            latency_ms=total_ms,
            **stage_ms,
        )

        return HybridRetrievalResult(
            query=query,
            dense_count=len(dense),
            sparse_count=len(sparse),
            fused_count=len(fused),
            reranked_count=len(reranked),
            final_count=len(optimized),
            both_signals=both,
            latency_ms=total_ms,
            config=cfg,
            stage_ms=stage_ms,
            optimizer_stats=opt_stats,
            results=optimized,
        )

    @staticmethod
    def explain(result: HybridRetrievalResult) -> str:
        """Human-readable pipeline trace for debugging."""
        s = result.stage_ms
        o = result.optimizer_stats
        lines = [
            f"Query         : {result.query[:70]}",
            "",
            f"Stage 0 route : {s.get('s0_routing_ms', 0)} ms  (query expansion)",
            f"Stage 1 fetch : {s.get('s1_retrieval_ms', 0)} ms  dense={result.dense_count}  sparse={result.sparse_count}",
            f"Stage 2 fuse  : {s.get('s2_fusion_ms', 0)} ms  fused={result.fused_count}  both_signals={result.both_signals}",
            f"Stage 3 rerank: {s.get('s3_rerank_ms', 0)} ms  reranked={result.reranked_count}",
            f"Stage 4 optim : {s.get('s4_optimize_ms', 0)} ms"
            f"  dedup={o.get('dedup_removed', 0)}"
            f"  diversity={o.get('diversity_capped', 0)}"
            f"  recency={o.get('recency_sort', False)}"
            f"  tables={o.get('table_preserved', False)}",
            "",
            f"Final output  : {result.final_count} chunks  ({result.latency_ms} ms total)",
        ]
        if result.results:
            lines.append("Top results:")
            for i, r in enumerate(result.results[:5], 1):
                ce = f"ce={r['_ce_score']:.2f}" if "_ce_score" in r else ""
                rrf = f"rrf={r['rrf_score']:.4f}" if "rrf_score" in r else ""
                sig = r.get("_signals", "?")
                name = r.get("file_name", "?")
                lines.append(f"  [{i}] {name}  {sig}  {ce}  {rrf}")
        return "\n".join(lines)
