"""Node functions for the RAG LangGraph.

Every node follows the LangGraph contract:
    (state: RAGState) -> dict   ← partial state update only

Dependencies (repo, retriever, ranker, generator) are injected via
NodeFactory, which is instantiated once at app startup and passed to
graph.py when the compiled graph is built.
"""
from __future__ import annotations

import re
import time

import structlog

from financial_pipeline.augmentation.citations import CitationFormatter
from financial_pipeline.augmentation.generator import AnswerGenerator
from financial_pipeline.augmentation.guardrails import (
    PostGenerationGuardrails,
    PreGenerationGuardrails,
)
from financial_pipeline.augmentation.prompts import PromptBuilder
from financial_pipeline.augmentation.ranker import ContextRanker
from financial_pipeline.retrieval.hybrid import ContextOptimizer
from financial_pipeline.retrieval.ontology_expansion import OntologyExpandedRetriever
from financial_pipeline.retrieval.ontology_reranker import OntologyAwareReranker
from financial_pipeline.retrieval.pipeline import SearchRouter
from financial_pipeline.retrieval.query_understanding import QueryAnalyzer
from financial_pipeline.retrieval.retriever import _rrf_merge
from financial_pipeline.graph.state import RAGState

log = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

_RETRIEVAL_BRANCH_ALL = ["dense", "sparse", "table", "metadata"]
_STOPWORDS = frozenset({
    "what", "is", "the", "are", "how", "many", "give", "me",
    "show", "find", "list", "total", "please", "can", "you",
    "tell", "about", "for", "in", "of", "a", "an",
})
_MONTH_NAMES = [
    "", "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
# ContextRanker's CE model scores every input chunk regardless of top_k (only
# truncates the *output*), so widening this to match rrf_fusion's merge limit
# costs nothing extra in CE inference — it just stops throwing away scored
# candidates before OntologyAwareReranker gets a chance to re-weigh them.
# The actual output size returned to the graph is still capped by
# _RERANK_FINAL_K, applied after the ontology bonus.
_RERANK_CANDIDATE_POOL = 20
_RERANK_FINAL_K = 8


def _expand_retrieval_query(query: str, intent) -> str:
    """Append temporal and domain context to improve recall.

    Adds the month name when intent.month is set but the month name is absent
    from the query (helps BM25 find documents that contain e.g. "August 2022").
    """
    if not intent:
        return query
    additions = []
    m = getattr(intent, "month", None)
    if m and 1 <= m <= 12:
        month_name = _MONTH_NAMES[m]
        if month_name.lower() not in query.lower():
            additions.append(month_name)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


# ── Dependency container ──────────────────────────────────────────────────────

class NodeFactory:
    """Holds all shared dependencies and exposes nodes as bound methods.

    Instantiate once at app startup:

        factory = NodeFactory(repo=repo, retriever=retriever)
        graph   = build_graph(factory)
    """

    def __init__(self, repo, retriever) -> None:
        self._repo      = repo
        self._retriever = retriever
        self._generator = AnswerGenerator()
        embed_fn = retriever._encode if getattr(retriever, "semantic_available", False) else None
        # QueryAnalyzer's Stage 3 (_LLMExtractor) is a small structured-JSON
        # classification call — routed to the cheaper OpenAI tier, kept
        # separate from self._generator (which stays on the default provider).
        self._analyzer  = QueryAnalyzer(
            embed_fn=embed_fn,
            generator=AnswerGenerator(provider="openai"),
        )
        self._ontology_retriever = OntologyExpandedRetriever(
            repo=repo, encode_fn=self._retriever._encode, rrf_merge_fn=_rrf_merge,
        )
        self._ontology_ranker = OntologyAwareReranker()
        self._router    = SearchRouter()
        self._ranker    = ContextRanker(use_cross_encoder=True, top_k=_RERANK_CANDIDATE_POOL)
        self._optimizer = ContextOptimizer()
        self._cit_fmt   = CitationFormatter()
        self._prompter  = PromptBuilder()
        self._pre_guard = PreGenerationGuardrails()
        self._post_guard= PostGenerationGuardrails()

    # ── Node 1: analyze_query ─────────────────────────────────────────────────

    def analyze_query(self, state: RAGState) -> dict:
        query  = state.get("query", "")
        intent = self._analyzer.analyze(query)

        # Early policy gate: check advice/OOS before any retrieval.
        # Passes empty citations — only the pattern checks run (no source-count check).
        # has_identified_fund: a named AMC means a per-scheme NAV lookup is
        # answerable via fund_performance_sql, so the ambiguous-NAV OOS
        # patterns shouldn't block it — see guardrails.py's
        # _AMBIGUOUS_NAV_PATTERNS comment.
        pre_check = self._pre_guard.check(
            question=query, citations=[], intent_type=intent.intent_type,
            has_identified_fund=bool(intent.amc_names),
        )

        log.info("node.analyze_query",
                 intent=intent.intent_type,
                 secondary=intent.secondary_intent,
                 metric=intent.metric,
                 aggregation=intent.aggregation,
                 needs_analytical=intent.needs_analytical,
                 year=intent.year, month=intent.month,
                 year_from=intent.year_from, year_to=intent.year_to,
                 scheme_types=intent.scheme_types,
                 amc_names=intent.amc_names,
                 stage=intent.stage_used,
                 confidence=round(intent.confidence, 2),
                 early_blocked=not pre_check.should_proceed)
        result: dict = {
            "intent":          intent,
            "retry_count":     state.get("retry_count", 0),
            "repair_count":    state.get("repair_count", 0),
            "rewritten_query": None,
        }
        if not pre_check.should_proceed:
            result["blocked"]    = True
            result["pre_result"] = pre_check
        return result

    # ── Node 2: route ─────────────────────────────────────────────────────────

    def route(self, state: RAGState) -> dict:
        intent = state["intent"]

        # On CRAG retry, use rewritten query
        if state.get("rewritten_query"):
            intent.search_query = state["rewritten_query"]

        plan           = self._router.route(intent, top_k=12)
        active_branches: list[str] = ["dense", "sparse"]

        if plan.fetch_tables or intent.intent_type in ("tabular", "factual"):
            active_branches.append("table")
        if plan.fetch_documents or intent.intent_type == "lookup":
            active_branches.append("metadata")

        log.info("node.route", branches=active_branches,
                 chunk_mode=plan.chunk_mode, intent=intent.intent_type)
        return {"active_branches": active_branches}

    # ── Nodes 3a–3d: parallel retrieval ──────────────────────────────────────

    def retrieve_dense(self, state: RAGState) -> dict:
        intent    = state["intent"]
        query     = state.get("rewritten_query") or intent.search_query or intent.raw_query
        query     = _expand_retrieval_query(query, intent)
        doc_ids   = state.get("found_document_ids") or None
        year_from = getattr(intent, "year_from", None)
        year_to   = getattr(intent, "year_to", None)
        try:
            results = self._ontology_retriever.dense(
                query, intent,
                limit=20,
                period_year=intent.year,
                period_month=intent.month,
                category=intent.category,
                document_ids=doc_ids,
                year_from=year_from,
                year_to=year_to,
            )
            log.debug("node.retrieve_dense", count=len(results))
            return {"dense_results": results}
        except Exception as exc:
            log.warning("node.retrieve_dense.failed", error=str(exc))
            return {"dense_results": []}

    def retrieve_sparse(self, state: RAGState) -> dict:
        intent    = state["intent"]
        query     = state.get("rewritten_query") or intent.search_query or intent.raw_query
        query     = _expand_retrieval_query(query, intent)
        doc_ids   = state.get("found_document_ids") or None
        year_from = getattr(intent, "year_from", None)
        year_to   = getattr(intent, "year_to", None)
        try:
            results = self._ontology_retriever.sparse(
                query, intent,
                limit=20,
                period_year=intent.year,
                period_month=intent.month,
                category=intent.category,
                document_ids=doc_ids,
                year_from=year_from,
                year_to=year_to,
            )
            log.debug("node.retrieve_sparse", count=len(results))
            return {"sparse_results": results}
        except Exception as exc:
            log.warning("node.retrieve_sparse.failed", error=str(exc))
            return {"sparse_results": []}

    def retrieve_table(self, state: RAGState) -> dict:
        intent    = state.get("intent")
        if not intent:
            return {"table_results": []}
        doc_ids   = state.get("found_document_ids") or None
        year_from = getattr(intent, "year_from", None)
        year_to   = getattr(intent, "year_to", None)
        results   = []

        # Exact-match injection for known scheme types
        for scheme in intent.scheme_types[:2]:
            try:
                from sqlalchemy import text as sa_text
                if year_from and year_to:
                    # Year-range query: use a window function to get 2 chunks
                    # per year so all years 2020-2026 are represented, not
                    # just the most recent year at the top of ORDER BY DESC.
                    params: dict = {
                        "pattern":  f"%{scheme}%",
                        "year_from": year_from,
                        "year_to":   year_to,
                    }
                    sql = """
                        SELECT chunk_id, chunk_index, text,
                               period_year, period_month, category,
                               file_name, document_type, s3_processed_key,
                               1.0 AS similarity
                        FROM (
                            SELECT dc.chunk_id, dc.chunk_index, dc.text,
                                   dc.period_year, dc.period_month, dc.category,
                                   dm.file_name, dm.document_type, dm.s3_processed_key,
                                   ROW_NUMBER() OVER (
                                       PARTITION BY dc.period_year
                                       ORDER BY dc.chunk_index ASC
                                   ) AS rn
                            FROM document_chunks dc
                            JOIN document_metadata dm ON dc.document_id = dm.document_id
                            WHERE dc.text ILIKE :pattern
                              AND dc.period_year BETWEEN :year_from AND :year_to
                        ) ranked
                        WHERE rn <= 2
                        ORDER BY period_year ASC, period_month ASC
                    """
                else:
                    # Point-in-time query: simple filters + recency ordering
                    conditions = ["dc.text ILIKE :pattern"]
                    params: dict = {"pattern": f"%{scheme}%", "limit": 3}
                    if intent.year:
                        conditions.append("dc.period_year = :year")
                        params["year"] = intent.year
                    if intent.month:
                        conditions.append("dc.period_month = :month")
                        params["month"] = intent.month
                    if intent.category:
                        conditions.append("dc.category = :cat")
                        params["cat"] = intent.category
                    if doc_ids:
                        conditions.append("CAST(dc.document_id AS text) = ANY(:doc_ids)")
                        params["doc_ids"] = doc_ids
                    sql = f"""
                        SELECT dc.chunk_id, dc.chunk_index, dc.text,
                               dc.period_year, dc.period_month, dc.category,
                               dm.file_name, dm.document_type, dm.s3_processed_key,
                               1.0 AS similarity
                        FROM document_chunks dc
                        JOIN document_metadata dm ON dc.document_id = dm.document_id
                        WHERE {' AND '.join(conditions)}
                        ORDER BY dc.period_year DESC NULLS LAST,
                                 dc.period_month DESC NULLS LAST,
                                 dc.chunk_index ASC
                        LIMIT :limit
                    """
                with self._repo._engine.connect() as conn:
                    rows = conn.execute(sa_text(sql), params).mappings().all()
                for r in rows:
                    d = dict(r)
                    d["similarity"]   = float(d.get("similarity") or 1.0)
                    d["_source"]      = "chunk"
                    d["search_mode"]  = "exact"
                    results.append(d)
            except Exception as exc:
                log.warning("node.retrieve_table.scheme_failed",
                            scheme=scheme, error=str(exc))

        log.debug("node.retrieve_table", count=len(results))
        return {"table_results": results}

    def retrieve_metadata(self, state: RAGState) -> dict:
        intent = state["intent"]
        try:
            from sqlalchemy import text as sa_text
            conditions = ["1=1"]
            params: dict = {"limit": 5}
            if intent.year:
                conditions.append("period_year = :year")
                params["year"] = intent.year
            if intent.month:
                conditions.append("period_month = :month")
                params["month"] = intent.month
            if intent.category:
                conditions.append("category = :cat")
                params["cat"] = intent.category

            sql = f"""
                SELECT document_id, file_name, file_type, category,
                       period_year, period_month, s3_processed_key
                FROM document_metadata
                WHERE {' AND '.join(conditions)}
                ORDER BY period_year DESC NULLS LAST,
                         period_month DESC NULLS LAST
                LIMIT :limit
            """
            with self._repo._engine.connect() as conn:
                rows = conn.execute(sa_text(sql), params).mappings().all()
            results = [dict(r) | {"_source": "document"} for r in rows]
            log.debug("node.retrieve_metadata", count=len(results))
            return {"metadata_results": results}
        except Exception as exc:
            log.warning("node.retrieve_metadata.failed", error=str(exc))
            return {"metadata_results": []}

    # ── Sequential lookup path: metadata first ────────────────────────────────

    def retrieve_metadata_first(self, state: RAGState) -> dict:
        """Phase 1 of the lookup sequential path.

        Runs the same metadata query as retrieve_metadata but additionally
        extracts document_ids so downstream retrieve_dense/sparse/table nodes
        can scope their search to exactly those documents.
        """
        intent = state["intent"]
        try:
            from sqlalchemy import text as sa_text
            conditions = ["1=1"]
            params: dict = {"limit": 5}
            if intent.year:
                conditions.append("period_year = :year")
                params["year"] = intent.year
            if intent.month:
                conditions.append("period_month = :month")
                params["month"] = intent.month
            if intent.category:
                conditions.append("category = :cat")
                params["cat"] = intent.category

            sql = f"""
                SELECT document_id, file_name, file_type, category,
                       period_year, period_month, s3_processed_key
                FROM document_metadata
                WHERE {' AND '.join(conditions)}
                ORDER BY period_year DESC NULLS LAST,
                         period_month DESC NULLS LAST
                LIMIT :limit
            """
            with self._repo._engine.connect() as conn:
                rows = conn.execute(sa_text(sql), params).mappings().all()

            results      = [dict(r) | {"_source": "document"} for r in rows]
            doc_ids      = [str(r["document_id"]) for r in rows]

            log.info("node.retrieve_metadata_first",
                     found=len(doc_ids),
                     doc_ids=doc_ids[:3])
            return {
                "metadata_results":    results,
                "found_document_ids":  doc_ids,
            }
        except Exception as exc:
            log.warning("node.retrieve_metadata_first.failed", error=str(exc))
            return {"metadata_results": [], "found_document_ids": []}

    # ── Node 4: rrf_fusion ────────────────────────────────────────────────────

    def rrf_fusion(self, state: RAGState) -> dict:
        dense    = state.get("dense_results", [])
        sparse   = state.get("sparse_results", [])
        table    = state.get("table_results", [])
        metadata = state.get("metadata_results", [])

        # Table + exact chunks prepended before RRF so they survive the merge
        chunk_lists = [table + dense, sparse]
        fused = _rrf_merge(chunk_lists[0], chunk_lists[1], limit=20)

        # Append metadata separately (different source type)
        fused += metadata

        log.debug("node.rrf_fusion",
                  dense=len(dense), sparse=len(sparse),
                  table=len(table), fused=len(fused))
        return {"fused_results": fused}

    # ── Node 5: rerank ────────────────────────────────────────────────────────

    def rerank(self, state: RAGState) -> dict:
        intent = state.get("intent")
        query  = state.get("rewritten_query") or (intent.search_query if intent else "") or state.get("query", "")
        chunks = [r for r in state.get("fused_results", [])
                  if r.get("_source") == "chunk"]
        docs   = [r for r in state.get("fused_results", [])
                  if r.get("_source") == "document"]

        # For year-range trend queries, the cross-encoder scores each chunk
        # against the full multi-year query and rejects historically-relevant
        # chunks that only match one specific year. The window-function SQL in
        # retrieve_table already picked 2 representative chunks per year —
        # pass them through with RRF ordering intact rather than filtering.
        if getattr(intent, "year_from", None) and getattr(intent, "year_to", None):
            # Keep ALL chunks — window-function SQL already capped at 2/year.
            # Cutting to [:8] would drop 2024/2025/2026 which arrive last in
            # year-ASC order.
            reranked = chunks + docs
            log.debug("node.rerank.passthrough",
                      reason="year_range", chunks=len(reranked))
        else:
            reranked = self._ranker.rerank(query, chunks)
            reranked = self._ontology_ranker.rerank(query, intent, reranked, final_k=_RERANK_FINAL_K)
            reranked += docs

        log.debug("node.rerank", input=len(chunks), output=len(reranked))
        return {"reranked_results": reranked}

    # ── Node 6: context_optimizer ─────────────────────────────────────────────

    def context_optimizer(self, state: RAGState) -> dict:
        reranked = state.get("reranked_results", [])
        intent   = state.get("intent")

        if not reranked:
            return {
                "optimized_results": [],
                "optimizer_stats": {"reason": "no_reranked_results"},
            }

        year_from = getattr(intent, "year_from", None)
        year_to   = getattr(intent, "year_to", None)

        # Year-range path: Jaccard dedup treats all table chunks as near-duplicates
        # because they share the same long column header (85%+ word overlap).
        # Replace with year-based dedup: keep the 2 best chunks per year so
        # every year in the range is represented in the LLM context.
        if year_from and year_to:
            from collections import defaultdict
            by_year: dict = defaultdict(list)
            for chunk in reranked:
                yr = chunk.get("period_year")
                if yr:
                    by_year[yr].append(chunk)

            optimized = []
            for yr in sorted(by_year.keys()):        # year ASC → oldest first
                optimized.extend(by_year[yr][:2])    # max 2 chunks per year

            years_found = sorted(by_year.keys())
            stats = {
                "year_based_dedup": True,
                "years_found":      years_found,
                "chunks_per_year":  {yr: len(by_year[yr]) for yr in years_found},
            }
            log.info("node.context_optimizer.year_range",
                     years=years_found, output=len(optimized))
            return {"optimized_results": optimized, "optimizer_stats": stats}

        # Point-in-time path: standard ContextOptimizer passes
        try:
            optimized, stats = self._optimizer.optimize(
                chunks=reranked, intent=intent, final_top_k=8,
            )
        except Exception as exc:
            # Safeguard 2: optimizer crash → pass raw results forward
            log.warning("node.context_optimizer.exception", error=str(exc))
            return {
                "optimized_results": reranked,
                "optimizer_stats": {"reason": "optimizer_error", "error": str(exc)},
            }

        # Safeguard 1: aggressive passes reduced list to empty → top-8 fallback
        if not optimized:
            log.warning("node.context_optimizer.empty_output")
            return {
                "optimized_results": reranked[:8],
                "optimizer_stats": {**stats, "reason": "fallback_after_empty_optimize"},
            }

        log.debug("node.context_optimizer", before=len(reranked), after=len(optimized))
        return {"optimized_results": optimized, "optimizer_stats": stats}

    # ── Node 7: grade_context ─────────────────────────────────────────────────

    def grade_context(self, state: RAGState) -> dict:
        results = state.get("optimized_results", [])
        intent  = state.get("intent")
        query   = state.get("query", "").lower()

        if not results:
            log.info("node.grade_context", grade="retry", reason="no_results")
            return {"grade": "retry"}

        combined_text = " ".join(r.get("text", "") for r in results).lower()

        # Check 1: scheme type must appear in retrieved text
        if intent and intent.scheme_types:
            missing = [s for s in intent.scheme_types
                       if s.lower() not in combined_text]
            if missing:
                log.info("node.grade_context", grade="retry",
                         reason="scheme_missing", missing=missing)
                return {"grade": "retry"}

        # Check 2: numeric queries need at least one number in retrieved text
        numeric_query = any(kw in query for kw in
                            ("folio", "aum", "nav", "how many", "number of",
                             "total", "inflow", "outflow", "crore", "lakh",
                             "investment", "mobilized", "mobilised"))
        if numeric_query:
            has_numbers = bool(re.search(r'\d[\d,\.]+\d', combined_text))
            if not has_numbers:
                log.info("node.grade_context", grade="retry",
                         reason="no_numbers_for_numeric_query")
                return {"grade": "retry"}

        # Check 3: year-range queries need data from at least half the years
        year_from = getattr(intent, "year_from", None)
        year_to   = getattr(intent, "year_to", None)
        if year_from and year_to:
            years_needed   = set(range(year_from, year_to + 1))
            years_in_ctx   = {r.get("period_year") for r in results
                              if r.get("period_year")}
            years_covered  = years_needed & years_in_ctx
            min_coverage   = max(1, len(years_needed) // 2)   # at least 50%
            if len(years_covered) < min_coverage:
                log.info("node.grade_context", grade="retry",
                         reason="insufficient_year_coverage",
                         covered=sorted(years_covered),
                         needed=sorted(years_needed))
                return {"grade": "retry"}
            log.info("node.grade_context", grade="sufficient",
                     years_covered=sorted(years_covered))

        log.info("node.grade_context", grade="sufficient",
                 result_count=len(results))
        return {"grade": "sufficient"}

    # ── Node 8: rewrite_query ─────────────────────────────────────────────────

    def rewrite_query(self, state: RAGState) -> dict:
        original    = state.get("rewritten_query") or state.get("query", "")
        intent      = state.get("intent")
        retry_count = state.get("retry_count", 0)

        tokens   = [t for t in original.lower().split() if t not in _STOPWORDS]
        rewritten = " ".join(tokens)

        # On second retry, force scheme types to the front
        if intent and intent.scheme_types and retry_count >= 1:
            prefix    = " ".join(intent.scheme_types)
            rewritten = f"{prefix} {rewritten}"

        # Append temporal context if present but not in rewritten query
        if intent and intent.year and str(intent.year) not in rewritten:
            rewritten = f"{rewritten} {intent.year}"

        new_count = retry_count + 1
        log.info("node.rewrite_query", retry=new_count, rewritten=rewritten[:80])
        return {
            "rewritten_query": rewritten.strip(),
            "retry_count":     new_count,
        }

    # ── Node 9: augment ───────────────────────────────────────────────────────

    def augment(self, state: RAGState) -> dict:
        optimized = state.get("optimized_results", [])
        citations = self._cit_fmt.format(optimized)

        # Build context_text using PromptBuilder's internal block builder
        context_text = self._prompter._context_block(optimized, citations)

        log.info("node.augment", citations=len(citations),
                 context_chars=len(context_text))
        return {"citations": citations, "context_text": context_text}

    # ── Node 10: pre_guardrail ────────────────────────────────────────────────

    def pre_guardrail(self, state: RAGState) -> dict:
        intent   = state.get("intent")
        citations= state.get("citations", [])
        intent_t = intent.intent_type if intent else "factual"

        result  = self._pre_guard.check(
            question=state.get("query", ""),
            citations=citations,
            intent_type=intent_t,
            has_identified_fund=bool(intent.amc_names) if intent else False,
        )
        blocked = not result.should_proceed
        log.info("node.pre_guardrail", blocked=blocked,
                 advice=result.is_investment_advice)
        return {"pre_result": result, "blocked": blocked}

    # ── Node 11: generate ─────────────────────────────────────────────────────

    def generate(self, state: RAGState) -> dict:
        query        = state.get("query", "")
        context_text = state.get("context_text", "No relevant content found.")
        citations    = state.get("citations", [])
        intent       = state.get("intent")
        intent_t     = intent.intent_type if intent else "factual"

        structured_answer = state.get("structured_answer")
        if structured_answer:
            gen_meta = dict(state.get("generation_meta") or {})
            gen_meta.setdefault("model", "deterministic")
            gen_meta.setdefault("provider", "rules")
            gen_meta.setdefault("prompt_tokens", 0)
            gen_meta.setdefault("completion_tokens", 0)
            gen_meta.setdefault("latency_ms", 0)
            log.info("node.generate.deterministic",
                     model=gen_meta.get("model"),
                     provider=gen_meta.get("provider"))
            return {
                "answer": structured_answer,
                "generation_meta": gen_meta,
            }

        # SQL path: use the sql-specific prompt (context is a DB result, not doc chunks)
        if state.get("sql_context"):
            intent_t = "sql"

        if intent_t in self._EXTRACTIVE_INTENTS:
            answer = self._extractive_answer(query=query, citations=citations, intent_type=intent_t)
            log.info("node.generate.extractive",
                     intent=intent_t,
                     citations=len(citations),
                     answer_len=len(answer))
            return {
                "answer": answer,
                "structured_answer": answer,
                "generation_meta": {
                    "model":             "extractive",
                    "provider":          "rules",
                    "prompt_tokens":     0,
                    "completion_tokens": 0,
                    "latency_ms":        0,
                },
            }

        # Build full messages list (system prompt + user turn with context)
        messages = self._prompter.build(
            question=query,
            chunks=state.get("optimized_results", []),
            citations=citations,
            intent_type=intent_t,
        )

        try:
            result = self._generator.generate(messages, intent_type=intent_t)
            log.info("node.generate", model=result.model,
                     prompt_tokens=result.prompt_tokens,
                     completion_tokens=result.completion_tokens,
                     latency_ms=result.latency_ms)
            return {
                "answer": result.answer,
                "generation_meta": {
                    "model":             result.model,
                    "provider":          result.provider,
                    "prompt_tokens":     result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "latency_ms":        result.latency_ms,
                },
            }
        except Exception as exc:
            fallback = self._extractive_answer(query=query, citations=citations, intent_type=intent_t)
            log.warning("node.generate.fallback", error=str(exc), intent=intent_t, answer_len=len(fallback))
            return {
                "answer": fallback,
                "structured_answer": fallback,
                "generation_meta": {
                    "model":             "deterministic-fallback",
                    "provider":          "rules",
                    "prompt_tokens":     0,
                    "completion_tokens": 0,
                    "latency_ms":        0,
                },
            }

    _EXTRACTIVE_INTENTS = frozenset({
        "factual",
        "lookup",
        "definition",
        "tabular",
        "trend",
        "comparison",
        "regulatory",
    })

    def _extractive_answer(self, query: str, citations: list, intent_type: str) -> str:
        """Return a grounded answer assembled directly from retrieved citations."""
        if not citations:
            return (
                "I couldn’t find enough supporting passages in the retrieved AMFI documents "
                "to answer this confidently."
            )

        def _snippet(text: str) -> str:
            cleaned = " ".join((text or "").strip().split())
            if not cleaned:
                return ""
            if cleaned.startswith("|"):
                lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
                return "\n".join(lines[:5])
            # Prefer the first sentence; fall back to a short excerpt.
            sentence_end = re.search(r"[.!?]\s", cleaned)
            if sentence_end and sentence_end.start() >= 80:
                return cleaned[: sentence_end.end()].strip()
            return cleaned[:320]

        lines = [
            "Here’s the grounded evidence I found in the retrieved passages:",
        ]
        for citation in citations[:3]:
            snippet = _snippet(citation.excerpt)
            if not snippet:
                continue
            lines.append(
                f"• [{citation.number}] {citation.reference_string()} — {snippet}"
            )

        if len(lines) == 1:
            return (
                "I couldn’t find enough supporting passages in the retrieved AMFI documents "
                "to answer this confidently."
            )

        if intent_type == "lookup":
            lines.append("")
            lines.append("I’m keeping the answer limited to those passages to avoid inventing details.")

        return "\n".join(lines).strip()

    # ── Node 12: post_guardrail ───────────────────────────────────────────────

    def post_guardrail(self, state: RAGState) -> dict:
        answer    = state.get("answer", "")
        citations = state.get("citations", [])
        structured = bool(state.get("structured_answer"))

        result = self._post_guard.check(
            answer=answer,
            citations=citations,
            chunks=state.get("optimized_results", []),
        )

        # Analytical / SQL paths: numbers come from structured DB extraction, not RAG chunks.
        # number_consistent and faithfulness_score both compare against cited text excerpts —
        # meaningless when citations are DB rows rather than document passages.
        if state.get("is_analytical") or state.get("sql_context") or structured:
            result.citation_valid = True
            result.number_consistent = True
            result.hallucination_risk = "low"
            result.faithfulness_score = -2.0   # sentinel: "not applicable" (distinct from -1.0 = error)
            result.passed = result.answer_safe

        log.info("node.post_guardrail", passed=result.passed,
                 risk=result.hallucination_risk, safe=result.answer_safe,
                 analytical=state.get("is_analytical", False))
        return {"post_result": result}

    # ── Node 13: repair ───────────────────────────────────────────────────────

    def repair(self, state: RAGState) -> dict:
        original_answer = state.get("answer", "")
        post_result     = state.get("post_result")
        citations       = state.get("citations", [])
        repair_count    = state.get("repair_count", 0)

        try:
            repaired      = original_answer
            num_citations = len(citations)

            # Fix 1: strip sentences with invalid [N] citation markers
            if post_result and not post_result.citation_valid:
                def _cite_ok(sentence: str) -> bool:
                    markers = [int(m) for m in re.findall(r'\[(\d+)\]', sentence)]
                    return all(1 <= m <= num_citations for m in markers)
                sentences = re.split(r'(?<=[.!?])\s+', repaired)
                repaired  = " ".join(s for s in sentences if _cite_ok(s))

            # Fix 2: strip sentences with numbers not grounded in cited text
            if post_result and not post_result.number_consistent:
                cited_text = " ".join(c.excerpt for c in citations)
                def _nums_grounded(sentence: str) -> bool:
                    nums = re.findall(r'\b\d[\d,\.]*\d\b', sentence)
                    return all(n in cited_text for n in nums if len(n) >= 3)
                sentences = re.split(r'(?<=[.!?])\s+', repaired)
                repaired  = " ".join(s for s in sentences if _nums_grounded(s))

            log.info("node.repair.done",
                     before=len(original_answer), after=len(repaired))
            return {
                "answer":          repaired,
                "repair_attempted": True,
                "repair_count":    repair_count + 1,
            }

        except Exception as exc:
            # Production safeguard: repair crash must not strand the graph.
            # Return original answer + increment repair_count so
            # post_guardrail routes to abstain on next evaluation.
            log.warning("node.repair.failed", error=str(exc))
            return {
                "answer":          original_answer,
                "repair_attempted": True,
                "repair_count":    repair_count + 1,
            }

    # ── Node 14: format_response ──────────────────────────────────────────────

    def format_response(self, state: RAGState) -> dict:
        intent      = state.get("intent")
        pre_result  = state.get("pre_result")
        post_result = state.get("post_result")
        gen_meta    = state.get("generation_meta", {})
        citations   = state.get("citations", [])
        blocked     = state.get("blocked", False)
        grade       = state.get("grade", "sufficient")

        # Determine final answer text
        if blocked:
            answer = (pre_result.block_reason if pre_result and pre_result.block_reason
                      else "This query cannot be answered due to policy restrictions.")
        elif grade == "abstain" or (not state.get("answer")):
            answer = ("This information is not available in the provided AMFI documents "
                      "after multiple retrieval attempts.")
        else:
            answer = state.get("answer", "")

        response = {
            "question":          state.get("query", ""),
            "answer":            answer,
            "sources": [
                {
                    "citation":     c.number,
                    "file_name":    c.file_name,
                    "period_year":  c.period_year,
                    "period_month": c.period_month,
                    "category":     c.category,
                    "preview":      c.excerpt[:200],
                }
                for c in citations
            ],
            "model":              gen_meta.get("model", ""),
            "provider":           gen_meta.get("provider", ""),
            "latency_ms":         gen_meta.get("latency_ms", 0),
            "prompt_tokens":      gen_meta.get("prompt_tokens", 0),
            "completion_tokens":  gen_meta.get("completion_tokens", 0),
            "retrieval_count":    len(citations),
            "retry_count":        state.get("retry_count", 0),
            "repair_attempted":   state.get("repair_attempted", False),
            "guardrail": {
                "blocked":              blocked,
                "block_reason":         pre_result.block_reason if pre_result else None,
                "is_investment_advice": pre_result.is_investment_advice if pre_result else False,
                "is_out_of_scope":      pre_result.is_out_of_scope if pre_result else False,
                "pre_passed":           pre_result.should_proceed if pre_result else True,
                "post_passed":          post_result.passed if post_result else True,
                "answer_safe":          post_result.answer_safe if post_result else True,
                "hallucination_risk":   post_result.hallucination_risk if post_result else "unknown",
                "faithfulness_score":   (
                    None if (post_result and post_result.faithfulness_score == -2.0)
                    else (post_result.faithfulness_score if post_result else -1.0)
                ),
                "citation_valid":       post_result.citation_valid if post_result else True,
                "number_consistent":    post_result.number_consistent if post_result else True,
                "abstention_detected":  post_result.abstention_detected if post_result else False,
                "warnings": (
                    (pre_result.warnings if pre_result else []) +
                    (post_result.warnings if post_result else [])
                ),
            },
            "optimizer_stats": state.get("optimizer_stats", {}),
            "grade":           grade,
        }

        log.info("node.format_response",
                 blocked=blocked, answer_len=len(answer),
                 citations=len(citations))
        return {"response": response}
