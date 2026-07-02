"""RetrievalPipeline — the full 5-stage retrieval flow.

Stage 1  QueryAnalyzer      → understand the question, extract intent
Stage 2  SearchRouter       → decide which sources to query and how
Stage 3  MultiSourceFetcher → fetch chunks, table metadata, documents
Stage 4  ResultRanker       → deduplicate, score, MMR diversity filter
Stage 5  ContextAssembler   → numbered citations, formatted context text

Usage::

    pipeline = RetrievalPipeline(repo)
    result   = pipeline.retrieve("How many equity schemes existed in March 2024?")

    print(result.context_text)   # LLM-ready passage block
    print(result.citations)      # structured source list
    print(result.intent.describe())   # what the pipeline understood
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from financial_pipeline.config import settings
from financial_pipeline.retrieval.query_understanding import QueryAnalyzer, QueryIntent
from financial_pipeline.retrieval.retriever import Retriever
from financial_pipeline.storage.document_repo import DocumentRepository

log = structlog.get_logger()


# ── Output types ──────────────────────────────────────────────────────────────


@dataclass
class Citation:
    number: int
    source_type: str  # "chunk" | "table" | "document"
    file_name: str | None
    period_year: int | None
    period_month: int | None
    category: str | None
    preview: str
    score: float | None = None
    chunk_index: int | None = None
    table_index: int | None = None


@dataclass
class GroundedContext:
    """Everything downstream (LLM, API response) needs from one retrieval."""

    query: str
    intent: QueryIntent
    context_text: str  # formatted for LLM system message
    citations: list[Citation]  # numbered, structured
    raw_results: list[dict]  # unformatted, for API pass-through
    table_data: list[dict]  # table summaries when tables fetched
    stats: dict[str, Any] = field(default_factory=dict)  # timing, counts


# ── Stage 2: Search Router ────────────────────────────────────────────────────


@dataclass
class SearchPlan:
    chunk_mode: str = "hybrid"  # hybrid | semantic | keyword
    chunk_limit: int = 12  # fetch more than we'll return (for MMR)
    fetch_tables: bool = False
    table_limit: int = 5
    fetch_documents: bool = False
    doc_limit: int = 5


class SearchRouter:
    """Stage 2 — Converts a QueryIntent into a concrete SearchPlan."""

    def route(self, intent: QueryIntent, top_k: int = 8) -> SearchPlan:
        plan = SearchPlan(chunk_limit=top_k * 2)  # fetch 2× for MMR headroom

        # Chunk search mode
        if intent.intent_type in ("definition", "factual"):
            plan.chunk_mode = "hybrid"
        elif intent.intent_type == "trend":
            plan.chunk_mode = "hybrid"
        elif intent.intent_type == "tabular":
            plan.chunk_mode = "keyword"  # tabular queries have exact term matches
        elif intent.intent_type == "comparison":
            plan.chunk_mode = "semantic"  # comparisons are conceptual
        elif intent.intent_type == "lookup":
            plan.chunk_mode = "keyword"
        else:
            plan.chunk_mode = "hybrid"

        # Table search
        if intent.needs_tables or intent.intent_type in ("tabular", "trend"):
            plan.fetch_tables = True

        # Document-level search
        if intent.needs_documents or intent.intent_type == "lookup":
            plan.fetch_documents = True

        log.debug(
            "router.plan",
            intent=intent.intent_type,
            chunk_mode=plan.chunk_mode,
            fetch_tables=plan.fetch_tables,
        )
        return plan


# ── Stage 3: Multi-Source Fetcher ─────────────────────────────────────────────


class MultiSourceFetcher:
    """Stage 3 — Fetches results from all sources the router selected."""

    def __init__(self, repo: DocumentRepository, retriever: Retriever) -> None:
        self._repo = repo
        self._retriever = retriever

    def fetch(
        self,
        intent: QueryIntent,
        plan: SearchPlan,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Returns (chunks, tables, documents)."""
        chunks = self._fetch_chunks(intent, plan)
        tables = self._fetch_tables(intent, plan) if plan.fetch_tables else []
        documents = self._fetch_documents(intent, plan) if plan.fetch_documents else []
        return chunks, tables, documents

    def _fetch_chunks(self, intent: QueryIntent, plan: SearchPlan) -> list[dict]:
        query = intent.search_query or intent.raw_query
        try:
            results = self._retriever.search(
                query=query,
                mode=plan.chunk_mode,
                limit=plan.chunk_limit,
                year=intent.year,
                month=intent.month,
                category=intent.category,
            )
            for r in results:
                r["_source"] = "chunk"
            return results
        except Exception as exc:
            log.warning("fetcher.chunks_failed", error=str(exc))
            return []

    def _fetch_tables(self, intent: QueryIntent, plan: SearchPlan) -> list[dict]:
        """Query document_table_assets for tables matching the intent filters."""
        conditions = ["1=1"]
        params: dict = {"limit": plan.table_limit}

        if intent.year:
            conditions.append("dm.period_year = :year")
            params["year"] = intent.year
        if intent.month:
            conditions.append("dm.period_month = :month")
            params["month"] = intent.month
        if intent.category:
            conditions.append("dm.category = :category")
            params["category"] = intent.category

        where = " AND ".join(conditions)
        from sqlalchemy import text as sa_text

        sql = f"""
            SELECT ta.table_asset_id, ta.table_index, ta.table_name,
                   ta.page_number, ta.parquet_s3_key,
                   ta.row_count, ta.column_count, ta.schema_json,
                   dm.file_name, dm.period_year, dm.period_month, dm.category
              FROM document_table_assets ta
              JOIN document_metadata dm ON ta.document_id = dm.document_id
             WHERE {where}
             ORDER BY dm.period_year DESC NULLS LAST,
                      dm.period_month DESC NULLS LAST,
                      ta.table_index ASC
             LIMIT :limit
        """
        try:
            with self._repo._engine.connect() as conn:
                rows = conn.execute(sa_text(sql), params).mappings().all()
            results = [dict(r) | {"_source": "table"} for r in rows]
            log.debug("fetcher.tables", count=len(results))
            return results
        except Exception as exc:
            log.warning("fetcher.tables_failed", error=str(exc))
            return []

    def _fetch_documents(self, intent: QueryIntent, plan: SearchPlan) -> list[dict]:
        """Metadata-level lookup for document-type queries."""
        conditions = ["processing_status = 'embedded'"]
        params: dict = {"limit": plan.doc_limit}

        if intent.year:
            conditions.append("period_year = :year")
            params["year"] = intent.year
        if intent.month:
            conditions.append("period_month = :month")
            params["month"] = intent.month
        if intent.category:
            conditions.append("category = :category")
            params["category"] = intent.category

        where = " AND ".join(conditions)
        from sqlalchemy import text as sa_text

        sql = f"""
            SELECT document_id, file_name, file_type, category,
                   period_year, period_month, s3_processed_key
              FROM document_metadata
             WHERE {where}
             ORDER BY period_year DESC NULLS LAST, period_month DESC NULLS LAST
             LIMIT :limit
        """
        try:
            with self._repo._engine.connect() as conn:
                rows = conn.execute(sa_text(sql), params).mappings().all()
            results = [dict(r) | {"_source": "document"} for r in rows]
            log.debug("fetcher.documents", count=len(results))
            return results
        except Exception as exc:
            log.warning("fetcher.documents_failed", error=str(exc))
            return []


# ── Stage 4: Result Ranker ────────────────────────────────────────────────────


class ResultRanker:
    """Stage 4 — Deduplicate, score, and diversify retrieved results.

    Applies three passes:
    1. Deduplication   — same chunk_id or near-identical text (first 80 chars)
    2. Source blending — interleave chunks and tables so neither dominates
    3. MMR pruning     — penalise back-to-back chunks from the same document
                         (encourages breadth across the corpus)
    """

    MMR_LAMBDA = 0.7  # 0=max diversity, 1=max relevance
    MIN_SCORE = 0.0  # drop chunks below this similarity
    # Chunks where >40% of characters are table-noise (|, -, spaces in patterns)
    # are mostly raw XLS table rows — useless for prose queries.
    TABLE_NOISE_THRESHOLD = 0.40

    def _is_prose(self, text: str, intent: QueryIntent) -> bool:
        """Return False for chunks that are mostly raw table-data noise."""
        if not text:
            return False
        # Tabular and trend intents accept table-heavy chunks
        if intent.intent_type in ("tabular", "trend"):
            return True
        pipe_ratio = text.count("|") / max(len(text), 1)
        return pipe_ratio < self.TABLE_NOISE_THRESHOLD

    def rank(
        self,
        chunks: list[dict],
        tables: list[dict],
        documents: list[dict],
        top_k: int,
        intent: QueryIntent,
    ) -> list[dict]:

        # 1. Deduplicate + quality-filter chunks
        seen_ids: set[str] = set()
        seen_prefix: set[str] = set()
        deduped: list[dict] = []
        for r in chunks:
            cid = str(r.get("chunk_id", ""))
            text = r.get("text") or ""
            prefix = text[:80].strip()
            if cid in seen_ids or prefix in seen_prefix:
                continue
            if (r.get("similarity") or 0) < self.MIN_SCORE:
                continue
            if not self._is_prose(text, intent):
                continue  # skip raw table-data chunks for prose queries
            seen_ids.add(cid)
            seen_prefix.add(prefix)
            deduped.append(r)

        # 2. MMR diversity pass — penalise same-document repetition
        selected: list[dict] = []
        doc_seen: dict[str, int] = {}  # doc_id → count selected from it

        for candidate in deduped:
            doc_id = str(candidate.get("document_id", ""))
            penalty = doc_seen.get(doc_id, 0) * (1 - self.MMR_LAMBDA)
            rel_score = candidate.get("similarity") or candidate.get("rrf_score") or 0.5
            mmr_score = self.MMR_LAMBDA * rel_score - penalty
            candidate["_mmr_score"] = round(mmr_score, 5)
            selected.append(candidate)
            doc_seen[doc_id] = doc_seen.get(doc_id, 0) + 1

        # Sort by MMR score
        selected.sort(key=lambda r: r.get("_mmr_score", 0), reverse=True)
        ranked = selected[:top_k]

        # 3. Append table results after ranked chunks (tables have no similarity score)
        table_slots = max(0, top_k - len(ranked))
        if tables and table_slots > 0:
            ranked.extend(tables[:table_slots])

        # 4. Append document results if lookup intent
        if documents and intent.intent_type == "lookup":
            for doc in documents[:2]:
                if len(ranked) < top_k + 2:
                    ranked.append(doc)

        log.debug("ranker.done", chunks_in=len(deduped), tables_in=len(tables), ranked_out=len(ranked))
        return ranked


# ── Stage 5: Context Assembler ────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a financial analyst specialising in Indian mutual funds.
Answer questions using ONLY the numbered context passages below.
Cite each fact with its passage number: "Total AUM grew 41% [1]."
If the answer is not in the passages, say: "This information is not in the provided documents."
Use exact figures when the context provides them. Do not fabricate data.
"""


class ContextAssembler:
    """Stage 5 — Format ranked results into LLM-ready text with numbered citations."""

    MAX_CHUNK_CHARS = 600
    MAX_CONTEXT_CHARS = 6000

    def assemble(
        self,
        ranked: list[dict],
        intent: QueryIntent,
    ) -> tuple[str, list[Citation]]:
        """Returns (context_text, citations)."""
        passages: list[str] = []
        citations: list[Citation] = []
        total_chars = 0

        for i, item in enumerate(ranked, 1):
            src = item.get("_source", "chunk")

            if src == "chunk":
                text = (item.get("text") or "").strip()
                if len(text) > self.MAX_CHUNK_CHARS:
                    text = text[: self.MAX_CHUNK_CHARS] + "…"
                meta = self._meta(item)
                passage = f"[{i}] {meta}\n{text}"

            elif src == "table":
                schema = item.get("schema_json") or []
                cols = ", ".join(c.get("name", "") for c in schema[:8])
                meta = self._meta(item)
                rows = item.get("row_count", "?")
                passage = (
                    f"[{i}] {meta}  (table #{item.get('table_index', i)})\n"
                    f"Columns: {cols}\n"
                    f"Rows: {rows}  |  "
                    f"Key: {item.get('parquet_s3_key', '')}"
                )

            elif src == "document":
                meta = self._meta(item)
                passage = f"[{i}] {meta}\nDocument: {item.get('file_name', '')}  |  {item.get('s3_processed_key', '')}"

            else:
                continue

            if total_chars + len(passage) > self.MAX_CONTEXT_CHARS:
                break

            passages.append(passage)
            total_chars += len(passage)

            citations.append(
                Citation(
                    number=i,
                    source_type=src,
                    file_name=item.get("file_name"),
                    period_year=item.get("period_year"),
                    period_month=item.get("period_month"),
                    category=item.get("category"),
                    preview=(item.get("text") or item.get("table_name") or "")[:200],
                    score=item.get("similarity") or item.get("rrf_score") or item.get("_mmr_score"),
                    chunk_index=item.get("chunk_index"),
                    table_index=item.get("table_index"),
                )
            )

        context_text = "\n\n".join(passages) if passages else "No relevant content found."
        return context_text, citations

    def build_llm_messages(
        self,
        question: str,
        context_text: str,
    ) -> list[dict]:
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n\n{context_text}\n\nQuestion: {question}"},
        ]

    @staticmethod
    def _meta(item: dict) -> str:
        parts = [item.get("file_name") or "unknown"]
        y, m = item.get("period_year"), item.get("period_month")
        if y and m:
            parts.append(f"{y}/{m:02d}")
        elif y:
            parts.append(str(y))
        if item.get("category"):
            parts.append(item["category"])
        return "  |  ".join(parts)


# ── Orchestrator ──────────────────────────────────────────────────────────────


class RetrievalPipeline:
    """Orchestrates all 5 retrieval stages.

    Parameters
    ----------
    repo:
        DocumentRepository (owns DB connection + search methods).
    retriever:
        Retriever (lazy-loads embedding model, wraps pgvector search).
    top_k:
        Final number of results to return (post-MMR).
    """

    def __init__(
        self,
        repo: DocumentRepository,
        retriever: Retriever,
        top_k: int | None = None,
    ) -> None:
        self._analyzer = QueryAnalyzer()
        self._router = SearchRouter()
        self._fetcher = MultiSourceFetcher(repo, retriever)
        self._ranker = ResultRanker()
        self._assembler = ContextAssembler()
        self._top_k = top_k or settings.api_top_k

    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        # explicit overrides (passed from API when user sets them directly)
        year: int | None = None,
        month: int | None = None,
        category: str | None = None,
        mode: str | None = None,  # override chunk search mode
    ) -> GroundedContext:
        t0 = time.perf_counter()
        k = top_k or self._top_k

        # ── Stage 1: Understand ───────────────────────────────────────
        intent = self._analyzer.analyze(query)
        # API-level overrides take priority over extracted values
        if year:
            intent.year = year
        if month:
            intent.month = month
        if category:
            intent.category = category
        log.info(
            "pipeline.intent",
            **{
                k: v
                for k, v in {
                    "intent": intent.intent_type,
                    "year": intent.year,
                    "month": intent.month,
                    "needs_tables": intent.needs_tables,
                }.items()
                if v is not None
            },
        )

        # ── Stage 2: Route ────────────────────────────────────────────
        plan = self._router.route(intent, top_k=k)
        if mode:
            plan.chunk_mode = mode

        # ── Stage 3: Fetch ────────────────────────────────────────────
        chunks, tables, documents = self._fetcher.fetch(intent, plan)

        # ── Stage 4: Rank ─────────────────────────────────────────────
        ranked = self._ranker.rank(chunks, tables, documents, k, intent)

        # ── Stage 5: Assemble ─────────────────────────────────────────
        context_text, citations = self._assembler.assemble(ranked, intent)

        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        log.info("pipeline.done", results=len(ranked), citations=len(citations), latency_ms=elapsed_ms)

        return GroundedContext(
            query=query,
            intent=intent,
            context_text=context_text,
            citations=citations,
            raw_results=ranked,
            table_data=tables,
            stats={
                "latency_ms": elapsed_ms,
                "chunks_fetched": len(chunks),
                "tables_fetched": len(tables),
                "docs_fetched": len(documents),
                "ranked_returned": len(ranked),
                "chunk_mode": plan.chunk_mode,
            },
        )

    def build_llm_messages(
        self,
        question: str,
        context_text: str,
    ) -> list[dict]:
        return self._assembler.build_llm_messages(question, context_text)
