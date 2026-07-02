"""RAGPipeline — retrieval-augmented generation over the AMFI knowledge base.

Flow:
  query → Retriever (hybrid search, pgvector + BM25)
        → ContextBuilder (numbered citations)
        → LLM (auto-detected: OpenAI or Anthropic/Claude)
        → Answer + cited sources

Provider auto-detection from API key prefix:
  sk-ant-...  → Anthropic Claude  (claude-3-5-haiku-20241022 default)
  sk-...      → OpenAI            (gpt-4o-mini default)
  anything    → OpenAI-compatible (set OPENAI_BASE_URL for Ollama, Azure, etc.)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from financial_pipeline.config import settings
from financial_pipeline.retrieval.context import ContextBuilder
from financial_pipeline.retrieval.retriever import Retriever

log = structlog.get_logger()


@dataclass
class RAGResponse:
    query:        str
    answer:       str
    sources:      list[dict]   = field(default_factory=list)
    model:        str          = ""
    latency_ms:   int          = 0
    prompt_tokens: int         = 0
    completion_tokens: int     = 0
    retrieval_count: int       = 0


class RAGPipeline:
    """Retrieve-then-generate pipeline.

    Parameters
    ----------
    retriever:
        A configured :class:`Retriever` instance.
    top_k:
        Number of chunks to retrieve per query (default from config).
    llm_model:
        Model name forwarded to the chat-completion API.
    stream:
        If True, streams the completion (not yet exposed via HTTP — useful
        for CLI usage).
    """

    def __init__(
        self,
        retriever:  Retriever,
        top_k:      int | None  = None,
        llm_model:  str | None  = None,
        stream:     bool        = False,
    ) -> None:
        self._retriever      = retriever
        self._ctx            = ContextBuilder()
        self._top_k          = top_k or settings.api_top_k
        self._model          = llm_model or settings.openai_model
        self._stream         = stream
        self._client         = None   # lazy init

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def ask(
        self,
        question:  str,
        year:      int | None = None,
        month:     int | None = None,
        category:  str | None = None,
        top_k:     int | None = None,
    ) -> RAGResponse:
        """Retrieve relevant chunks and generate a grounded answer.

        Raises `ValueError` if `OPENAI_API_KEY` is not set.
        """
        t0 = time.perf_counter()

        # 1. Retrieve
        limit  = top_k or self._top_k
        chunks = self._retriever.get_context_chunks(
            question, limit=limit, year=year, month=month, category=category
        )
        log.info("rag.retrieved", count=len(chunks), question=question[:80])

        # 2. Build messages
        messages = self._ctx.build_messages(question, chunks)
        sources  = self._ctx.format_sources(chunks)

        # 3. Generate (provider-agnostic)
        answer, usage = self._complete(messages, self._model)

        latency = int((time.perf_counter() - t0) * 1000)

        # Token counts differ between providers
        if settings.llm_provider == "anthropic":
            prompt_tok = getattr(usage, "input_tokens",  0) if usage else 0
            compl_tok  = getattr(usage, "output_tokens", 0) if usage else 0
        else:
            prompt_tok = getattr(usage, "prompt_tokens",     0) if usage else 0
            compl_tok  = getattr(usage, "completion_tokens", 0) if usage else 0

        log.info("rag.generated",
                 provider=settings.llm_provider,
                 model=self._model,
                 latency_ms=latency,
                 prompt_tokens=prompt_tok,
                 completion_tokens=compl_tok)

        return RAGResponse(
            query             = question,
            answer            = answer,
            sources           = sources,
            model             = self._model,
            latency_ms        = latency,
            prompt_tokens     = prompt_tok,
            completion_tokens = compl_tok,
            retrieval_count   = len(chunks),
        )

    def is_llm_configured(self) -> bool:
        return bool(settings.openai_api_key)

    @property
    def provider(self) -> str:
        return settings.llm_provider

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            if not settings.openai_api_key:
                raise ValueError(
                    "No LLM API key set. Add OPENAI_API_KEY (OpenAI) or "
                    "an Anthropic key (sk-ant-...) to .env to enable /api/ask."
                )
            if settings.llm_provider == "anthropic":
                from anthropic import Anthropic
                self._client = Anthropic(api_key=settings.openai_api_key)
            else:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key  = settings.openai_api_key,
                    base_url = settings.openai_base_url,
                )
        return self._client

    def _complete(self, messages: list[dict], model: str) -> tuple[str, object]:
        """Call the appropriate LLM API and return (answer_text, usage)."""
        client = self._get_client()

        if settings.llm_provider == "anthropic":
            # Anthropic separates system prompt from user messages
            system = next((m["content"] for m in messages if m["role"] == "system"), "")
            user_msgs = [m for m in messages if m["role"] != "system"]
            resp = client.messages.create(
                model      = model,
                max_tokens = 1024,
                system     = system,
                messages   = user_msgs,
            )
            answer = resp.content[0].text
            usage  = resp.usage
            return answer, usage
        else:
            resp   = client.chat.completions.create(
                model       = model,
                messages    = messages,
                temperature = 0.1,
                max_tokens  = 1024,
            )
            return resp.choices[0].message.content or "", resp.usage
