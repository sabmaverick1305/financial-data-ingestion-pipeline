"""AugmentationPipeline — orchestrates all stages including dual guardrails.

Full flow:

  User question
      │
      ▼
  ① RetrievalPipeline        → raw chunks + QueryIntent
      │
      ▼
  ② ContextRanker            cross-encoder re-ranks chunks
      │
      ▼
  ③ CitationFormatter        assigns [1]…[N] markers + confidence scores
      │
      ▼
  ④ PRE-GENERATION GUARDRAILS  ← NEW
      │ 1. Investment advice block
      │ 2. Unsupported claims check
      │ 3. Source requirement
      │
      ├── BLOCKED? → return BlockedResponse immediately (no LLM call)
      │
      ▼
  ⑤ PromptBuilder            intent-aware system prompt + context block
      │
      ▼
  ⑥ AnswerGenerator          LLM call (Anthropic / OpenAI, auto-detected)
      │
      ▼
  ⑦ POST-GENERATION GUARDRAILS  ← NEW (was ⑥ HallucinationGuardrails)
      │ 4. Citation validation
      │ 5. Hallucination detection
      │ 6. Answer safety (financial advice language)
      │ 7. Numeric consistency
      │
      ▼
  AugmentedResponse
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import structlog

from financial_pipeline.augmentation.citations import Citation, CitationFormatter
from financial_pipeline.augmentation.generator import AnswerGenerator, GenerationResult
from financial_pipeline.augmentation.guardrails import (
    PostGenerationGuardrails,
    PostGuardrailResult,
    PreGenerationGuardrails,
    PreGuardrailResult,
)
from financial_pipeline.augmentation.prompts import PromptBuilder
from financial_pipeline.augmentation.ranker import ContextRanker
from financial_pipeline.retrieval.pipeline import RetrievalPipeline

log = structlog.get_logger()


# ── Response dataclasses ──────────────────────────────────────────────────────


@dataclass
class GuardrailSummary:
    """Combined view of both guardrail layers — what the API exposes."""

    pre_passed: bool
    post_passed: bool
    blocked: bool
    block_reason: str | None
    is_investment_advice: bool
    answer_safe: bool
    hallucination_risk: str  # "low" | "medium" | "high"
    faithfulness_score: float
    citation_valid: bool
    number_consistent: bool
    abstention_detected: bool
    warnings: list[str]

    @property
    def passed(self) -> bool:
        return not self.blocked and self.post_passed

    def summary(self) -> str:
        if self.blocked:
            return f"blocked: {self.block_reason[:60] if self.block_reason else '?'}"
        if self.abstention_detected:
            return "abstained"
        if not self.post_passed:
            return "post_failed"
        return "passed"


@dataclass
class AugmentedResponse:
    """Everything the API or caller needs from one augmented Q&A."""

    question: str
    answer: str
    citations: list[Citation]
    guardrail: GuardrailSummary
    generation: GenerationResult | None  # None when blocked pre-gen
    intent_type: str
    retrieval_stats: dict = field(default_factory=dict)
    augment_stats: dict = field(default_factory=dict)

    @property
    def is_safe(self) -> bool:
        return self.guardrail.passed

    @property
    def abstained(self) -> bool:
        return self.guardrail.abstention_detected

    @property
    def was_blocked(self) -> bool:
        return self.guardrail.blocked


# ── Sentinel generation result (used when pre-guardrails block) ───────────────
_BLOCKED_GEN = GenerationResult(answer="", model="", provider="", prompt_tokens=0, completion_tokens=0, latency_ms=0)


# ── Pipeline ──────────────────────────────────────────────────────────────────


class AugmentationPipeline:
    """Orchestrates retrieval, re-ranking, citation, dual guardrails, and generation.

    Parameters
    ----------
    retrieval_pipeline:
        Configured RetrievalPipeline (query understanding + search + MMR).
    use_cross_encoder:
        Enable cross-encoder re-ranking (Stage 2).
    top_k_retrieve:
        Chunks fetched from retrieval (fed into cross-encoder).
    top_k_augment:
        Chunks kept after re-ranking (go into the LLM prompt).
    embed_model:
        sentence-transformers model for faithfulness scoring in post-guardrails.
        If None, faithfulness check is skipped (faster, less accurate).
    """

    def __init__(
        self,
        retrieval_pipeline: RetrievalPipeline,
        use_cross_encoder: bool = True,
        top_k_retrieve: int = 12,
        top_k_augment: int = 6,
        embed_model=None,
    ) -> None:
        self._retrieval = retrieval_pipeline
        self._ranker = ContextRanker(use_cross_encoder=use_cross_encoder, top_k=top_k_augment)
        self._cit_fmt = CitationFormatter()
        self._pre_guards = PreGenerationGuardrails()
        self._prompt_bld = PromptBuilder()
        self._generator = AnswerGenerator()
        self._post_guards = PostGenerationGuardrails(embed_model=embed_model)
        self._top_k_ret = top_k_retrieve

    # ------------------------------------------------------------------

    def run(
        self,
        question: str,
        year: int | None = None,
        month: int | None = None,
        category: str | None = None,
        model: str | None = None,
        top_k: int | None = None,
    ) -> AugmentedResponse:
        """Run the full augmentation pipeline with dual guardrails."""
        t0 = time.perf_counter()

        if not self._generator.is_configured():
            raise ValueError("No LLM API key. Set OPENAI_API_KEY in .env.")

        # ── Stage 1: Retrieve ─────────────────────────────────────────
        ctx = self._retrieval.retrieve(
            query=question,
            top_k=top_k or self._top_k_ret,
            year=year,
            month=month,
            category=category,
        )
        intent_type = ctx.intent.intent_type
        log.info("augmentation.retrieved", intent=intent_type, chunks=len(ctx.raw_results))

        # ── Stage 2: Re-rank ──────────────────────────────────────────
        reranked = self._ranker.rerank(question, ctx.raw_results)

        # ── Stage 3: Format citations ─────────────────────────────────
        citations = self._cit_fmt.format(reranked)

        # ── Stage 4: PRE-GENERATION GUARDRAILS ───────────────────────
        pre = self._pre_guards.check(
            question=question,
            citations=citations,
            intent_type=intent_type,
        )
        log.info(
            "pre_guardrail.summary",
            proceed=pre.should_proceed,
            advice=pre.is_investment_advice,
            sources=pre.source_count,
            quality=pre.context_quality,
        )

        if not pre.should_proceed:
            # Block immediately — do NOT call the LLM
            total_ms = int((time.perf_counter() - t0) * 1000)
            log.warning("augmentation.blocked", reason=pre.block_reason)
            return AugmentedResponse(
                question=question,
                answer=pre.block_reason or "Request blocked by safety guardrails.",
                citations=[],
                guardrail=_make_summary(pre, None, blocked=True),
                generation=_BLOCKED_GEN,
                intent_type=intent_type,
                retrieval_stats=ctx.stats,
                augment_stats={
                    "total_latency_ms": total_ms,
                    "blocked": True,
                    "intent": intent_type,
                },
            )

        # ── Stage 5: Build prompt ─────────────────────────────────────
        messages = self._prompt_bld.build(
            question=question,
            chunks=reranked,
            citations=citations,
            intent_type=intent_type,
            add_few_shot=True,
        )

        # ── Stage 6: Generate ─────────────────────────────────────────
        gen = self._generator.generate(
            messages=messages,
            intent_type=intent_type,
            model=model,
        )

        # ── Stage 7: POST-GENERATION GUARDRAILS ──────────────────────
        post = self._post_guards.check(
            answer=gen.answer,
            citations=citations,
            chunks=reranked,
        )

        total_ms = int((time.perf_counter() - t0) * 1000)
        summary = _make_summary(pre, post, blocked=False)

        log.info(
            "augmentation.done",
            intent=intent_type,
            citations=len(citations),
            guardrail=summary.summary(),
            total_ms=total_ms,
        )

        return AugmentedResponse(
            question=question,
            answer=gen.answer,
            citations=citations,
            guardrail=summary,
            generation=gen,
            intent_type=intent_type,
            retrieval_stats=ctx.stats,
            augment_stats={
                "total_latency_ms": total_ms,
                "llm_latency_ms": gen.latency_ms,
                "chunks_retrieved": len(ctx.raw_results),
                "chunks_reranked": len(reranked),
                "chunks_in_context": len(citations),
                "prompt_tokens": gen.prompt_tokens,
                "completion_tokens": gen.completion_tokens,
                "model": gen.model,
                "provider": gen.provider,
                "intent": intent_type,
                "pre_guardrail": pre.summary(),
                "post_guardrail": post.summary() if post else "n/a",
                "context_quality": pre.context_quality,
            },
        )

    def is_configured(self) -> bool:
        return self._generator.is_configured()


# ── Helper ────────────────────────────────────────────────────────────────────


def _make_summary(
    pre: PreGuardrailResult,
    post: PostGuardrailResult | None,
    blocked: bool,
) -> GuardrailSummary:
    warnings = list(pre.warnings) + (list(post.warnings) if post else [])
    return GuardrailSummary(
        pre_passed=pre.should_proceed,
        post_passed=post.passed if post else True,
        blocked=blocked,
        block_reason=pre.block_reason,
        is_investment_advice=pre.is_investment_advice,
        answer_safe=post.answer_safe if post else True,
        hallucination_risk=post.hallucination_risk if post else "unknown",
        faithfulness_score=post.faithfulness_score if post else -1.0,
        citation_valid=post.citation_valid if post else False,
        number_consistent=post.number_consistent if post else True,
        abstention_detected=post.abstention_detected if post else False,
        warnings=warnings,
    )
