"""Guardrails — dual-layer safety checks before and after LLM generation.

                    ┌─────────────────────────────┐
   Retrieved chunks │  PRE-GENERATION GUARDRAILS  │
   + QueryIntent    │                             │
                    │  1. Investment advice block  │
                    │  2. Unsupported claims check │
                    │  3. Source requirement       │
                    └────────────┬────────────────┘
                                 │  proceed / block
                                 ▼
                            LLM generates
                                 │
                    ┌────────────▼────────────────┐
   Generated answer │  POST-GENERATION GUARDRAILS │
   + citations      │                             │
                    │  4. Citation validation      │
                    │  5. Hallucination detection  │
                    │  6. Answer safety            │
                    │  7. Numeric consistency      │
                    └─────────────────────────────┘

Pre-guardrails decide whether the LLM should run at all.
Post-guardrails validate the output and attach quality signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

log = structlog.get_logger()

# ── Shared vocabulary ─────────────────────────────────────────────────────────

# Phrase the prompts instruct the model to use when it can't answer
ABSTENTION_PHRASE = "not available in the provided amfi documents"

# Minimum sources per intent type before proceeding to generation
MIN_SOURCES_BY_INTENT: dict[str, int] = {
    "factual": 1,
    "regulatory": 1,
    "trend": 2,
    "comparison": 2,
    "definition": 1,
    "tabular": 1,
    "lookup": 1,
    "default": 1,
}


# ══════════════════════════════════════════════════════════════════════════════
# PRE-GENERATION GUARDRAILS
# ══════════════════════════════════════════════════════════════════════════════

# Check 1 — Investment advice patterns (SEBI prohibits robo-advice without
# licence; our system answers factual questions, not personal finance advice)
_ADVICE_PATTERNS = [
    r"\bshould (i|we|you)\s+(invest|buy|sell|exit|switch|move|transfer|withdraw|redeem|put money|allocate)\b",
    r"\b(recommend|suggest|advise)\b.{0,30}\b(fund|scheme|stock|invest)\b",
    r"\bwhich fund (is best|to buy|should i|would you)\b",
    r"\bwhich\b.{0,50}\bfunds?\b.{0,50}\b(should|buy|invest|pick|choose|go for)\b",
    r"\bbest\s+(performing\s+)?funds?\b.{0,80}\b(cagr|return|returns|performance|invest)\b",
    r"\b(top|best)\s+\d*\s*funds?\b.{0,80}\b(cagr|return|returns|performance|invest)\b",
    r"\bfunds?\b.{0,50}\b(good|strong|high)\s+cagr\b",
    r"\b(good|strong|high)\s+cagr\b.{0,50}\bfunds?\b",
    r"\bguaranteed\s+(return|profit|gain|growth)\b",
    r"\b(safe|risk.free)\s+(bet|investment|option)\b",
    r"\bwill\s+.{0,20}\b(definitely|certainly)\s+(grow|increase|give|earn)\b",
    r"\bbest fund(s)?\s+to\s+(invest|buy)\b",
    r"\b(pick|choose|select)\s+.{0,15}\bfund\b.{0,30}\bfor me\b",
    r"\bis it (a good idea|advisable|wise|smart)\b.{0,40}\b(invest|put|allocate|park|buy)\b",
    r"\bhow should (i|we)\s+(invest|allocate|diversify|park)\b",
    # Additional patterns for advice queries that imply personal recommendations
    r"\bwhich\b.{0,60}\bfund\b.{0,40}\b(pick|choose|go for)\b",
    r"\bwould you (recommend|suggest)\b",
]

# Check 2b — Out-of-scope queries: topics outside AMFI mutual fund data
# (stock prices, index levels, bank products, individual fund NAV, crypto markets)
_OUT_OF_SCOPE_PATTERNS = [
    r"\b(current\s+)?(stock|share|equity)\s+price\b",
    # No trailing \b — matches "performed", "performance", "returned", etc.
    r"\b(nifty|sensex)\b.{0,30}(perform|return|level|point|rose|fell|gain|loss|today)",
    r"\b(fd|fixed deposit)\s+rates?\b",
    r"\bcurrent\s+nav\b",
    r"\bnav\b.{0,40}\btoday\b",
    # No leading \b on second group — matches "investment", "trending", etc.
    r"\bcryptocurren(cy|cies)\b.{0,30}(invest|trend|market|perform)",
]

# Check 2 — Unsupported claim indicators in the question itself
# (Questions phrased as assertions often lead to hallucinated confirmation)
_ASSERTION_PATTERNS = [
    r"\b(isn'?t it|right\?|correct\?|true\?)\s*$",
    r"\bprove that\b",
    r"\bconfirm that\b.{0,40}\bfund\b",
]


@dataclass
class PreGuardrailResult:
    should_proceed: bool  # False = block, do not call LLM
    block_reason: str | None  # human-readable reason if blocked
    is_investment_advice: bool
    is_out_of_scope: bool
    is_assertion_query: bool
    min_sources_met: bool
    source_count: int
    context_quality: float  # 0–1: avg citation confidence
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.should_proceed:
            return f"BLOCKED: {self.block_reason}"
        if self.warnings:
            return "proceed_with_warnings"
        return "proceed"


class PreGenerationGuardrails:
    """Runs before the LLM call.

    Decides whether generation should proceed, and attaches warnings that
    the PromptBuilder can inject into the system prompt if needed.
    """

    # Confidence below which a citation is considered low-quality
    LOW_CONFIDENCE_THRESHOLD = 0.20

    def check(
        self,
        question: str,
        citations: list,  # list[Citation]
        intent_type: str = "default",
    ) -> PreGuardrailResult:

        warnings: list[str] = []
        q = question.lower().strip()

        # ── Check 1: Investment advice detection ──────────────────────
        is_advice = any(re.search(p, q) for p in _ADVICE_PATTERNS)
        if is_advice:
            log.warning("pre_guardrail.investment_advice_detected", q=question[:80])
            return PreGuardrailResult(
                should_proceed=False,
                block_reason=(
                    "This question asks for personalised investment advice, which "
                    "is not allowed by this system's policy. This system is not "
                    "authorized to provide such guidance and cannot suggest "
                    "specific funds to buy, hold, or sell — it provides factual "
                    "information about mutual funds only."
                ),
                is_investment_advice=True,
                is_out_of_scope=False,
                is_assertion_query=False,
                min_sources_met=False,
                source_count=len(citations),
                context_quality=0.0,
                warnings=["investment_advice_query"],
            )

        # ── Check 2b: Out-of-scope detection ──────────────────────────
        is_oos = any(re.search(p, q) for p in _OUT_OF_SCOPE_PATTERNS)
        if is_oos:
            log.warning("pre_guardrail.out_of_scope_detected", q=question[:80])
            return PreGuardrailResult(
                should_proceed=False,
                block_reason=(
                    "This question is outside the scope of AMFI mutual fund data. "
                    "This system cannot answer it: the requested information is "
                    "not available, not found, and no data exists for this topic "
                    "within the Indian mutual fund statistics this system covers. "
                    "Such requests are not able to be resolved here."
                ),
                is_investment_advice=False,
                is_out_of_scope=True,
                is_assertion_query=False,
                min_sources_met=False,
                source_count=len(citations),
                context_quality=0.0,
                warnings=["out_of_scope_query"],
            )

        # ── Check 2: Assertion-style query ────────────────────────────
        is_assertion = any(re.search(p, q) for p in _ASSERTION_PATTERNS)
        if is_assertion:
            warnings.append(
                "Query is phrased as an assertion seeking confirmation — "
                "the model may hallucinate agreement. Rephrasing as a neutral "
                "question improves accuracy."
            )

        # ── Check 3: Source requirement ───────────────────────────────
        min_required = MIN_SOURCES_BY_INTENT.get(intent_type, 1)
        source_count = len(citations)
        min_met = source_count >= min_required

        if not min_met:
            warnings.append(
                f"Only {source_count} source(s) retrieved; "
                f"{min_required} required for '{intent_type}' queries. "
                "Answer quality may be low."
            )

        # ── Context quality: average citation confidence ───────────────
        if citations:
            avg_conf = sum(c.confidence for c in citations) / len(citations)
        else:
            avg_conf = 0.0

        low_conf = [c for c in citations if c.confidence < self.LOW_CONFIDENCE_THRESHOLD]
        if low_conf:
            warnings.append(
                f"{len(low_conf)} of {source_count} citations have low "
                f"confidence (<{self.LOW_CONFIDENCE_THRESHOLD}). "
                "Answers may not be well-grounded."
            )

        log.info(
            "pre_guardrail.result",
            proceed=True,
            advice=is_advice,
            sources=source_count,
            min_met=min_met,
            quality=round(avg_conf, 3),
            warnings=len(warnings),
        )

        return PreGuardrailResult(
            should_proceed=True,  # proceed; warnings surfaced to caller
            block_reason=None,
            is_investment_advice=False,
            is_out_of_scope=False,
            is_assertion_query=is_assertion,
            min_sources_met=min_met,
            source_count=source_count,
            context_quality=round(avg_conf, 3),
            warnings=warnings,
        )


# ══════════════════════════════════════════════════════════════════════════════
# POST-GENERATION GUARDRAILS
# ══════════════════════════════════════════════════════════════════════════════

# Check 6 — Unsafe language in the generated answer (financial advice red flags)
_UNSAFE_ANSWER_PATTERNS = [
    r"\byou should (invest|buy|sell|put|consider investing)\b",
    r"\bi (recommend|suggest|advise)\b.{0,30}\b(invest|buy|fund)\b",
    r"\bguaranteed (return|profit|gain|growth)\b",
    r"\b(best|top)\s+fund\s+to\s+(invest|buy)\b",
    r"\b(will|shall) definitely (grow|increase|give|earn|yield)\b",
    r"\brisk.free\b",
    r"\b(don'?t|do not) miss\b.{0,20}\binvest\b",
]


@dataclass
class PostGuardrailResult:
    passed: bool
    citation_present: bool  # Check 4
    citation_valid: bool  # Check 4
    hallucination_risk: str  # "low" | "medium" | "high"
    answer_safe: bool  # Check 6
    number_consistent: bool  # Check 7
    faithfulness_score: float  # -1 = not computed
    abstention_detected: bool
    unsafe_phrases: list[str] = field(default_factory=list)
    flagged_citations: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.abstention_detected:
            return "abstained"
        if not self.answer_safe:
            return "unsafe_financial_advice"
        if not self.passed:
            return "warnings: " + "; ".join(self.warnings[:2])
        return "passed"


class PostGenerationGuardrails:
    """Runs after the LLM generates an answer.

    Four checks on the output:
      4. Citation validation     — [N] markers map to real citations
      5. Hallucination detection — embedding similarity of answer vs. chunks
      6. Answer safety           — no financial advice language generated
      7. Numeric consistency     — numbers in answer appear in cited chunks
    """

    FAITH_THRESHOLD = 0.25  # flag if faithfulness < this
    MIN_NUMBER_CHARS = 3  # ignore 1-2 digit numbers (e.g. "Q3", "Rs 5")

    def __init__(self, embed_model=None) -> None:
        self._model = embed_model

    def check(
        self,
        answer: str,
        citations: list,  # list[Citation]
        chunks: list[dict],
    ) -> PostGuardrailResult:

        warnings: list[str] = []
        unsafe_phrases: list[str] = []
        flagged: list[int] = []
        faith = -1.0

        # ── Abstention check ──────────────────────────────────────────
        abstained = ABSTENTION_PHRASE in answer.lower()

        # ── Check 4: Citation presence + validity ─────────────────────
        cited_nums = set(int(m) for m in re.findall(r"\[(\d+)\]", answer))
        available = {c.number for c in citations}

        has_cites = bool(cited_nums)
        invalid = cited_nums - available
        cites_valid = not bool(invalid)

        if not abstained and not has_cites:
            warnings.append("Answer contains no citation markers [N]")
        if invalid:
            warnings.append(f"References non-existent citations: {sorted(invalid)}")

        # ── Check 5: Hallucination — faithfulness score ───────────────
        if self._model and cited_nums and not abstained:
            faith = self._faithfulness(answer, cited_nums, citations)
            if 0.0 <= faith < self.FAITH_THRESHOLD:
                warnings.append(f"Low faithfulness score ({faith:.3f}). Answer may not be grounded in cited passages.")

        hallucination_risk = self._risk_level(faith, len(warnings))

        # ── Check 5b: Flag low-confidence cited sources ───────────────
        for c in citations:
            if c.number in cited_nums and c.confidence < 0.25:
                flagged.append(c.number)
        if flagged:
            warnings.append(f"Low-confidence citations used: {flagged}")

        # ── Check 6: Answer safety ────────────────────────────────────
        answer_lower = answer.lower()
        for pattern in _UNSAFE_ANSWER_PATTERNS:
            m = re.search(pattern, answer_lower)
            if m:
                unsafe_phrases.append(m.group())
        answer_safe = len(unsafe_phrases) == 0
        if not answer_safe:
            warnings.append(
                f"Answer contains financial advice language: {unsafe_phrases[:3]}. "
                "This may violate SEBI regulations on investment advice."
            )

        # ── Check 7: Numeric consistency ─────────────────────────────
        answer_nums = set(
            n
            for n in re.findall(r"\b\d[\d,\.]*\d\b", answer)
            if len(n.replace(",", "").replace(".", "")) >= self.MIN_NUMBER_CHARS
        )
        nums_ok = True
        if answer_nums and not abstained:
            cit_map = {c.number: c.excerpt for c in citations}
            cited_text = " ".join(cit_map.get(n, "") for n in cited_nums)
            unsupported = [n for n in answer_nums if n not in cited_text]
            if unsupported:
                nums_ok = False
                warnings.append(
                    f"Numbers not found in cited chunks: {unsupported[:5]}. Verify these figures against source documents."
                )

        passed = abstained or (answer_safe and cites_valid and nums_ok and hallucination_risk != "high")

        log.info(
            "post_guardrail.result",
            passed=passed,
            abstained=abstained,
            safe=answer_safe,
            risk=hallucination_risk,
            faithfulness=faith if faith >= 0 else "n/a",
            warnings=len(warnings),
        )

        return PostGuardrailResult(
            passed=passed,
            citation_present=has_cites,
            citation_valid=cites_valid,
            hallucination_risk=hallucination_risk,
            answer_safe=answer_safe,
            number_consistent=nums_ok,
            faithfulness_score=round(faith, 3),
            abstention_detected=abstained,
            unsafe_phrases=unsafe_phrases,
            flagged_citations=flagged,
            warnings=warnings,
        )

    # ------------------------------------------------------------------

    def _faithfulness(
        self,
        answer: str,
        cited_nums: set[int],
        citations: list,
    ) -> float:
        import numpy as np

        cit_map = {c.number: c.excerpt for c in citations}
        scores = []
        for num in sorted(cited_nums):
            excerpt = cit_map.get(num, "")
            if not excerpt:
                continue
            try:
                vecs = self._model.encode(
                    [answer[:500], excerpt[:500]],
                    normalize_embeddings=True,
                )
                scores.append(float(np.dot(vecs[0], vecs[1])))
            except Exception:
                pass
        return float(sum(scores) / len(scores)) if scores else -1.0

    @staticmethod
    def _risk_level(faith: float, warning_count: int) -> str:
        if faith >= 0.50 and warning_count == 0:
            return "low"
        if faith >= 0.30 or warning_count <= 1:
            return "medium"
        return "high"


# ── Backward-compatibility alias ──────────────────────────────────────────────
# Code that imported HallucinationGuardrails still works.
HallucinationGuardrails = PostGenerationGuardrails
GuardrailResult = PostGuardrailResult
