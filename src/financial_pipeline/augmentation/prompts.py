"""Prompt Builder — Stage 3 of the augmentation layer.

Intent-aware prompt construction. Different question types need different
framing to get the best answers from the LLM:

  factual      → precise, cite numbers exactly
  regulatory   → list rules clearly, note effective dates
  trend        → describe direction + magnitude over time
  comparison   → structured comparison, use tables if possible
  definition   → clear explanation, no jargon
  lookup       → confirm which document, give coordinates
  tabular      → prefer structured output (markdown table)
"""
from __future__ import annotations

from financial_pipeline.augmentation.citations import Citation

# ── System prompts by intent ──────────────────────────────────────────────────

_BASE_RULES = """\
Rules:
- Answer using ONLY the numbered context passages below. Never invent data.
- Cite every factual claim with its passage number: "AUM reached ₹40 lakh crore [2]."
- If the answer is not in the passages, say exactly: \
"This information is not available in the provided AMFI documents."
- Never guess, extrapolate, or fill gaps with general knowledge.
- Use exact figures when the context provides them.
"""

SYSTEM_PROMPTS: dict[str, str] = {

    "factual": f"""\
You are a precise financial data analyst specialising in Indian mutual funds.
{_BASE_RULES}
Format: Answer in 2–4 concise sentences. Lead with the key figure or fact.""",

    "regulatory": f"""\
You are a mutual fund regulatory compliance expert.
{_BASE_RULES}
Format: Use a numbered list for each distinct regulatory change.
Include the SEBI circular reference or date if mentioned in the passages.
Note whether the change is a new requirement, amendment, or clarification.""",

    "trend": f"""\
You are a financial trend analyst covering the Indian mutual fund industry.
{_BASE_RULES}
Format: Describe the direction (growth/decline), magnitude (%), and \
time period clearly. If multiple data points are available, mention the \
earliest and most recent values to show the trajectory.""",

    "comparison": f"""\
You are a comparative financial analyst.
{_BASE_RULES}
Format: Structure your answer as a comparison — use "vs." or a brief \
markdown table if two or more categories are being compared.
Highlight the key difference in one final sentence.""",

    "definition": f"""\
You are a financial education expert explaining mutual fund concepts.
{_BASE_RULES}
Format: Define the term clearly in plain language (no jargon).
If the passages list fund names rather than definitions, say so honestly.""",

    "lookup": f"""\
You are a document librarian for AMFI financial documents.
{_BASE_RULES}
Format: Confirm the document name, period, and file type.
Provide the S3 path if present in the context.""",

    "tabular": f"""\
You are a financial data extraction specialist.
{_BASE_RULES}
Format: Present the data as a markdown table when possible.
Include column headers from the source. State total/subtotal rows explicitly.""",

    "default": f"""\
You are a knowledgeable financial analyst specialising in Indian mutual funds.
{_BASE_RULES}
Format: Answer clearly and concisely. Cite every claim.""",
}

# ── Few-shot examples (appended for low-confidence intents) ──────────────────

FEW_SHOT: dict[str, str] = {
    "regulatory": """\
Example:
Question: What did SEBI say about derivatives?
Answer: SEBI issued guidelines permitting mutual funds to trade interest rate \
derivatives through stock exchanges for hedging purposes, subject to appropriate \
disclosures in offer documents [3]. A separate circular clarified the use of \
derivatives for portfolio rebalancing [5].
""",
    "factual": """\
Example:
Question: What was the AUM in March 2020?
Answer: The total AUM of the mutual fund industry as of March 2020 was \
₹22.26 lakh crore, representing a decline from the previous month due to \
market corrections [2].
""",
}


class PromptBuilder:
    """Builds LLM-ready message lists from retrieved chunks and intent."""

    MAX_CONTEXT_CHARS = 7000    # stay well within 8 K token context
    MAX_CHUNK_CHARS   = 600

    def build(
        self,
        question:  str,
        chunks:    list[dict],
        citations: list[Citation],
        intent_type: str = "default",
        add_few_shot: bool = True,
    ) -> list[dict]:
        """Return an OpenAI / Anthropic compatible messages list."""
        system   = self._system(intent_type, add_few_shot)
        context  = self._context_block(chunks, citations)
        user_msg = f"Context passages:\n\n{context}\n\nQuestion: {question}"

        return [
            {"role": "system", "content": system},
            {"role": "user",   "content": user_msg},
        ]

    def estimate_tokens(self, messages: list[dict]) -> int:
        """Rough token estimate (1 token ≈ 4 chars)."""
        total = sum(len(m["content"]) for m in messages)
        return total // 4

    # ------------------------------------------------------------------

    def _system(self, intent: str, add_few_shot: bool) -> str:
        base = SYSTEM_PROMPTS.get(intent, SYSTEM_PROMPTS["default"])
        if add_few_shot and intent in FEW_SHOT:
            base += f"\n\n{FEW_SHOT[intent]}"
        return base

    def _context_block(
        self,
        chunks:    list[dict],
        citations: list[Citation],
    ) -> str:
        parts:      list[str] = []
        total_chars = 0

        for c in citations:
            text = c.excerpt.strip()
            if len(text) > self.MAX_CHUNK_CHARS:
                text = text[: self.MAX_CHUNK_CHARS] + "…"

            ref  = c.reference_string()
            part = f"[{c.number}] {ref}\n{text}"

            if total_chars + len(part) > self.MAX_CONTEXT_CHARS:
                break

            parts.append(part)
            total_chars += len(part)

        return "\n\n".join(parts) if parts else "No relevant passages found."
