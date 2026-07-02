"""Citation Formatter — Stage 2 of the augmentation layer.

Produces structured, verifiable citations from retrieved chunks.
Each citation carries:
  - Inline marker:  [1], [2], …
  - Full reference: file, period, category, page/chunk index
  - Excerpt:        the specific passage the claim should be grounded in
  - Confidence:     derived from the cross-encoder or retrieval score

Also provides post-generation citation verification — checks that each [N]
marker in the LLM's answer actually corresponds to a chunk that supports
the claim (grounding check via embedding similarity).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Citation:
    number:       int
    source_type:  str         # "chunk" | "table" | "document"
    file_name:    str | None
    period_year:  int | None
    period_month: int | None
    category:     str | None
    chunk_index:  int | None
    table_index:  int | None
    excerpt:      str         # the raw passage text (truncated)
    confidence:   float       # 0–1 relevance score
    rank_method:  str         # "cross_encoder" | "fallback"

    def reference_string(self) -> str:
        """Human-readable reference like APA-lite format."""
        parts = []
        if self.file_name:
            parts.append(self.file_name)
        if self.period_year and self.period_month:
            parts.append(f"{self.period_year}/{self.period_month:02d}")
        elif self.period_year:
            parts.append(str(self.period_year))
        if self.category:
            parts.append(self.category)
        return "  ·  ".join(parts) if parts else "unknown source"

    def inline_marker(self) -> str:
        return f"[{self.number}]"


@dataclass
class GroundingResult:
    """Result of checking whether the answer is grounded in citations."""
    is_grounded:       bool
    grounding_ratio:   float          # 0–1: fraction of cited claims verified
    unverified:        list[int]      # citation numbers flagged as low-confidence
    has_unsupported_numbers: bool     # numbers in answer not found in chunks
    faithfulness_scores: list[float]  # per-citation similarity scores
    warnings:          list[str] = field(default_factory=list)


class CitationFormatter:
    """Converts re-ranked chunks into structured Citation objects."""

    MAX_EXCERPT_CHARS = 500

    def format(self, chunks: list[dict]) -> list[Citation]:
        citations = []
        for i, chunk in enumerate(chunks, 1):
            text  = (chunk.get("text") or chunk.get("table_name") or "").strip()
            score = (
                chunk.get("_ce_score") or
                chunk.get("similarity") or
                chunk.get("rrf_score") or
                chunk.get("_mmr_score") or 0.0
            )
            # Normalise cross-encoder logits (approx range -10 to +10) to 0–1
            if chunk.get("_rank_method") == "cross_encoder":
                confidence = max(0.0, min(1.0, (float(score) + 10) / 20))
            else:
                confidence = max(0.0, min(1.0, float(score)))

            citations.append(Citation(
                number       = i,
                source_type  = chunk.get("_source", "chunk"),
                file_name    = chunk.get("file_name"),
                period_year  = chunk.get("period_year"),
                period_month = chunk.get("period_month"),
                category     = chunk.get("category"),
                chunk_index  = chunk.get("chunk_index"),
                table_index  = chunk.get("table_index"),
                excerpt      = text[: self.MAX_EXCERPT_CHARS],
                confidence   = round(confidence, 3),
                rank_method  = chunk.get("_rank_method", "fallback"),
            ))
        return citations

    def build_reference_list(self, citations: list[Citation]) -> str:
        """Formatted reference block appended to the context."""
        lines = ["References:"]
        for c in citations:
            conf_str = f"(confidence: {c.confidence:.2f})"
            lines.append(f"  [{c.number}] {c.reference_string()}  {conf_str}")
        return "\n".join(lines)

    def verify_grounding(
        self,
        answer:    str,
        citations: list[Citation],
        model=None,                 # sentence-transformers model, optional
    ) -> GroundingResult:
        """Check whether answer claims are grounded in retrieved citations.

        Performs three checks:
        1. Citation markers present ([1], [2], …)
        2. Number consistency: any number in the answer appears in its cited chunk
        3. Semantic faithfulness: embedding similarity between answer sentence
           and cited chunk (only when *model* is provided)
        """
        cited_numbers = set(int(m) for m in re.findall(r'\[(\d+)\]', answer))
        available     = {c.number for c in citations}
        unverified    = []
        faith_scores: list[float] = []

        # Check 1: every cited number exists
        invalid = cited_numbers - available
        warnings = []
        if invalid:
            warnings.append(f"Answer references non-existent citations: {sorted(invalid)}")

        # Check 2: low-confidence citations
        for c in citations:
            if c.number in cited_numbers and c.confidence < 0.30:
                unverified.append(c.number)

        # Check 3: number consistency
        answer_numbers = set(re.findall(r'\b\d[\d,\.]*\d\b', answer))
        has_unsupported_numbers = False
        if answer_numbers:
            cited_text = " ".join(
                c.excerpt for c in citations if c.number in cited_numbers
            )
            unsupported = [n for n in answer_numbers if n not in cited_text]
            if unsupported:
                has_unsupported_numbers = True
                warnings.append(
                    f"Numbers in answer not found in cited chunks: {unsupported[:5]}"
                )

        # Check 4: semantic faithfulness (optional, requires embedding model)
        if model is not None:
            cit_map = {c.number: c.excerpt for c in citations}
            for num in sorted(cited_numbers):
                excerpt = cit_map.get(num, "")
                if not excerpt:
                    continue
                try:
                    # Encode answer + chunk excerpt, compute cosine similarity
                    import numpy as np
                    vecs = model.encode([answer[:500], excerpt[:500]],
                                        normalize_embeddings=True)
                    sim  = float(np.dot(vecs[0], vecs[1]))
                    faith_scores.append(sim)
                    if sim < 0.30:
                        warnings.append(
                            f"[{num}] low faithfulness similarity: {sim:.3f}"
                        )
                except Exception:
                    pass

        grounding_ratio = (
            len(cited_numbers - set(unverified)) / max(len(cited_numbers), 1)
            if cited_numbers else 0.0
        )

        return GroundingResult(
            is_grounded            = (len(warnings) == 0 and grounding_ratio >= 0.5),
            grounding_ratio        = round(grounding_ratio, 3),
            unverified             = unverified,
            has_unsupported_numbers= has_unsupported_numbers,
            faithfulness_scores    = [round(s, 3) for s in faith_scores],
            warnings               = warnings,
        )
