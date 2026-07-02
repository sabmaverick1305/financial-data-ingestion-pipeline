"""ContextBuilder — formats retrieved chunks into LLM-ready context.

Produces:
  - A structured system prompt describing the knowledge base
  - A context block with numbered citations
  - OpenAI-compatible message list for direct use with the chat API
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are an expert financial analyst specialising in Indian mutual funds.
You answer questions using ONLY the context passages provided below.
Each passage is labelled [1], [2], … and comes from an AMFI (Association of
Mutual Funds in India) official document.

Rules:
- Cite every claim with its passage number, e.g. "Total AUM grew 41% [1]."
- If the answer is not in the context, say "I don't have that information in
  the provided documents."
- Use precise numbers when the context includes them.
- Do not hallucinate fund names, NAVs, or AUM figures.
"""


class ContextBuilder:
    """Converts a list of retrieved chunk dicts into LLM-ready text."""

    MAX_CHARS_PER_CHUNK = 800  # truncate very long chunks to keep context tight

    def build_context_block(self, chunks: list[dict]) -> str:
        """Return a numbered context block ready to embed in a prompt."""
        if not chunks:
            return "No relevant passages found."

        parts: list[str] = []
        for i, chunk in enumerate(chunks, 1):
            text = (chunk.get("text") or "").strip()
            if len(text) > self.MAX_CHARS_PER_CHUNK:
                text = text[: self.MAX_CHARS_PER_CHUNK] + "…"

            meta = self._meta_line(chunk)
            parts.append(f"[{i}] {meta}\n{text}")

        return "\n\n".join(parts)

    def build_messages(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[dict]:
        """Build an OpenAI-compatible messages list (system + user turn)."""
        context = self.build_context_block(chunks)
        user_content = f"Context passages:\n\n{context}\n\nQuestion: {query}"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def format_sources(self, chunks: list[dict]) -> list[dict]:
        """Return a clean source list suitable for API responses."""
        sources = []
        for i, chunk in enumerate(chunks, 1):
            sources.append(
                {
                    "citation": i,
                    "file_name": chunk.get("file_name"),
                    "period_year": chunk.get("period_year"),
                    "period_month": chunk.get("period_month"),
                    "category": chunk.get("category"),
                    "chunk_index": chunk.get("chunk_index"),
                    "similarity": chunk.get("similarity"),
                    "rrf_score": chunk.get("rrf_score"),
                    "preview": (chunk.get("text") or "")[:200].strip() + "…",
                }
            )
        return sources

    # ------------------------------------------------------------------

    @staticmethod
    def _meta_line(chunk: dict) -> str:
        parts = [chunk.get("file_name", "unknown")]
        y, m = chunk.get("period_year"), chunk.get("period_month")
        if y and m:
            parts.append(f"{y}/{m:02d}")
        elif y:
            parts.append(str(y))
        cat = chunk.get("category")
        if cat:
            parts.append(cat)
        return "  |  ".join(parts)
