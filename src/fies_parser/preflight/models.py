"""Parser-independent output of the preflight stage.

`DocumentProfile`/`PageProfile` are what crosses the preflight -> routing
boundary — cheap signals a `RoutingPolicy` can decide on without paying for a
full parse. Nothing PDF-specific (no PyMuPDF types) appears here; the
profilers that produce these models may depend on a specific library
internally, but what they return does not.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class PageProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    page_number: int
    has_text: bool
    word_count: int
    image_count: int = 0
    # Cheap, non-ML layout signals (PyMuPDF's built-in table finder / vector
    # drawing count) — not a substitute for Docling's real layout model, just
    # enough for routing to prefer a table-capable parser when it matters.
    table_count: int = 0
    drawing_count: int = 0


class DocumentProfile(BaseModel):
    model_config = ConfigDict(frozen=True)

    document_id: str
    mime_type: str
    page_count: int
    file_size_bytes: int
    has_text_layer: bool
    text_page_ratio: float
    # True if any *sampled* page looks like it contains a table. A cheap,
    # possibly-incomplete signal (only sampled pages are checked) — routing
    # treats it as "probably has tables", not a guarantee.
    likely_has_tables: bool = False
    pages: tuple[PageProfile, ...] = Field(default_factory=tuple)
