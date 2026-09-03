"""Wires `TextExtractor`'s PDF path through the Parser Engine.

`fies_parser` knows nothing about `ExtractResult` — it's a
`financial_pipeline`-specific shape that predates the engine and that
chunking/table workers still depend on. This module is the one place that
bridges the two: it owns the `ParserRegistry`/`ParserEngine` instance, spills
in-memory PDF bytes to a temp file (the engine's `SourceDocument` requires a
real file path), runs the `pymupdf` adapter, and maps the resulting
`ParserCandidate` back into `ExtractResult` via `LegacyExtractResultMapper`.

`TextExtractor._has_text_layer()` intentionally still calls PyMuPDF directly
— it's a lightweight pre-parse sampling heuristic, not "parsing" in the
`ParserAdapter` sense, and folding it into the engine contract would require
widening that contract for a milestone-1 concern it was never meant to cover.

Also fires shadow-routing execution (`ShadowRouter`, see
`parser_composition_root.py`) alongside the real extraction, controlled by
`Settings.parser_shadow_routing_sample_rate`/`_execute` — off by default. This
is observation only: PyMuPDF's result is always what gets returned, and a
shadow-routing failure can never affect it.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from fies_parser.adapters.pymupdf_adapter import PyMuPDFAdapter
from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.engine.models import ParseRequest, SourceDocument
from fies_parser.engine.parser_engine import ParserEngine
from fies_parser.engine.registry import ParserRegistry
from fies_parser.mappers.base import CandidateMapper
from financial_pipeline.config import settings
from financial_pipeline.processing.extractor import ExtractResult
from financial_pipeline.processing.parser_composition_root import build_shadow_router

_registry = ParserRegistry()
_registry.register(PyMuPDFAdapter())
_engine = ParserEngine(_registry)
_shadow_router = build_shadow_router(
    sample_rate=settings.parser_shadow_routing_sample_rate,
    execute=settings.parser_shadow_routing_execute,
)


class LegacyExtractResultMapper(CandidateMapper[ExtractResult]):
    """Maps a `ParserCandidate` (pymupdf) into the legacy `ExtractResult` shape
    that `chunker.py` and the table/embed workers are built around."""

    def __init__(self, has_text_layer: bool) -> None:
        self._has_text_layer = has_text_layer

    def map(self, candidate: ParserCandidate) -> ExtractResult:
        pages = [{"page": page.page_number, "text": page.text or ""} for page in candidate.pages]
        full_text = "\n\n".join(str(page["text"]) for page in pages if page["text"])

        return ExtractResult(
            pages=pages,
            tables=[],
            full_text=full_text,
            markdown=full_text,
            metadata=candidate.document_metadata,
            has_text_layer=self._has_text_layer,
            figures=[],
            extraction_engine=candidate.parser_name,
            warnings=list(candidate.warnings),
        )


def extract_pdf_via_parser_engine(raw: bytes, has_text_layer: bool) -> ExtractResult:
    """Runs the `pymupdf` adapter through `ParserEngine` on in-memory PDF bytes
    and returns the result in the legacy `ExtractResult` shape."""
    file_hash = hashlib.sha256(raw).hexdigest()

    with tempfile.TemporaryDirectory(prefix="fies_parser_") as tmp_dir:
        file_path = Path(tmp_dir) / "document.pdf"
        file_path.write_bytes(raw)

        document = SourceDocument(
            document_id=file_hash,
            file_path=file_path,
            file_name="document.pdf",
            mime_type="application/pdf",
            file_hash=file_hash,
            source="legacy_text_extractor",
        )
        request = ParseRequest(document=document)
        candidate = _engine.run("pymupdf", request)

        # Must run before the temp file goes away with this `with` block.
        _shadow_router.maybe_shadow(request, authoritative_parser_name="pymupdf")

    return LegacyExtractResultMapper(has_text_layer=has_text_layer).map(candidate)
