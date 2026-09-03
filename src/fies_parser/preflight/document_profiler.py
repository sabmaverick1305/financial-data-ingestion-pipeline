"""Cheap, pre-parse profiling of a document — used by routing to pick an
adapter without paying for a full parse.

Uses PyMuPDF directly for PDF introspection (page count, text-layer
sampling), the same carve-out
`financial_pipeline.processing.extractor.TextExtractor._has_text_layer`
already relies on: this is sniffing, not "parsing" in the `ParserAdapter`
sense, so it sits outside the adapter contract and is not routed through
`ParserEngine`. Sampling thresholds default to the values that heuristic has
run in production with.
"""

from __future__ import annotations

from fies_parser.engine.exceptions import UnsupportedDocumentError
from fies_parser.engine.models import SourceDocument
from fies_parser.preflight.models import DocumentProfile, PageProfile
from fies_parser.preflight.page_profiler import PageProfiler

DEFAULT_SAMPLE_PAGES = 20
DEFAULT_TEXT_PAGE_RATIO = 0.4


class DocumentProfiler:
    def __init__(
        self,
        page_profiler: PageProfiler | None = None,
        sample_pages: int = DEFAULT_SAMPLE_PAGES,
        text_page_ratio_threshold: float = DEFAULT_TEXT_PAGE_RATIO,
    ) -> None:
        self._page_profiler = page_profiler or PageProfiler()
        self._sample_pages = sample_pages
        self._text_page_ratio_threshold = text_page_ratio_threshold

    def profile(self, document: SourceDocument) -> DocumentProfile:
        if document.mime_type != "application/pdf":
            raise UnsupportedDocumentError(document.document_id, f"preflight does not support mime type {document.mime_type!r}")

        import pymupdf

        file_size_bytes = document.file_path.stat().st_size

        with pymupdf.open(document.file_path) as doc:
            page_count = len(doc)
            sample_indices = self._sample_page_indices(page_count)

            pages: list[PageProfile] = [self._page_profiler.profile(doc[index], index + 1) for index in sample_indices]

        text_pages = sum(1 for page in pages if page.has_text)
        text_page_ratio = text_pages / len(pages) if pages else 0.0

        return DocumentProfile(
            document_id=document.document_id,
            mime_type=document.mime_type,
            page_count=page_count,
            file_size_bytes=file_size_bytes,
            has_text_layer=text_page_ratio >= self._text_page_ratio_threshold,
            text_page_ratio=text_page_ratio,
            likely_has_tables=any(page.table_count > 0 for page in pages),
            pages=tuple(pages),
        )

    def _sample_page_indices(self, page_count: int) -> list[int]:
        if page_count == 0:
            return []
        if page_count <= self._sample_pages:
            return list(range(page_count))
        step = page_count // self._sample_pages
        return list(range(0, page_count, step))[: self._sample_pages]
