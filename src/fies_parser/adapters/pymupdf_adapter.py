"""PyMuPDF adapter — fast text/layout extraction for `application/pdf`.

Table extraction is intentionally out of scope for this milestone; every
candidate returned here has `tables=[]`. No financial normalization, heading
classification, or chunking happens here either — this adapter's only job is
converting PyMuPDF's output into common candidate models.
"""

from __future__ import annotations

import pymupdf

from fies_parser.adapters.base import ParserAdapter
from fies_parser.adapters.capabilities import ParserCapabilities
from fies_parser.canonical.candidate_models import (
    BoundingBox,
    CandidateElement,
    CandidatePage,
    ElementType,
    ParserCandidate,
    ParserExecutionMetrics,
)
from fies_parser.engine.exceptions import InvalidPageSelectionError
from fies_parser.engine.models import ParseRequest

_TEXT_BLOCK_TYPE = 0


class PyMuPDFAdapter(ParserAdapter):
    """Adapter over PyMuPDF. `financial_pipeline.processing.extractor`'s main
    PDF extraction path is wired through this adapter (see
    `parser_engine_integration.py`); its `_has_text_layer()` pre-parse
    heuristic still calls PyMuPDF directly, since that's a sampling check, not
    "parsing" in the `ParserAdapter` sense."""

    name = "pymupdf"

    def __init__(self) -> None:
        self.version = pymupdf.__version__

    def supports(self, request: ParseRequest) -> bool:
        return request.document.mime_type == "application/pdf"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            supports_page_selection=True,
            supports_tables=False,
            supports_ocr=False,
            supports_headings=False,
            supported_mime_types=("application/pdf",),
        )

    def parse(self, request: ParseRequest) -> ParserCandidate:
        document = request.document

        with pymupdf.open(document.file_path) as doc:
            page_count = len(doc)
            page_numbers = self._resolve_page_numbers(request.pages, page_count)
            document_metadata = dict(doc.metadata or {})

            pages: list[CandidatePage] = []
            elements: list[CandidateElement] = []

            for page_number in page_numbers:
                page = doc[page_number - 1]
                page_elements = self._extract_elements(document.document_id, page_number, page)
                elements.extend(page_elements)

                pages.append(
                    CandidatePage(
                        page_number=page_number,
                        width=page.rect.width,
                        height=page.rect.height,
                        rotation=page.rotation,
                        text=page.get_text("text").strip() or None,
                        element_ids=[el.element_id for el in page_elements],
                        table_ids=[],
                    )
                )

        metrics = ParserExecutionMetrics(
            pages_requested=len(page_numbers),
            pages_processed=len(pages),
        )

        return ParserCandidate(
            document_id=document.document_id,
            parser_name=self.name,
            parser_version=self.version,
            pages=pages,
            elements=elements,
            tables=[],
            metrics=metrics,
            document_metadata=document_metadata,
        )

    def _resolve_page_numbers(self, requested: tuple[int, ...] | None, page_count: int) -> list[int]:
        if requested is None:
            return list(range(1, page_count + 1))

        # ParseRequest's field validator already guarantees `requested` is
        # deduplicated and sorted before it reaches the adapter.
        out_of_range = tuple(p for p in requested if p > page_count)
        if out_of_range:
            raise InvalidPageSelectionError(f"document has {page_count} page(s)", out_of_range)
        return list(requested)

    def _extract_elements(self, document_id: str, page_number: int, page: pymupdf.Page) -> list[CandidateElement]:
        # PyMuPDF's block order is not guaranteed to be reading order; sort
        # top-to-bottom, then left-to-right as a first-pass approximation.
        blocks = sorted(page.get_text("blocks"), key=lambda b: (round(b[1], 1), round(b[0], 1)))

        elements: list[CandidateElement] = []
        for index, block in enumerate(blocks):
            x0, y0, x1, y1, text, block_no, block_type = block[:7]
            if block_type != _TEXT_BLOCK_TYPE:
                continue

            text = text.strip()
            if not text:
                continue

            elements.append(
                CandidateElement(
                    element_id=f"{document_id}-p{page_number}-b{index}",
                    element_type=ElementType.PARAGRAPH,
                    page_number=page_number,
                    text=text,
                    bbox=BoundingBox(x0=x0, y0=y0, x1=x1, y1=y1),
                    raw_payload={"block_no": block_no, "block_type": block_type},
                )
            )

        return elements
