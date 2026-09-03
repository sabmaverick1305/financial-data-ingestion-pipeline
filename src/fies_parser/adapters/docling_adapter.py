"""Docling adapter — layout-aware parsing with reading order, hierarchy, and
table structure for `application/pdf`.

Owns the only `import docling` in this codebase outside of
`financial_pipeline.processing.extractor` (pre-existing, not yet migrated —
see migration notes). No financial normalization or chunking happens here;
table cell values are preserved exactly as Docling extracted them.
"""

from __future__ import annotations

from importlib.metadata import version as _package_version
from pathlib import Path
from typing import Any

from fies_parser.adapters.base import ParserAdapter
from fies_parser.adapters.capabilities import ParserCapabilities
from fies_parser.canonical.candidate_models import (
    BoundingBox,
    CandidateElement,
    CandidatePage,
    CandidateTable,
    ElementType,
    ParserCandidate,
    ParserExecutionMetrics,
)
from fies_parser.engine.exceptions import InvalidPageSelectionError
from fies_parser.engine.models import ParseRequest

_LABEL_TO_ELEMENT_TYPE = {
    "title": ElementType.TITLE,
    "section_header": ElementType.HEADING,
    "text": ElementType.PARAGRAPH,
    "paragraph": ElementType.PARAGRAPH,
    "list_item": ElementType.LIST,
    "footnote": ElementType.FOOTNOTE,
    "picture": ElementType.IMAGE,
    "caption": ElementType.CAPTION,
    "formula": ElementType.FORMULA,
}


class DoclingAdapter(ParserAdapter):
    """Adapter over Docling. `ocr` mirrors `extractor.py`'s `TableExtractor`
    (ocr=False) / `OcrExtractor` (ocr=True) split — construct two instances
    and register them under different names if both are needed."""

    name = "docling"

    def __init__(self, *, ocr: bool = False, num_threads: int = 1) -> None:
        self._ocr = ocr
        self._num_threads = num_threads
        self._converter: Any | None = None
        try:
            self.version = _package_version("docling")
        except Exception:
            self.version = "unknown"

    def supports(self, request: ParseRequest) -> bool:
        return request.document.mime_type == "application/pdf"

    @property
    def capabilities(self) -> ParserCapabilities:
        return ParserCapabilities(
            supports_page_selection=True,
            supports_tables=True,
            supports_ocr=self._ocr,
            supports_headings=True,
            supported_mime_types=("application/pdf",),
        )

    def parse(self, request: ParseRequest) -> ParserCandidate:
        document = request.document

        result = self._convert(document.file_path, request.pages)
        doc = result.document

        page_numbers = self._resolve_page_numbers(request.pages, doc)
        elements = self._build_elements(document.document_id, doc, page_numbers)
        tables = self._build_tables(document.document_id, doc, page_numbers)
        pages = self._build_pages(doc, page_numbers, elements, tables)
        document_metadata = self._extract_metadata(doc)

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
            tables=tables,
            metrics=metrics,
            document_metadata=document_metadata,
        )

    def _convert(self, file_path: Path, requested_pages: tuple[int, ...] | None) -> Any:
        converter = self._get_converter()
        if requested_pages:
            # Docling's `page_range` is a contiguous inclusive window; a
            # non-contiguous request converts the spanning range and is
            # filtered down to the exact pages afterward.
            return converter.convert(file_path, page_range=(min(requested_pages), max(requested_pages)))
        return converter.convert(file_path)

    def _get_converter(self) -> Any:
        if self._converter is None:
            from docling.datamodel.pipeline_options import (
                AcceleratorDevice,
                AcceleratorOptions,
                PdfPipelineOptions,
            )
            from docling.document_converter import DocumentConverter, PdfFormatOption

            accel = AcceleratorOptions(num_threads=self._num_threads, device=AcceleratorDevice.CPU)
            opts = PdfPipelineOptions()
            opts.do_ocr = self._ocr
            opts.do_table_structure = True
            opts.accelerator_options = accel
            self._converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})
        return self._converter

    def _resolve_page_numbers(self, requested: tuple[int, ...] | None, doc: Any) -> set[int]:
        available = set(doc.pages.keys())
        if requested is None:
            return available

        out_of_range = tuple(p for p in requested if p not in available)
        if out_of_range:
            raise InvalidPageSelectionError(f"document has pages {sorted(available)}", out_of_range)
        return set(requested)

    def _to_bounding_box(self, doc: Any, page_no: int | None, docling_bbox: Any) -> BoundingBox | None:
        if docling_bbox is None or page_no is None:
            return None
        page = doc.pages.get(page_no)
        page_height = page.size.height if page is not None and getattr(page, "size", None) else None
        box = docling_bbox.to_top_left_origin(page_height) if page_height else docling_bbox
        return BoundingBox(x0=box.l, y0=box.t, x1=box.r, y1=box.b)

    def _build_elements(self, document_id: str, doc: Any, page_numbers: set[int]) -> list[CandidateElement]:
        kept: list[tuple[Any, int, str, Any]] = []
        for item, _level in doc.iterate_items():
            prov = getattr(item, "prov", None) or []
            if not prov:
                continue
            page_no = prov[0].page_no
            if page_no not in page_numbers:
                continue
            text = (getattr(item, "text", None) or "").strip()
            if not text:
                continue
            kept.append((item, page_no, text, prov[0].bbox))

        ref_to_id = {
            getattr(item, "self_ref", None): f"{document_id}-p{page_no}-e{index}"
            for index, (item, page_no, _text, _bbox) in enumerate(kept)
        }

        elements: list[CandidateElement] = []
        for index, (item, page_no, text, docling_bbox) in enumerate(kept):
            self_ref = getattr(item, "self_ref", None)
            parent_ref = getattr(getattr(item, "parent", None), "cref", None)
            label = getattr(item, "label", None)
            label_value = str(getattr(label, "value", "") or "")

            elements.append(
                CandidateElement(
                    element_id=ref_to_id[self_ref] if self_ref in ref_to_id else f"{document_id}-p{page_no}-e{index}",
                    element_type=_LABEL_TO_ELEMENT_TYPE.get(label_value, ElementType.UNKNOWN),
                    page_number=page_no,
                    text=text,
                    bbox=self._to_bounding_box(doc, page_no, docling_bbox),
                    parent_element_id=ref_to_id.get(parent_ref),
                    raw_payload={"label": label_value},
                )
            )
        return elements

    def _build_tables(self, document_id: str, doc: Any, page_numbers: set[int]) -> list[CandidateTable]:
        tables: list[CandidateTable] = []
        for index, tbl in enumerate(getattr(doc, "tables", []), start=1):
            try:
                df = tbl.export_to_dataframe(doc=doc)
            except Exception:
                continue
            if df is None or df.empty:
                continue

            prov = getattr(tbl, "prov", None) or []
            page_no = prov[0].page_no if prov else None
            if page_no is not None and page_no not in page_numbers:
                continue

            caption = None
            captions = getattr(tbl, "captions", [])
            if captions:
                caption = str(getattr(captions[0], "text", "") or "").strip() or None

            tables.append(
                CandidateTable(
                    table_id=f"{document_id}-t{index}",
                    page_start=page_no or 1,
                    page_end=page_no or 1,
                    headers=[str(column) for column in df.columns],
                    rows=[[str(value) for value in row] for row in df.itertuples(index=False)],
                    title=caption,
                    bbox=self._to_bounding_box(doc, page_no, prov[0].bbox) if prov else None,
                )
            )
        return tables

    def _build_pages(
        self, doc: Any, page_numbers: set[int], elements: list[CandidateElement], tables: list[CandidateTable]
    ) -> list[CandidatePage]:
        pages: list[CandidatePage] = []
        for page_no in sorted(page_numbers):
            page_item = doc.pages.get(page_no)
            size = getattr(page_item, "size", None) if page_item is not None else None
            page_elements = [element for element in elements if element.page_number == page_no]
            page_tables = [table for table in tables if table.page_start <= page_no <= table.page_end]

            pages.append(
                CandidatePage(
                    page_number=page_no,
                    width=getattr(size, "width", None) if size else None,
                    height=getattr(size, "height", None) if size else None,
                    rotation=0,  # Docling's PageItem does not currently surface rotation.
                    text="\n".join(element.text for element in page_elements if element.text) or None,
                    element_ids=[element.element_id for element in page_elements],
                    table_ids=[table.table_id for table in page_tables],
                )
            )
        return pages

    def _extract_metadata(self, doc: Any) -> dict[str, Any]:
        metadata: dict[str, Any] = {}
        origin = getattr(doc, "origin", None)
        if origin is not None:
            metadata["filename"] = getattr(origin, "filename", None)
            metadata["mimetype"] = getattr(origin, "mimetype", None)
        name = getattr(doc, "name", None)
        if name:
            metadata["name"] = name
        return metadata
