from __future__ import annotations

import gc
import io
import os
import resource
import traceback
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import structlog

log = structlog.get_logger()

BLANK_PAGE_THRESHOLD = 30
MAX_EXCEL_PREVIEW_ROWS = 200
MAX_TABLE_ROWS_IN_MEMORY = 100_000


def _rss_mb() -> float:
    """
    Peak RSS memory in MB.
    Note: this is peak memory, not current memory.
    """
    import sys

    divisor = 1024 if sys.platform == "linux" else 1024 * 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def _dedupe_columns(columns) -> list[str]:
    """Rename repeated column labels (col, col.1, col.2, ...) so pyarrow's
    Parquet writer never raises "Duplicate column names found" — Docling's
    table export flattens multi-level/merged headers (e.g. "No. of Schemes"
    appearing under both an "Open-ended" and "Close-ended" group) into a
    single header row, which collides when the leaf labels repeat."""
    seen: dict[str, int] = {}
    result = []
    for col in columns:
        col = str(col)
        if col not in seen:
            seen[col] = 0
            result.append(col)
        else:
            seen[col] += 1
            result.append(f"{col}.{seen[col]}")
    return result


@dataclass
class TableMeta:
    """Per-table metadata alongside each extracted DataFrame."""

    page_number: int | None = None
    table_name: str | None = None  # caption or label from Docling
    section_title: str | None = None  # nearest heading (best-effort)


@dataclass
class ExtractResult:
    pages: list[dict]
    tables: list[pd.DataFrame]
    full_text: str
    markdown: str
    metadata: dict
    has_text_layer: bool
    figures: list[dict]
    warnings: list[str] = field(default_factory=list)
    extraction_engine: str = ""
    failed_tables: int = 0
    table_meta: list[TableMeta] = field(default_factory=list)  # parallel to tables


class DocumentExtractionError(RuntimeError):
    pass


# ──────────────────────────────────────────────────────────────────────────
# Stage 1 — text-worker (low memory: PyMuPDF + pandas, no ML models loaded)
# ──────────────────────────────────────────────────────────────────────────


class TextExtractor:
    """
    Low-memory worker. Always runs first.

    PDF      -> PyMuPDF page-wise text + has_text_layer detection.
    XLS/XLSX -> pandas (cheap; tables are produced here directly since no
                ML model is involved for spreadsheets).
    """

    def extract(self, raw: bytes, file_type: str) -> ExtractResult:
        if not raw:
            raise DocumentExtractionError("Empty file received")

        ft = file_type.lower().lstrip(".")

        log.info(
            "text_extractor.start",
            file_type=ft,
            size_mb=round(len(raw) / 1024 / 1024, 2),
            rss_mb=round(_rss_mb(), 1),
        )

        try:
            if ft == "pdf":
                has_text = self._has_text_layer(raw)
                return self._extract_pdf_with_pymupdf(raw, has_text)

            if ft in {"xls", "xlsx"}:
                return self._extract_spreadsheet(raw, ft)

            raise DocumentExtractionError(f"Unsupported file type: {file_type}")

        except Exception as exc:
            log.exception(
                "text_extractor.failed",
                file_type=ft,
                error=str(exc),
                traceback=traceback.format_exc(limit=5),
            )
            raise

        finally:
            gc.collect()
            log.info("text_extractor.end", rss_mb=round(_rss_mb(), 1))

    def _has_text_layer(self, raw: bytes) -> bool:
        """Detect whether a PDF has a selectable text layer.

        Uses two signals sampled across the whole document (not just the first
        few pages, which are often image-heavy cover pages):

        1. Embedded fonts — if a page has fonts, it was rendered with text,
           not scanned as an image.
        2. Word count — pages with embedded fonts but no visible text (e.g.
           image-only pages with an invisible font descriptor) still get caught.

        A page is considered "has text" when it has at least one embedded font
        OR at least MIN_WORDS_PER_PAGE words of actual text content. We sample
        up to SAMPLE_PAGES pages spread evenly across the document and require
        TEXT_PAGE_RATIO of them to have text before classifying as text-layer.
        """
        import pymupdf

        SAMPLE_PAGES = 20
        MIN_WORDS_PER_PAGE = 10
        TEXT_PAGE_RATIO = 0.4  # 40% of sampled pages must have text

        doc = None
        try:
            doc = pymupdf.open(stream=raw, filetype="pdf")
            page_count = len(doc)
            if page_count == 0:
                return False

            # Sample pages spread across the full document so cover pages
            # (which are often image-only) don't dominate the decision.
            if page_count <= SAMPLE_PAGES:
                sample = list(range(page_count))
            else:
                # Always include a few pages from the middle and end, plus
                # a random sample, so we catch both front-matter and body.
                step = page_count // SAMPLE_PAGES
                sample = list(range(0, page_count, step))[:SAMPLE_PAGES]

            text_pages = 0
            for i in sample:
                page = doc[i]
                has_fonts = bool(page.get_fonts())
                word_count = len((page.get_text("text") or "").split())
                if has_fonts or word_count >= MIN_WORDS_PER_PAGE:
                    text_pages += 1

            ratio = text_pages / len(sample)
            has_text = ratio >= TEXT_PAGE_RATIO

            log.info(
                "text_extractor.pdf_text_layer_check",
                pages=page_count,
                pages_sampled=len(sample),
                text_pages=text_pages,
                ratio=round(ratio, 2),
                has_text_layer=has_text,
            )

            return has_text

        finally:
            if doc:
                doc.close()

    def _extract_pdf_with_pymupdf(self, raw: bytes, has_text: bool) -> ExtractResult:
        import pymupdf

        doc = None
        pages: list[dict] = []

        try:
            doc = pymupdf.open(stream=raw, filetype="pdf")
            metadata = dict(doc.metadata or {})

            for index in range(len(doc)):
                page = doc[index]
                text = page.get_text("text") or ""

                pages.append(
                    {
                        "page": index + 1,
                        "text": text.strip(),
                    }
                )

            full_text = "\n\n".join(p["text"] for p in pages if p["text"])
            markdown = full_text

            log.info(
                "text_extractor.pdf_pymupdf_done",
                pages=len(pages),
                chars=len(full_text),
                rss_mb=round(_rss_mb(), 1),
            )

            return ExtractResult(
                pages=pages,
                tables=[],
                full_text=full_text,
                markdown=markdown,
                metadata=metadata,
                has_text_layer=has_text,
                figures=[],
                extraction_engine="pymupdf",
            )

        finally:
            if doc:
                doc.close()

    def _extract_spreadsheet(self, raw: bytes, ext: str) -> ExtractResult:
        # AMFI serves some files with a .xls extension whose content is
        # actually modern xlsx (zip-based) — xlrd rejects those outright.
        # Detect the real format from the file's magic bytes instead of
        # trusting the extension.
        if raw[:4] == b"PK\x03\x04":
            engine = "openpyxl"
        elif raw[:8] == b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1":
            engine = "xlrd"
        else:
            engine = "openpyxl" if ext == "xlsx" else "xlrd"

        try:
            xl = pd.ExcelFile(io.BytesIO(raw), engine=engine)
        except Exception as exc:
            raise DocumentExtractionError(f"Failed to open spreadsheet using engine={engine}: {exc}") from exc

        tables: list[pd.DataFrame] = []
        markdown_parts: list[str] = []
        metadata = {
            "sheet_count": len(xl.sheet_names),
            "sheets": [],
        }

        for sheet in xl.sheet_names:
            try:
                df = xl.parse(sheet)
                df = self._clean_dataframe(df)
                tables.append(df)

                sheet_meta = {
                    "sheet_name": sheet,
                    "rows": int(len(df)),
                    "columns": [str(c) for c in df.columns],
                }
                metadata["sheets"].append(sheet_meta)

                preview = df.head(MAX_EXCEL_PREVIEW_ROWS)
                markdown_parts.append(
                    f"## Sheet: {sheet}\n"
                    f"Rows: {len(df)}\n"
                    f"Columns: {', '.join(map(str, df.columns))}\n\n"
                    f"{preview.to_markdown(index=False)}"
                )

            except Exception as exc:
                log.warning(
                    "text_extractor.spreadsheet_sheet_failed",
                    sheet=sheet,
                    error=str(exc),
                )
                metadata["sheets"].append({"sheet_name": sheet, "error": str(exc)})

        full_text = "\n\n".join(markdown_parts)

        log.info(
            "text_extractor.spreadsheet_done",
            sheets=len(tables),
            chars=len(full_text),
            rss_mb=round(_rss_mb(), 1),
        )

        return ExtractResult(
            pages=[],
            tables=tables,
            full_text=full_text,
            markdown=full_text,
            metadata=metadata,
            has_text_layer=True,
            figures=[],
            extraction_engine=f"pandas_{engine}",
        )

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
        df.columns = _dedupe_columns(str(col).strip().replace("\n", " ") for col in df.columns)

        # Object columns with mixed types (e.g. numbers and strings in the
        # same column — common in AMFI's hand-formatted sheets) make pyarrow
        # raise "Expected bytes, got a 'int' object" on to_parquet(). Coerce
        # those columns to string so the write never fails on real data.
        for col in df.columns:
            if df[col].dtype == object:
                types_present = df[col].map(lambda v: type(v) if v is not None else None).dropna().unique()
                if len(types_present) > 1:
                    df[col] = df[col].astype(str).replace("nan", "")

        return df


# ──────────────────────────────────────────────────────────────────────────
# Stage 2 — table-worker / ocr-worker (high memory: Docling + torch)
# ──────────────────────────────────────────────────────────────────────────


class _DoclingExtractorBase:
    """Shared Docling conversion logic. Subclassed by TableExtractor (ocr=False)
    and OcrExtractor (ocr=True) so each gets its own cached converter and its
    own right-sized ECS task — neither loads unless that worker runs."""

    _converter: object | None = None
    _ocr_mode: bool = False
    _engine_name: str = "docling"

    def extract(self, raw: bytes) -> ExtractResult:
        if not raw:
            raise DocumentExtractionError("Empty file received")

        log.info("docling_extractor.start", ocr=self._ocr_mode, rss_mb=round(_rss_mb(), 1))

        try:
            result = self._extract_pdf_with_docling(raw)
            log.info(
                "docling_extractor.success",
                pages=len(result.pages),
                tables=len(result.tables),
                chars=len(result.full_text),
            )
            return result

        except Exception as exc:
            log.exception(
                "docling_extractor.failed",
                ocr=self._ocr_mode,
                error=str(exc),
                traceback=traceback.format_exc(limit=5),
            )
            raise

        finally:
            gc.collect()
            log.info("docling_extractor.end", rss_mb=round(_rss_mb(), 1))

    def _extract_pdf_with_docling(self, raw: bytes) -> ExtractResult:
        from docling.datamodel.base_models import DocumentStream

        converter = self._get_docling_converter()
        stream = DocumentStream(name="document.pdf", stream=io.BytesIO(raw))

        log.info("docling_extractor.before_convert", rss_mb=round(_rss_mb(), 1))
        result = converter.convert(stream)
        log.info("docling_extractor.after_convert", rss_mb=round(_rss_mb(), 1))

        doc = result.document
        markdown = doc.export_to_markdown() or ""

        pages = self._extract_docling_pages(doc, markdown)
        full_text = "\n\n".join(p["text"] for p in pages if p["text"]) or markdown

        tables, table_meta, failed_tables = self._extract_docling_tables(doc)
        figures = self._extract_docling_figures(doc)
        metadata = self._extract_docling_metadata(doc)

        del result, doc, stream
        gc.collect()

        return ExtractResult(
            pages=pages,
            tables=tables,
            full_text=full_text,
            markdown=markdown,
            metadata=metadata,
            has_text_layer=not self._ocr_mode,
            figures=figures,
            extraction_engine=self._engine_name,
            failed_tables=failed_tables,
            table_meta=table_meta,
        )

    def _get_docling_converter(self) -> object:
        from docling.datamodel.pipeline_options import (
            AcceleratorDevice,
            AcceleratorOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption

        if self.__class__._converter is None:
            accel = AcceleratorOptions(
                num_threads=int(os.getenv("DOCLING_THREADS", "1")),
                device=AcceleratorDevice.CPU,
            )
            opts = PdfPipelineOptions()
            opts.do_ocr = self._ocr_mode
            opts.do_table_structure = True
            opts.accelerator_options = accel

            self.__class__._converter = DocumentConverter(format_options={"pdf": PdfFormatOption(pipeline_options=opts)})

        return self.__class__._converter

    def _extract_docling_pages(self, doc: Any, markdown: str) -> list[dict]:
        pages: list[dict] = []
        try:
            for page_no, page in doc.pages.items():
                text_items = []
                assembled = getattr(page, "assembled", None)
                body = getattr(assembled, "body", []) if assembled else []
                for item in body:
                    text = getattr(item, "text", None)
                    if text:
                        text_items.append(text)
                pages.append({"page": int(page_no), "text": "\n".join(text_items).strip()})
        except Exception as exc:
            log.warning("docling_extractor.page_extract_failed", error=str(exc))

        if not pages:
            pages = [{"page": 1, "text": markdown}]
        return pages

    def _extract_docling_tables(self, doc: Any) -> tuple[list[pd.DataFrame], list[TableMeta], int]:
        tables: list[pd.DataFrame] = []
        metas: list[TableMeta] = []
        failed = 0

        for table_index, tbl in enumerate(getattr(doc, "tables", []), start=1):
            try:
                df = tbl.export_to_dataframe()
                if df is None or df.empty:
                    continue
                df.columns = _dedupe_columns(df.columns)
                if len(df) > MAX_TABLE_ROWS_IN_MEMORY:
                    log.warning(
                        "docling_extractor.table_too_large_truncated",
                        table_index=table_index,
                        rows=len(df),
                        max_rows=MAX_TABLE_ROWS_IN_MEMORY,
                    )
                    df = df.head(MAX_TABLE_ROWS_IN_MEMORY)
                tables.append(df)

                # Extract page number and caption from Docling provenance
                page_no = None
                caption = None
                try:
                    prov = getattr(tbl, "prov", None)
                    if prov:
                        page_no = getattr(prov[0], "page_no", None)
                    captions = getattr(tbl, "captions", [])
                    if captions:
                        caption = str(getattr(captions[0], "text", "") or "").strip() or None
                except Exception:
                    pass
                metas.append(TableMeta(page_number=page_no, table_name=caption))

            except Exception as exc:
                failed += 1
                log.warning(
                    "docling_extractor.table_export_failed",
                    table_index=table_index,
                    error=str(exc),
                )

        return tables, metas, failed

    def _extract_docling_figures(self, doc: Any) -> list[dict]:
        figures: list[dict] = []
        for fig in getattr(doc, "pictures", []):
            try:
                page_no = None
                prov = getattr(fig, "prov", None)
                if prov:
                    page_no = getattr(prov[0], "page_no", None)
                figures.append(
                    {
                        "page": page_no,
                        "caption": str(getattr(fig, "caption", "") or ""),
                        "label": str(getattr(fig, "label", "") or ""),
                    }
                )
            except Exception as exc:
                log.warning("docling_extractor.figure_extract_failed", error=str(exc))
        return figures

    def _extract_docling_metadata(self, doc: Any) -> dict:
        meta = {}
        try:
            m = getattr(doc, "meta", None)
            if m:
                meta = {
                    "title": getattr(m, "title", None),
                    "author": getattr(m, "authors", None),
                    "language": getattr(m, "language", None),
                }
        except Exception as exc:
            log.warning("docling_extractor.metadata_extract_failed", error=str(exc))
        return meta


class TableExtractor(_DoclingExtractorBase):
    """High-memory worker for native-text PDFs (has_text_layer=True).
    Docling without OCR: layout + table-structure models only."""

    _converter: object | None = None
    _ocr_mode = False
    _engine_name = "docling_text"


class OcrExtractor(_DoclingExtractorBase):
    """High-memory worker for scanned PDFs (has_text_layer=False).
    Docling with OCR enabled, in addition to layout + table-structure."""

    _converter: object | None = None
    _ocr_mode = True
    _engine_name = "docling_ocr"
