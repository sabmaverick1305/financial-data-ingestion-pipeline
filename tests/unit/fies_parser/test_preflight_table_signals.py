from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from fies_parser.preflight.document_profiler import DocumentProfiler
from fies_parser.preflight.page_profiler import PageProfiler

from .conftest import make_source_document


def _build_table_pdf(path: Path) -> Path:
    doc = pymupdf.open()
    try:
        page = doc.new_page()
        x0, y0, cell_w, cell_h, rows, cols = 72, 100, 100, 20, 4, 3
        for r in range(rows + 1):
            page.draw_line((x0, y0 + r * cell_h), (x0 + cols * cell_w, y0 + r * cell_h))
        for c in range(cols + 1):
            page.draw_line((x0 + c * cell_w, y0), (x0 + c * cell_w, y0 + rows * cell_h))
        for r in range(rows):
            for c in range(cols):
                page.insert_text((x0 + c * cell_w + 5, y0 + r * cell_h + 14), f"R{r}C{c}", fontsize=8)
        doc.save(path)
    finally:
        doc.close()
    return path


@pytest.fixture
def table_pdf(tmp_path: Path) -> Path:
    return _build_table_pdf(tmp_path / "table.pdf")


def test_page_profiler_detects_table(table_pdf: Path) -> None:
    with pymupdf.open(table_pdf) as doc:
        profile = PageProfiler().profile(doc[0], page_number=1)

    assert profile.table_count == 1
    assert profile.drawing_count > 0


def test_page_profiler_reports_zero_tables_for_plain_text(two_page_pdf: Path) -> None:
    with pymupdf.open(two_page_pdf) as doc:
        profile = PageProfiler().profile(doc[0], page_number=1)

    assert profile.table_count == 0


def test_document_profiler_sets_likely_has_tables(table_pdf: Path) -> None:
    profile = DocumentProfiler().profile(make_source_document(table_pdf))

    assert profile.likely_has_tables is True


def test_document_profiler_likely_has_tables_false_for_plain_text(two_page_pdf: Path) -> None:
    profile = DocumentProfiler().profile(make_source_document(two_page_pdf))

    assert profile.likely_has_tables is False
