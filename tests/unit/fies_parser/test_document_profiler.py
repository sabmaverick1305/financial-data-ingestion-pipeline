from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from fies_parser.engine.exceptions import UnsupportedDocumentError
from fies_parser.engine.models import SourceDocument
from fies_parser.preflight.document_profiler import DocumentProfiler
from fies_parser.preflight.page_profiler import PageProfiler

from .conftest import build_pdf, make_source_document


def test_profiles_text_document(two_page_pdf: Path) -> None:
    profile = DocumentProfiler().profile(make_source_document(two_page_pdf))

    assert profile.page_count == 2
    assert profile.has_text_layer is True
    assert profile.text_page_ratio == 1.0
    assert len(profile.pages) == 2
    assert profile.file_size_bytes == two_page_pdf.stat().st_size


def test_profiles_scanned_looking_document(tmp_path: Path) -> None:
    blank_pdf = build_pdf(tmp_path / "blank.pdf", ["", "", ""])

    profile = DocumentProfiler().profile(make_source_document(blank_pdf))

    assert profile.has_text_layer is False
    assert profile.text_page_ratio == 0.0
    assert all(not page.has_text for page in profile.pages)


def test_non_pdf_mime_type_raises(tmp_path: Path) -> None:
    file_path = tmp_path / "sheet.xlsx"
    file_path.write_bytes(b"not really an xlsx")
    document = SourceDocument(
        document_id="doc-1",
        file_path=file_path,
        file_name=file_path.name,
        mime_type="application/vnd.ms-excel",
        file_hash="deadbeef",
        source="test",
    )

    with pytest.raises(UnsupportedDocumentError):
        DocumentProfiler().profile(document)


def test_sampling_caps_number_of_profiled_pages(tmp_path: Path) -> None:
    five_page_pdf = build_pdf(tmp_path / "five.pdf", ["one", "two", "three", "four", "five"])

    profile = DocumentProfiler(sample_pages=2).profile(make_source_document(five_page_pdf))

    assert profile.page_count == 5
    assert len(profile.pages) == 2


def test_page_profiler_word_count_and_has_text(two_page_pdf: Path) -> None:
    with pymupdf.open(two_page_pdf) as doc:
        profile = PageProfiler().profile(doc[0], page_number=1)

    assert profile.page_number == 1
    assert profile.has_text is True
    assert profile.word_count > 0


def test_page_profiler_reports_no_text_for_blank_page(tmp_path: Path) -> None:
    blank_pdf = build_pdf(tmp_path / "blank_single.pdf", [""])

    with pymupdf.open(blank_pdf) as doc:
        profile = PageProfiler().profile(doc[0], page_number=1)

    assert profile.has_text is False
    assert profile.word_count == 0
