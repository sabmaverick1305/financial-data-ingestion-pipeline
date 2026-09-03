from __future__ import annotations

import pymupdf
import pytest

from financial_pipeline.processing.extractor import TextExtractor
from financial_pipeline.processing.parser_engine_integration import extract_pdf_via_parser_engine


def _build_pdf_bytes(pages_text: list[str]) -> bytes:
    doc = pymupdf.open()
    try:
        for text in pages_text:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
        return doc.tobytes()
    finally:
        doc.close()


@pytest.fixture
def two_page_pdf_bytes() -> bytes:
    return _build_pdf_bytes(["First page body text", "Second page body text"])


def test_extract_pdf_via_parser_engine_maps_pages_and_text(two_page_pdf_bytes: bytes) -> None:
    result = extract_pdf_via_parser_engine(two_page_pdf_bytes, has_text_layer=True)

    assert [p["page"] for p in result.pages] == [1, 2]
    assert "First page body text" in result.pages[0]["text"]
    assert "Second page body text" in result.pages[1]["text"]
    assert "First page body text" in result.full_text
    assert result.markdown == result.full_text
    assert result.extraction_engine == "pymupdf"
    assert result.has_text_layer is True
    assert result.tables == []


def test_extract_pdf_via_parser_engine_carries_document_metadata(two_page_pdf_bytes: bytes) -> None:
    result = extract_pdf_via_parser_engine(two_page_pdf_bytes, has_text_layer=True)

    assert isinstance(result.metadata, dict)
    assert "format" in result.metadata


def test_text_extractor_pdf_path_uses_parser_engine(two_page_pdf_bytes: bytes) -> None:
    result = TextExtractor().extract(two_page_pdf_bytes, "pdf")

    assert result.extraction_engine == "pymupdf"
    assert len(result.pages) == 2
    assert result.tables == []
