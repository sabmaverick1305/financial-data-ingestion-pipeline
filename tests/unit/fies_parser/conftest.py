from __future__ import annotations

import hashlib
from pathlib import Path

import pymupdf
import pytest

from fies_parser.engine.models import SourceDocument


def build_pdf(path: Path, pages_text: list[str], rotation: int = 0) -> Path:
    """Generate a tiny PDF with one page per string in `pages_text`. An empty
    string produces a page with no text blocks."""
    doc = pymupdf.open()
    try:
        for text in pages_text:
            page = doc.new_page()
            if text:
                page.insert_text((72, 72), text)
            if rotation:
                page.set_rotation(rotation)
        doc.save(path)
    finally:
        doc.close()
    return path


def make_source_document(path: Path, document_id: str = "doc-1") -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        file_path=path,
        file_name=path.name,
        mime_type="application/pdf",
        file_hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        source="test",
    )


@pytest.fixture
def two_page_pdf(tmp_path: Path) -> Path:
    return build_pdf(tmp_path / "two_page.pdf", ["First page has some text", "Second page has other text"])


@pytest.fixture
def blank_page_pdf(tmp_path: Path) -> Path:
    return build_pdf(tmp_path / "blank_page.pdf", ["Some real text on page one", ""])


@pytest.fixture
def rotated_pdf(tmp_path: Path) -> Path:
    return build_pdf(tmp_path / "rotated.pdf", ["Rotated page text"], rotation=90)
