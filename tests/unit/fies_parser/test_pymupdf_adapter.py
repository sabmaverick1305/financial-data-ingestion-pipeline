from __future__ import annotations

from pathlib import Path

import pytest

from fies_parser.adapters import pymupdf_adapter as pymupdf_adapter_module
from fies_parser.adapters.pymupdf_adapter import PyMuPDFAdapter
from fies_parser.canonical.candidate_models import ElementType
from fies_parser.engine.exceptions import InvalidPageSelectionError
from fies_parser.engine.models import ParseRequest

from .conftest import make_source_document


def test_complete_document_parsing(two_page_pdf: Path) -> None:
    adapter = PyMuPDFAdapter()
    request = ParseRequest(document=make_source_document(two_page_pdf))

    candidate = adapter.parse(request)

    assert [p.page_number for p in candidate.pages] == [1, 2]
    assert candidate.tables == []
    assert any("First page" in (el.text or "") for el in candidate.elements)
    assert any("Second page" in (el.text or "") for el in candidate.elements)


def test_selected_page_parsing(two_page_pdf: Path) -> None:
    adapter = PyMuPDFAdapter()
    request = ParseRequest(document=make_source_document(two_page_pdf), pages=(2,))

    candidate = adapter.parse(request)

    assert [p.page_number for p in candidate.pages] == [2]
    assert all(el.page_number == 2 for el in candidate.elements)
    assert all("Second page" in (el.text or "") for el in candidate.elements)


def test_invalid_page_number_raises(two_page_pdf: Path) -> None:
    adapter = PyMuPDFAdapter()
    request = ParseRequest(document=make_source_document(two_page_pdf), pages=(99,))

    with pytest.raises(InvalidPageSelectionError) as excinfo:
        adapter.parse(request)

    assert excinfo.value.pages == (99,)


def test_duplicate_page_selection_is_deduplicated(two_page_pdf: Path) -> None:
    request = ParseRequest(document=make_source_document(two_page_pdf), pages=(2, 1, 2, 1))

    assert request.pages == (1, 2)


def test_empty_text_blocks_are_skipped(blank_page_pdf: Path) -> None:
    adapter = PyMuPDFAdapter()
    request = ParseRequest(document=make_source_document(blank_page_pdf))

    candidate = adapter.parse(request)

    page_two = next(p for p in candidate.pages if p.page_number == 2)
    assert page_two.element_ids == []
    assert page_two.text is None
    assert not any(el.page_number == 2 for el in candidate.elements)


def test_one_based_page_numbers(two_page_pdf: Path) -> None:
    adapter = PyMuPDFAdapter()
    request = ParseRequest(document=make_source_document(two_page_pdf), pages=(1,))

    candidate = adapter.parse(request)

    assert candidate.pages[0].page_number == 1
    assert candidate.elements[0].page_number == 1


def test_bounding_box_extraction(two_page_pdf: Path) -> None:
    adapter = PyMuPDFAdapter()
    request = ParseRequest(document=make_source_document(two_page_pdf), pages=(1,))

    candidate = adapter.parse(request)

    element = candidate.elements[0]
    assert element.element_type == ElementType.PARAGRAPH
    assert element.bbox is not None
    assert element.bbox.x1 > element.bbox.x0
    assert element.bbox.y1 > element.bbox.y0


def test_page_rotation_and_dimensions(rotated_pdf: Path) -> None:
    adapter = PyMuPDFAdapter()
    request = ParseRequest(document=make_source_document(rotated_pdf))

    candidate = adapter.parse(request)

    page = candidate.pages[0]
    assert page.rotation == 90
    assert page.width is not None
    assert page.height is not None


def test_file_handle_cleanup_after_execution(two_page_pdf: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_open = pymupdf_adapter_module.pymupdf.open
    opened_docs = []

    def spy_open(*args, **kwargs):
        doc = real_open(*args, **kwargs)
        opened_docs.append(doc)
        return doc

    monkeypatch.setattr(pymupdf_adapter_module.pymupdf, "open", spy_open)

    adapter = PyMuPDFAdapter()
    request = ParseRequest(document=make_source_document(two_page_pdf))
    adapter.parse(request)

    assert len(opened_docs) == 1
    assert opened_docs[0].is_closed
