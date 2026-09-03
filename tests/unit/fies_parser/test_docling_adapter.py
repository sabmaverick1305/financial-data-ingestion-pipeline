from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from fies_parser.adapters.docling_adapter import DoclingAdapter
from fies_parser.canonical.candidate_models import ElementType
from fies_parser.engine.exceptions import InvalidPageSelectionError
from fies_parser.engine.models import ParseRequest

from .conftest import make_source_document


class _FakeBBox(SimpleNamespace):
    def to_top_left_origin(self, page_height: float) -> _FakeBBox:
        return self


def _fake_item(
    text: str,
    label: str,
    page_no: int | None,
    self_ref: str,
    parent_ref: str | None = None,
    bbox: _FakeBBox | None = None,
) -> SimpleNamespace:
    parent = SimpleNamespace(cref=parent_ref) if parent_ref else None
    prov = [SimpleNamespace(page_no=page_no, bbox=bbox)] if page_no is not None else []
    return SimpleNamespace(
        text=text,
        label=SimpleNamespace(value=label),
        self_ref=self_ref,
        parent=parent,
        prov=prov,
    )


def _fake_table(page_no: int, caption: str | None = None) -> SimpleNamespace:
    df = pd.DataFrame({"Scheme": ["Fund A", "Fund B"], "NAV": ["3,41,201.50", "1,20,000.00"]})
    captions = [SimpleNamespace(text=caption)] if caption else []
    return SimpleNamespace(
        export_to_dataframe=lambda doc=None: df,
        prov=[SimpleNamespace(page_no=page_no, bbox=None)],
        captions=captions,
    )


def _fake_doc(
    items: list[SimpleNamespace], pages: dict[int, tuple[float, float]], tables: list[SimpleNamespace]
) -> SimpleNamespace:
    return SimpleNamespace(
        iterate_items=lambda: [(item, 0) for item in items],
        pages={no: SimpleNamespace(size=SimpleNamespace(width=w, height=h), page_no=no) for no, (w, h) in pages.items()},
        tables=tables,
        origin=SimpleNamespace(filename="doc.pdf", mimetype="application/pdf"),
        name="doc",
    )


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"%PDF-1.4 fake content")
    return path


def _adapter_with_fake_convert(fake_doc: SimpleNamespace) -> DoclingAdapter:
    adapter = DoclingAdapter()
    adapter._get_converter = lambda: SimpleNamespace(  # type: ignore[method-assign]
        convert=lambda *args, **kwargs: SimpleNamespace(document=fake_doc)
    )
    return adapter


def test_maps_text_items_to_elements_with_type_and_page(pdf_path: Path) -> None:
    items = [
        _fake_item("Report Title", "title", 1, "#/texts/0"),
        _fake_item("A body paragraph", "text", 1, "#/texts/1"),
        _fake_item("Section Heading", "section_header", 2, "#/texts/2"),
    ]
    doc = _fake_doc(items, {1: (595, 842), 2: (595, 842)}, [])
    adapter = _adapter_with_fake_convert(doc)

    candidate = adapter.parse(ParseRequest(document=make_source_document(pdf_path)))

    types_by_text = {el.text: el.element_type for el in candidate.elements}
    assert types_by_text["Report Title"] == ElementType.TITLE
    assert types_by_text["A body paragraph"] == ElementType.PARAGRAPH
    assert types_by_text["Section Heading"] == ElementType.HEADING
    assert [p.page_number for p in candidate.pages] == [1, 2]


def test_resolves_parent_element_id(pdf_path: Path) -> None:
    items = [
        _fake_item("Parent heading", "section_header", 1, "#/texts/0"),
        _fake_item("Child paragraph", "text", 1, "#/texts/1", parent_ref="#/texts/0"),
    ]
    doc = _fake_doc(items, {1: (595, 842)}, [])
    adapter = _adapter_with_fake_convert(doc)

    candidate = adapter.parse(ParseRequest(document=make_source_document(pdf_path)))

    parent = next(el for el in candidate.elements if el.text == "Parent heading")
    child = next(el for el in candidate.elements if el.text == "Child paragraph")
    assert child.parent_element_id == parent.element_id


def test_empty_text_items_are_skipped(pdf_path: Path) -> None:
    items = [
        _fake_item("", "text", 1, "#/texts/0"),
        _fake_item("Real text", "text", 1, "#/texts/1"),
    ]
    doc = _fake_doc(items, {1: (595, 842)}, [])
    adapter = _adapter_with_fake_convert(doc)

    candidate = adapter.parse(ParseRequest(document=make_source_document(pdf_path)))

    assert [el.text for el in candidate.elements] == ["Real text"]


def test_bounding_box_extraction(pdf_path: Path) -> None:
    bbox = _FakeBBox(l=10.0, t=20.0, r=110.0, b=40.0)
    items = [_fake_item("With bbox", "text", 1, "#/texts/0", bbox=bbox)]
    doc = _fake_doc(items, {1: (595, 842)}, [])
    adapter = _adapter_with_fake_convert(doc)

    candidate = adapter.parse(ParseRequest(document=make_source_document(pdf_path)))

    element = candidate.elements[0]
    assert element.bbox is not None
    assert (element.bbox.x0, element.bbox.y0, element.bbox.x1, element.bbox.y1) == (10.0, 20.0, 110.0, 40.0)


def test_tables_preserve_raw_string_values(pdf_path: Path) -> None:
    doc = _fake_doc([], {1: (595, 842)}, [_fake_table(page_no=1, caption="NAV Summary")])
    adapter = _adapter_with_fake_convert(doc)

    candidate = adapter.parse(ParseRequest(document=make_source_document(pdf_path)))

    assert len(candidate.tables) == 1
    table = candidate.tables[0]
    assert table.headers == ["Scheme", "NAV"]
    assert table.rows == [["Fund A", "3,41,201.50"], ["Fund B", "1,20,000.00"]]
    assert table.title == "NAV Summary"
    assert table.page_start == table.page_end == 1


def test_selected_page_parsing_filters_items(pdf_path: Path) -> None:
    items = [
        _fake_item("Page one text", "text", 1, "#/texts/0"),
        _fake_item("Page two text", "text", 2, "#/texts/1"),
    ]
    doc = _fake_doc(items, {1: (595, 842), 2: (595, 842)}, [])
    adapter = _adapter_with_fake_convert(doc)

    candidate = adapter.parse(ParseRequest(document=make_source_document(pdf_path), pages=(2,)))

    assert [p.page_number for p in candidate.pages] == [2]
    assert [el.text for el in candidate.elements] == ["Page two text"]


def test_invalid_page_number_raises(pdf_path: Path) -> None:
    doc = _fake_doc([], {1: (595, 842)}, [])
    adapter = _adapter_with_fake_convert(doc)

    with pytest.raises(InvalidPageSelectionError):
        adapter.parse(ParseRequest(document=make_source_document(pdf_path), pages=(99,)))


def test_document_metadata_is_captured(pdf_path: Path) -> None:
    doc = _fake_doc([], {1: (595, 842)}, [])
    adapter = _adapter_with_fake_convert(doc)

    candidate = adapter.parse(ParseRequest(document=make_source_document(pdf_path)))

    assert candidate.document_metadata["filename"] == "doc.pdf"


def test_supports_pdf_mime_type(pdf_path: Path) -> None:
    adapter = DoclingAdapter()
    supported = ParseRequest(document=make_source_document(pdf_path))

    assert adapter.supports(supported) is True
