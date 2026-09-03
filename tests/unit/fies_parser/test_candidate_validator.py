from __future__ import annotations

import pytest

from fies_parser.canonical.candidate_models import (
    BoundingBox,
    CandidateElement,
    CandidatePage,
    CandidateTable,
    ElementType,
    ParserCandidate,
)
from fies_parser.engine.candidate_validator import CandidateValidator
from fies_parser.engine.exceptions import InvalidCandidateError
from fies_parser.engine.models import ParseRequest
from fies_parser.engine.parser_engine import ParserEngine
from fies_parser.engine.registry import ParserRegistry

from .conftest import make_source_document


def _element(element_id: str = "doc-p1-b0", page_number: int = 1, bbox: BoundingBox | None = None) -> CandidateElement:
    return CandidateElement(
        element_id=element_id,
        element_type=ElementType.PARAGRAPH,
        page_number=page_number,
        text="hello",
        bbox=bbox,
    )


def _page(page_number: int = 1, element_ids: list[str] | None = None, table_ids: list[str] | None = None) -> CandidatePage:
    return CandidatePage(page_number=page_number, element_ids=element_ids or [], table_ids=table_ids or [])


def _candidate(**overrides: object) -> ParserCandidate:
    defaults: dict[str, object] = {
        "document_id": "doc-1",
        "parser_name": "stub",
        "parser_version": "1.0.0",
    }
    defaults.update(overrides)
    return ParserCandidate(**defaults)  # type: ignore[arg-type]


def test_valid_candidate_has_no_issues() -> None:
    element = _element()
    candidate = _candidate(elements=[element], pages=[_page(element_ids=[element.element_id])])

    assert CandidateValidator().validate(candidate) == []


def test_duplicate_element_id_is_flagged() -> None:
    candidate = _candidate(elements=[_element("dup"), _element("dup")])

    issues = CandidateValidator().validate(candidate)

    assert any("duplicate element_id" in issue for issue in issues)


def test_duplicate_table_id_is_flagged() -> None:
    table = CandidateTable(table_id="tbl-1", page_start=1, page_end=1)
    candidate = _candidate(tables=[table, table])

    issues = CandidateValidator().validate(candidate)

    assert any("duplicate table_id" in issue for issue in issues)


def test_page_referencing_unknown_element_id_is_flagged() -> None:
    candidate = _candidate(pages=[_page(element_ids=["does-not-exist"])])

    issues = CandidateValidator().validate(candidate)

    assert any("unknown element_id" in issue for issue in issues)


def test_page_referencing_element_from_a_different_page_is_flagged() -> None:
    element = _element(page_number=2)
    candidate = _candidate(elements=[element], pages=[_page(page_number=1, element_ids=[element.element_id])])

    issues = CandidateValidator().validate(candidate)

    assert any("belongs to page 2" in issue for issue in issues)


def test_table_page_start_after_page_end_is_flagged() -> None:
    table = CandidateTable(table_id="tbl-1", page_start=3, page_end=1)
    candidate = _candidate(tables=[table])

    issues = CandidateValidator().validate(candidate)

    assert any("page_start > page_end" in issue for issue in issues)


def test_degenerate_bounding_box_is_flagged() -> None:
    bad_bbox = BoundingBox(x0=10, y0=10, x1=5, y1=5)
    element = _element(bbox=bad_bbox)
    candidate = _candidate(elements=[element])

    issues = CandidateValidator().validate(candidate)

    assert any("degenerate bounding box" in issue for issue in issues)


def test_engine_raises_invalid_candidate_error_for_structurally_broken_output(tmp_path) -> None:  # noqa: ANN001
    from fies_parser.adapters.base import ParserAdapter

    class _BrokenAdapter(ParserAdapter):
        name = "broken"
        version = "1.0.0"

        def supports(self, request: ParseRequest) -> bool:
            return True

        def parse(self, request: ParseRequest) -> ParserCandidate:
            return _candidate(document_id=request.document.document_id, pages=[_page(element_ids=["missing"])])

    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    document = make_source_document(file_path)

    registry = ParserRegistry()
    registry.register(_BrokenAdapter())
    engine = ParserEngine(registry)

    with pytest.raises(InvalidCandidateError) as excinfo:
        engine.run("broken", ParseRequest(document=document))

    assert excinfo.value.parser_name == "broken"
    assert any("unknown element_id" in issue for issue in excinfo.value.issues)
