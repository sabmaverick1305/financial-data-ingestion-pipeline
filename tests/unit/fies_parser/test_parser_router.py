from __future__ import annotations

from pathlib import Path

import pytest

from fies_parser.adapters.base import ParserAdapter
from fies_parser.adapters.capabilities import ParserCapabilities
from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.engine.exceptions import NoParserAvailableError
from fies_parser.engine.models import ParseRequest
from fies_parser.engine.parser_engine import ParserEngine
from fies_parser.engine.registry import ParserRegistry
from fies_parser.routing.parser_router import ParserRouter

from .conftest import build_pdf, make_source_document


class _StubAdapter(ParserAdapter):
    def __init__(self, name: str, version: str, capabilities: ParserCapabilities) -> None:
        self.name = name
        self.version = version
        self._capabilities = capabilities

    @property
    def capabilities(self) -> ParserCapabilities:
        return self._capabilities

    def supports(self, request: ParseRequest) -> bool:
        return request.document.mime_type in self._capabilities.supported_mime_types

    def parse(self, request: ParseRequest) -> ParserCandidate:
        return ParserCandidate(document_id=request.document.document_id, parser_name=self.name, parser_version=self.version)


def _fast_adapter() -> _StubAdapter:
    return _StubAdapter(
        "pymupdf", "1.0.0", ParserCapabilities(supports_page_selection=True, supported_mime_types=("application/pdf",))
    )


def _table_adapter() -> _StubAdapter:
    return _StubAdapter(
        "docling",
        "1.0.0",
        ParserCapabilities(supports_page_selection=True, supports_tables=True, supported_mime_types=("application/pdf",)),
    )


def _router_with(*adapters: ParserAdapter) -> ParserRouter:
    registry = ParserRegistry()
    for adapter in adapters:
        registry.register(adapter)
    return ParserRouter(registry, ParserEngine(registry))


def test_route_picks_fast_parser_for_text_document(two_page_pdf: Path) -> None:
    router = _router_with(_fast_adapter(), _table_adapter())
    request = ParseRequest(document=make_source_document(two_page_pdf))

    decision = router.route(request)

    assert decision.parser_name == "pymupdf"
    assert decision.fallback_parser_names == ("docling",)


def test_route_and_parse_executes_the_chosen_parser(two_page_pdf: Path) -> None:
    router = _router_with(_fast_adapter(), _table_adapter())
    request = ParseRequest(document=make_source_document(two_page_pdf))

    candidate = router.route_and_parse(request)

    assert candidate.parser_name == "pymupdf"
    assert candidate.document_id == "doc-1"


def test_route_raises_when_no_registered_parser_fits(tmp_path: Path) -> None:
    blank_pdf = build_pdf(tmp_path / "blank.pdf", ["", ""])
    ocr_only = _StubAdapter(
        "docling_ocr",
        "1.0.0",
        ParserCapabilities(supports_ocr=True, supported_mime_types=("application/pdf",)),
    )
    router = _router_with(_fast_adapter())  # no OCR-capable adapter registered
    request = ParseRequest(document=make_source_document(blank_pdf))

    with pytest.raises(NoParserAvailableError):
        router.route(request)

    # Sanity: with an OCR-capable adapter registered, the same scanned document routes fine.
    router_with_ocr = _router_with(_fast_adapter(), ocr_only)
    decision = router_with_ocr.route(request)
    assert decision.parser_name == "docling_ocr"
