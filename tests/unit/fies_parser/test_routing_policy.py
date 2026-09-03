from __future__ import annotations

import pytest

from fies_parser.adapters.capabilities import ParserCapabilities
from fies_parser.engine.exceptions import NoParserAvailableError
from fies_parser.preflight.models import DocumentProfile
from fies_parser.routing.routing_policy import DefaultRoutingPolicy

_PDF_CAPS_FAST = ParserCapabilities(supports_page_selection=True, supported_mime_types=("application/pdf",))
_PDF_CAPS_TABLES = ParserCapabilities(
    supports_page_selection=True, supports_tables=True, supported_mime_types=("application/pdf",)
)
_PDF_CAPS_OCR = ParserCapabilities(
    supports_page_selection=True, supports_ocr=True, supports_tables=True, supported_mime_types=("application/pdf",)
)


def _profile(has_text_layer: bool = True, mime_type: str = "application/pdf", likely_has_tables: bool = False) -> DocumentProfile:
    return DocumentProfile(
        document_id="doc-1",
        mime_type=mime_type,
        page_count=1,
        file_size_bytes=100,
        has_text_layer=has_text_layer,
        text_page_ratio=1.0 if has_text_layer else 0.0,
        likely_has_tables=likely_has_tables,
    )


def test_no_parsers_registered_raises() -> None:
    with pytest.raises(NoParserAvailableError):
        DefaultRoutingPolicy().decide(_profile(), {})


def test_no_parser_supports_mime_type_raises() -> None:
    available = {"pymupdf": ParserCapabilities(supported_mime_types=("application/vnd.ms-excel",))}

    with pytest.raises(NoParserAvailableError):
        DefaultRoutingPolicy().decide(_profile(), available)


def test_text_document_prefers_fast_parser_over_table_capable() -> None:
    available = {"pymupdf": _PDF_CAPS_FAST, "docling": _PDF_CAPS_TABLES}

    decision = DefaultRoutingPolicy().decide(_profile(has_text_layer=True), available)

    assert decision.parser_name == "pymupdf"
    assert "docling" in decision.fallback_parser_names


def test_text_document_falls_back_to_table_capable_when_no_fast_parser() -> None:
    available = {"docling": _PDF_CAPS_TABLES}

    decision = DefaultRoutingPolicy().decide(_profile(has_text_layer=True), available)

    assert decision.parser_name == "docling"


def test_scanned_document_routes_to_ocr_capable_parser() -> None:
    available = {"pymupdf": _PDF_CAPS_FAST, "docling_ocr": _PDF_CAPS_OCR}

    decision = DefaultRoutingPolicy().decide(_profile(has_text_layer=False), available)

    assert decision.parser_name == "docling_ocr"


def test_scanned_document_with_no_ocr_capable_parser_raises() -> None:
    available = {"pymupdf": _PDF_CAPS_FAST}

    with pytest.raises(NoParserAvailableError):
        DefaultRoutingPolicy().decide(_profile(has_text_layer=False), available)


def test_decision_is_deterministic_across_ties() -> None:
    available = {"zeta": _PDF_CAPS_FAST, "alpha": _PDF_CAPS_FAST}

    decision = DefaultRoutingPolicy().decide(_profile(has_text_layer=True), available)

    assert decision.parser_name == "alpha"
    assert decision.fallback_parser_names == ("zeta",)


def test_likely_has_tables_prefers_table_capable_parser_over_fast() -> None:
    available = {"pymupdf": _PDF_CAPS_FAST, "docling": _PDF_CAPS_TABLES}

    decision = DefaultRoutingPolicy().decide(_profile(has_text_layer=True, likely_has_tables=True), available)

    assert decision.parser_name == "docling"
    assert "pymupdf" in decision.fallback_parser_names


def test_likely_has_tables_falls_back_to_fast_parser_when_no_table_capable_registered() -> None:
    available = {"pymupdf": _PDF_CAPS_FAST}

    decision = DefaultRoutingPolicy().decide(_profile(has_text_layer=True, likely_has_tables=True), available)

    assert decision.parser_name == "pymupdf"
