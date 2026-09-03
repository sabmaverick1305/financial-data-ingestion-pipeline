from __future__ import annotations

import pytest
from pydantic import ValidationError

from fies_parser.adapters.base import ParserAdapter
from fies_parser.adapters.capabilities import ParserCapabilities
from fies_parser.adapters.docling_adapter import DoclingAdapter
from fies_parser.adapters.pymupdf_adapter import PyMuPDFAdapter
from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.engine.models import ParseRequest


class _MinimalAdapter(ParserAdapter):
    name = "minimal"
    version = "1.0.0"

    def supports(self, request: ParseRequest) -> bool:
        return True

    def parse(self, request: ParseRequest) -> ParserCandidate:
        raise NotImplementedError


def test_default_capabilities_declare_nothing() -> None:
    capabilities = _MinimalAdapter().capabilities

    assert capabilities == ParserCapabilities()
    assert capabilities.supports_page_selection is False
    assert capabilities.supports_tables is False
    assert capabilities.supported_mime_types == ()


def test_pymupdf_capabilities() -> None:
    capabilities = PyMuPDFAdapter().capabilities

    assert capabilities.supports_page_selection is True
    assert capabilities.supports_tables is False
    assert capabilities.supports_ocr is False
    assert "application/pdf" in capabilities.supported_mime_types


def test_docling_capabilities_reflect_ocr_flag() -> None:
    assert DoclingAdapter(ocr=False).capabilities.supports_ocr is False
    assert DoclingAdapter(ocr=True).capabilities.supports_ocr is True
    assert DoclingAdapter().capabilities.supports_tables is True
    assert DoclingAdapter().capabilities.supports_headings is True


def test_capabilities_are_frozen() -> None:
    capabilities = ParserCapabilities()

    with pytest.raises(ValidationError):
        capabilities.supports_tables = True  # type: ignore[misc]
