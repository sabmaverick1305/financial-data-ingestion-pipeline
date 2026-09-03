from __future__ import annotations

import pytest

from fies_parser.adapters.base import ParserAdapter
from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.engine.exceptions import DuplicateParserError, UnknownParserError
from fies_parser.engine.models import ParseRequest
from fies_parser.engine.registry import ParserRegistry


class _StubAdapter(ParserAdapter):
    name = "stub"
    version = "1.0.0"

    def supports(self, request: ParseRequest) -> bool:
        return True

    def parse(self, request: ParseRequest) -> ParserCandidate:
        raise NotImplementedError


def test_register_and_get_adapter() -> None:
    registry = ParserRegistry()
    adapter = _StubAdapter()

    registry.register(adapter)

    assert registry.get("stub") is adapter


def test_duplicate_registration_raises() -> None:
    registry = ParserRegistry()
    registry.register(_StubAdapter())

    with pytest.raises(DuplicateParserError) as excinfo:
        registry.register(_StubAdapter())

    assert excinfo.value.parser_name == "stub"


def test_unknown_parser_raises() -> None:
    registry = ParserRegistry()

    with pytest.raises(UnknownParserError) as excinfo:
        registry.get("does-not-exist")

    assert excinfo.value.parser_name == "does-not-exist"


def test_list_parsers() -> None:
    registry = ParserRegistry()
    registry.register(_StubAdapter())

    assert registry.list_parsers() == ("stub",)
