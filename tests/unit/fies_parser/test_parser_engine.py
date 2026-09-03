from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from fies_parser.adapters.base import ParserAdapter
from fies_parser.canonical.candidate_models import ParserCandidate, ParserExecutionMetrics
from fies_parser.engine.exceptions import ParserExecutionError, UnsupportedDocumentError
from fies_parser.engine.models import ParseRequest, SourceDocument
from fies_parser.engine.parser_engine import ParserEngine, compute_configuration_hash
from fies_parser.engine.registry import ParserRegistry


class _StubAdapter(ParserAdapter):
    name = "stub"
    version = "1.0.0"

    def __init__(self, supports_result: bool = True, should_fail: bool = False) -> None:
        self._supports_result = supports_result
        self._should_fail = should_fail

    def supports(self, request: ParseRequest) -> bool:
        return self._supports_result

    def parse(self, request: ParseRequest) -> ParserCandidate:
        if self._should_fail:
            raise RuntimeError("boom")
        return ParserCandidate(
            document_id=request.document.document_id,
            parser_name=self.name,
            parser_version=self.version,
            metrics=ParserExecutionMetrics(pages_requested=1, pages_processed=1),
        )


@pytest.fixture
def sample_document(tmp_path: Path) -> SourceDocument:
    file_path = tmp_path / "sample.pdf"
    file_path.write_bytes(b"%PDF-1.4 fake content")
    return SourceDocument(
        document_id="doc-1",
        file_path=file_path,
        file_name="sample.pdf",
        mime_type="application/pdf",
        file_hash=hashlib.sha256(file_path.read_bytes()).hexdigest(),
        source="test",
    )


def _engine_with(adapter: ParserAdapter) -> ParserEngine:
    registry = ParserRegistry()
    registry.register(adapter)
    return ParserEngine(registry)


def test_successful_execution(sample_document: SourceDocument) -> None:
    engine = _engine_with(_StubAdapter())
    request = ParseRequest(document=sample_document)

    candidate = engine.run("stub", request)

    assert candidate.document_id == "doc-1"
    assert candidate.metrics.pages_processed == 1


def test_unsupported_mime_type_raises(sample_document: SourceDocument) -> None:
    engine = _engine_with(_StubAdapter(supports_result=False))
    request = ParseRequest(document=sample_document)

    with pytest.raises(UnsupportedDocumentError):
        engine.run("stub", request)


def test_configuration_hash_is_deterministic() -> None:
    hash_a = compute_configuration_hash("stub", "1.0.0", {"a": 1, "b": 2})
    hash_b = compute_configuration_hash("stub", "1.0.0", {"b": 2, "a": 1})  # different insertion order

    assert hash_a == hash_b


def test_configuration_hash_changes_with_configuration() -> None:
    hash_a = compute_configuration_hash("stub", "1.0.0", {"a": 1})
    hash_b = compute_configuration_hash("stub", "1.0.0", {"a": 2})

    assert hash_a != hash_b


def test_adapter_failure_is_wrapped(sample_document: SourceDocument) -> None:
    engine = _engine_with(_StubAdapter(should_fail=True))
    request = ParseRequest(document=sample_document)

    with pytest.raises(ParserExecutionError) as excinfo:
        engine.run("stub", request)

    assert excinfo.value.parser_name == "stub"
    assert excinfo.value.document_id == "doc-1"
    assert isinstance(excinfo.value.original_error, RuntimeError)


def test_parser_metadata_is_assigned(sample_document: SourceDocument) -> None:
    engine = _engine_with(_StubAdapter())
    request = ParseRequest(document=sample_document, configuration={"x": 1})

    candidate = engine.run("stub", request)

    assert candidate.parser_name == "stub"
    assert candidate.parser_version == "1.0.0"
    assert candidate.parser_run_id
    assert candidate.configuration_hash == compute_configuration_hash("stub", "1.0.0", {"x": 1})


def test_duration_is_assigned(sample_document: SourceDocument) -> None:
    engine = _engine_with(_StubAdapter())
    request = ParseRequest(document=sample_document)

    candidate = engine.run("stub", request)

    assert candidate.metrics.duration_seconds >= 0
    assert isinstance(candidate.metrics.duration_seconds, float)
