from __future__ import annotations

from fies_parser.canonical.candidate_models import ParserCandidate, ParserExecutionMetrics
from fies_parser.routing.models import RoutingDecision
from fies_parser.routing.telemetry import RoutingTelemetry


def test_record_decision_does_not_raise() -> None:
    decision = RoutingDecision(parser_name="pymupdf", reason="fast path", fallback_parser_names=("docling",))

    RoutingTelemetry().record_decision("doc-1", decision)


def test_record_quality_does_not_raise() -> None:
    candidate = ParserCandidate(
        document_id="doc-1",
        parser_name="pymupdf",
        parser_version="1.0.0",
        metrics=ParserExecutionMetrics(duration_seconds=0.5, pages_requested=2, pages_processed=2),
        warnings=["something odd"],
    )

    RoutingTelemetry().record_quality(candidate)


def test_record_agreement_returns_whether_parsers_match() -> None:
    telemetry = RoutingTelemetry()

    assert telemetry.record_agreement("doc-1", "pymupdf", "pymupdf") is True
    assert telemetry.record_agreement("doc-1", "pymupdf", "docling") is False
