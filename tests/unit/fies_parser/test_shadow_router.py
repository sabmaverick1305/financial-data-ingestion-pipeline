from __future__ import annotations

from pathlib import Path

import pytest

from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.engine.models import ParseRequest
from fies_parser.routing.models import RoutingDecision
from fies_parser.routing.shadow_router import ShadowRouter

from .conftest import make_source_document


class _FakeRng:
    def __init__(self, value: float) -> None:
        self._value = value

    def random(self) -> float:
        return self._value


class _FakeRouter:
    def __init__(self, decision: RoutingDecision | None = None, candidate: ParserCandidate | None = None) -> None:
        self.route_calls = 0
        self.route_and_parse_calls = 0
        self._decision = decision
        self._candidate = candidate
        self.route_error: Exception | None = None
        self.execute_error: Exception | None = None

    def route(self, request: ParseRequest) -> RoutingDecision:
        self.route_calls += 1
        if self.route_error:
            raise self.route_error
        assert self._decision is not None
        return self._decision

    def route_and_parse(self, request: ParseRequest) -> ParserCandidate:
        self.route_and_parse_calls += 1
        if self.execute_error:
            raise self.execute_error
        assert self._candidate is not None
        return self._candidate


class _FakeTelemetry:
    def __init__(self) -> None:
        self.agreement_calls: list[tuple[str, str, str]] = []

    def record_agreement(self, document_id: str, authoritative_parser_name: str, routed_parser_name: str) -> bool:
        self.agreement_calls.append((document_id, authoritative_parser_name, routed_parser_name))
        return authoritative_parser_name == routed_parser_name


@pytest.fixture
def request_(tmp_path: Path) -> ParseRequest:
    file_path = tmp_path / "doc.pdf"
    file_path.write_bytes(b"%PDF-1.4")
    return ParseRequest(document=make_source_document(file_path))


def test_invalid_sample_rate_raises() -> None:
    with pytest.raises(ValueError):
        ShadowRouter(router=_FakeRouter(), telemetry=_FakeTelemetry(), sample_rate=1.5)


def test_zero_sample_rate_is_a_no_op(request_: ParseRequest) -> None:
    router = _FakeRouter()
    telemetry = _FakeTelemetry()
    shadow = ShadowRouter(router=router, telemetry=telemetry, sample_rate=0.0)

    shadow.maybe_shadow(request_, authoritative_parser_name="pymupdf")

    assert router.route_calls == 0
    assert telemetry.agreement_calls == []


def test_decision_only_mode_calls_route_not_route_and_parse(request_: ParseRequest) -> None:
    decision = RoutingDecision(parser_name="docling", reason="test")
    router = _FakeRouter(decision=decision)
    telemetry = _FakeTelemetry()
    shadow = ShadowRouter(router=router, telemetry=telemetry, sample_rate=1.0, execute=False, rng=_FakeRng(0.0))

    shadow.maybe_shadow(request_, authoritative_parser_name="pymupdf")

    assert router.route_calls == 1
    assert router.route_and_parse_calls == 0
    assert telemetry.agreement_calls == [("doc-1", "pymupdf", "docling")]


def test_execute_mode_calls_route_and_parse(request_: ParseRequest) -> None:
    candidate = ParserCandidate(document_id="doc-1", parser_name="docling", parser_version="1.0.0")
    router = _FakeRouter(candidate=candidate)
    telemetry = _FakeTelemetry()
    shadow = ShadowRouter(router=router, telemetry=telemetry, sample_rate=1.0, execute=True, rng=_FakeRng(0.0))

    shadow.maybe_shadow(request_, authoritative_parser_name="pymupdf")

    assert router.route_and_parse_calls == 1
    assert telemetry.agreement_calls == [("doc-1", "pymupdf", "docling")]


def test_decision_failure_is_swallowed(request_: ParseRequest) -> None:
    router = _FakeRouter()
    router.route_error = RuntimeError("boom")
    telemetry = _FakeTelemetry()
    shadow = ShadowRouter(router=router, telemetry=telemetry, sample_rate=1.0, execute=False, rng=_FakeRng(0.0))

    shadow.maybe_shadow(request_, authoritative_parser_name="pymupdf")  # must not raise

    assert telemetry.agreement_calls == []


def test_execution_failure_is_swallowed(request_: ParseRequest) -> None:
    router = _FakeRouter()
    router.execute_error = RuntimeError("boom")
    telemetry = _FakeTelemetry()
    shadow = ShadowRouter(router=router, telemetry=telemetry, sample_rate=1.0, execute=True, rng=_FakeRng(0.0))

    shadow.maybe_shadow(request_, authoritative_parser_name="pymupdf")  # must not raise

    assert telemetry.agreement_calls == []


def test_sampling_skips_when_rng_exceeds_sample_rate(request_: ParseRequest) -> None:
    decision = RoutingDecision(parser_name="docling", reason="test")
    router = _FakeRouter(decision=decision)
    telemetry = _FakeTelemetry()
    shadow = ShadowRouter(router=router, telemetry=telemetry, sample_rate=0.5, execute=False, rng=_FakeRng(0.9))

    shadow.maybe_shadow(request_, authoritative_parser_name="pymupdf")

    assert router.route_calls == 0
