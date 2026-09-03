"""Orchestrates Document Preflight -> Parser Routing Policy -> ParserEngine.

Candidate validation is not a separate step here — it already happens inside
`ParserEngine.run()`. `ParserRouter` picks exactly one parser and hands off;
it does not retry through the fallback parsers a `RoutingPolicy` may name.
That's fallback execution, a deliberately separate later layer.
"""

from __future__ import annotations

import structlog

from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.engine.models import ParseRequest
from fies_parser.engine.parser_engine import ParserEngine
from fies_parser.engine.registry import ParserRegistry
from fies_parser.preflight.document_profiler import DocumentProfiler
from fies_parser.routing.models import RoutingDecision
from fies_parser.routing.routing_policy import DefaultRoutingPolicy, RoutingPolicy
from fies_parser.routing.telemetry import RoutingTelemetry

log = structlog.get_logger()


class ParserRouter:
    def __init__(
        self,
        registry: ParserRegistry,
        engine: ParserEngine,
        profiler: DocumentProfiler | None = None,
        policy: RoutingPolicy | None = None,
        telemetry: RoutingTelemetry | None = None,
    ) -> None:
        self._registry = registry
        self._engine = engine
        self._profiler = profiler or DocumentProfiler()
        self._policy = policy or DefaultRoutingPolicy()
        self._telemetry = telemetry

    def route(self, request: ParseRequest) -> RoutingDecision:
        """Profile the document and decide which registered parser should
        handle it, without executing anything."""
        profile = self._profiler.profile(request.document)
        available = {name: self._registry.get(name).capabilities for name in self._registry.list_parsers()}
        decision = self._policy.decide(profile, available)

        log.info(
            "parser_router.decision",
            document_id=request.document.document_id,
            parser_name=decision.parser_name,
            reason=decision.reason,
            fallback_parser_names=decision.fallback_parser_names,
        )
        if self._telemetry is not None:
            self._telemetry.record_decision(request.document.document_id, decision)
        return decision

    def route_and_parse(self, request: ParseRequest) -> ParserCandidate:
        decision = self.route(request)
        candidate = self._engine.run(decision.parser_name, request)
        if self._telemetry is not None:
            self._telemetry.record_quality(candidate)
        return candidate
