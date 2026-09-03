"""Telemetry for routing decisions and parser output quality.

Structured logging (structlog) is the primary channel — it's what every
other ingestion-path module in this codebase already relies on for
observability (ECS workers ship JSON logs; they don't run a scraped
`/metrics` endpoint the way the FastAPI API does). Prometheus counters/
histograms are also defined here, using the same `prometheus-client`
dependency the API already uses, so they're populated and ready to expose
the same way if an ingestion worker ever starts a metrics server — as of
this milestone nothing scrapes them.
"""

from __future__ import annotations

import structlog
from prometheus_client import Counter, Histogram

from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.routing.models import RoutingDecision

log = structlog.get_logger()

ROUTING_DECISIONS_TOTAL = Counter(
    "fies_parser_routing_decisions_total",
    "Routing decisions made by a RoutingPolicy, by chosen parser",
    ["parser_name"],
)
PARSE_DURATION_SECONDS = Histogram(
    "fies_parser_parse_duration_seconds",
    "ParserCandidate parse duration, by parser",
    ["parser_name"],
)
PARSE_WARNINGS_TOTAL = Counter(
    "fies_parser_parse_warnings_total",
    "Warnings emitted during a parser run, by parser",
    ["parser_name"],
)
ROUTING_AGREEMENT_TOTAL = Counter(
    "fies_parser_routing_agreement_total",
    "Shadow-routing agreement between the routed parser and the authoritative one",
    ["agrees"],
)


class RoutingTelemetry:
    """Records routing decisions and parser-output quality signals.

    Never raises — telemetry failures must not be able to break parsing or
    routing. `ParserRouter` and `ShadowRouter` call this unconditionally.
    """

    def record_decision(self, document_id: str, decision: RoutingDecision) -> None:
        ROUTING_DECISIONS_TOTAL.labels(parser_name=decision.parser_name).inc()
        log.info(
            "parser_telemetry.routing_decision",
            document_id=document_id,
            parser_name=decision.parser_name,
            reason=decision.reason,
            fallback_parser_names=decision.fallback_parser_names,
        )

    def record_quality(self, candidate: ParserCandidate) -> None:
        PARSE_DURATION_SECONDS.labels(parser_name=candidate.parser_name).observe(candidate.metrics.duration_seconds)
        if candidate.warnings:
            PARSE_WARNINGS_TOTAL.labels(parser_name=candidate.parser_name).inc(len(candidate.warnings))

        log.info(
            "parser_telemetry.parse_quality",
            document_id=candidate.document_id,
            parser_name=candidate.parser_name,
            parser_run_id=candidate.parser_run_id,
            duration=round(candidate.metrics.duration_seconds, 4),
            pages_processed=candidate.metrics.pages_processed,
            element_count=len(candidate.elements),
            table_count=len(candidate.tables),
            warning_count=len(candidate.warnings),
        )

    def record_agreement(self, document_id: str, authoritative_parser_name: str, routed_parser_name: str) -> bool:
        agrees = authoritative_parser_name == routed_parser_name
        ROUTING_AGREEMENT_TOTAL.labels(agrees=str(agrees).lower()).inc()
        log.info(
            "parser_telemetry.routing_agreement",
            document_id=document_id,
            authoritative_parser_name=authoritative_parser_name,
            routed_parser_name=routed_parser_name,
            agrees=agrees,
        )
        return agrees
