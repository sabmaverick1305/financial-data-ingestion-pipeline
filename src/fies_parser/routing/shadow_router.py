"""Shadow-mode routing: runs `ParserRouter` alongside an authoritative parser
call for comparison, never affecting the caller's result.

Existing purely for observation before `ParserRouter` becomes the real call
site. Two hard guarantees:

1. It never raises — a shadow-execution failure must not be able to break
   the real extraction path it's shadowing. Failures are logged and
   swallowed.
2. It is sampled — Docling is expensive, and running it on every document
   just to compare against PyMuPDF would multiply ingestion cost/latency.

This runs synchronously on the calling thread. That is a known limitation
for a production rollout at volume (see migration notes) — moving shadow
execution off the hot path (a separate worker/queue) is future work, not
something this milestone's environment can stand up.
"""

from __future__ import annotations

import random
import time

import structlog

from fies_parser.engine.models import ParseRequest
from fies_parser.routing.parser_router import ParserRouter
from fies_parser.routing.telemetry import RoutingTelemetry

log = structlog.get_logger()


class ShadowRouter:
    def __init__(
        self,
        router: ParserRouter,
        telemetry: RoutingTelemetry,
        sample_rate: float = 0.0,
        execute: bool = False,
        rng: random.Random | None = None,
    ) -> None:
        if not 0.0 <= sample_rate <= 1.0:
            raise ValueError(f"sample_rate must be in [0.0, 1.0], got {sample_rate}")

        self._router = router
        self._telemetry = telemetry
        self._sample_rate = sample_rate
        self._execute = execute
        self._rng = rng or random.Random()

    def maybe_shadow(self, request: ParseRequest, authoritative_parser_name: str) -> None:
        """Best-effort: decide (and optionally execute) the routed parser for
        comparison against `authoritative_parser_name`. Never raises."""
        if self._sample_rate <= 0.0 or self._rng.random() > self._sample_rate:
            return

        document_id = request.document.document_id

        if not self._execute:
            try:
                decision = self._router.route(request)
            except Exception as exc:
                log.warning("shadow_router.decision_failed", document_id=document_id, error=str(exc))
                return
            self._telemetry.record_agreement(document_id, authoritative_parser_name, decision.parser_name)
            return

        try:
            start = time.perf_counter()
            candidate = self._router.route_and_parse(request)
            log.info(
                "shadow_router.execution_completed",
                document_id=document_id,
                parser_name=candidate.parser_name,
                duration=round(time.perf_counter() - start, 4),
            )
        except Exception as exc:
            log.warning("shadow_router.execution_failed", document_id=document_id, error=str(exc))
            return

        self._telemetry.record_agreement(document_id, authoritative_parser_name, candidate.parser_name)
