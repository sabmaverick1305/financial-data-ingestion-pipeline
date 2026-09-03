"""Policy for picking a registered parser given a document's preflight profile.

Deliberately separate from `ParserRouter`: the policy is pure decision logic
(profile + available capabilities in, one `RoutingDecision` out) with no I/O
and no knowledge of the registry/engine, so it's trivial to unit test and to
swap out without touching orchestration.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fies_parser.adapters.capabilities import ParserCapabilities
from fies_parser.engine.exceptions import NoParserAvailableError
from fies_parser.preflight.models import DocumentProfile
from fies_parser.routing.models import RoutingDecision


class RoutingPolicy(ABC):
    @abstractmethod
    def decide(self, profile: DocumentProfile, available: dict[str, ParserCapabilities]) -> RoutingDecision:
        """Pick a parser for `profile` from `available` (name -> capabilities).

        Raises `NoParserAvailableError` if nothing registered can handle it.
        """


class DefaultRoutingPolicy(RoutingPolicy):
    """Mirrors `extractor.py`'s existing two-stage production shape: a fast
    text-only parser handles the common case, a table-capable parser is used
    when nothing faster is registered *or* preflight thinks the document
    likely has tables, and a scanned document (no text layer) requires an
    OCR-capable parser specifically. Ties are broken by parser name for
    determinism.

    The `likely_has_tables` rule is intentionally a plain if/else, not a
    scored threshold — preflight only checks *sampled* pages, so it's a
    coarse "probably" signal. Tightening this into something scored belongs
    to `scripts/benchmark_parser_routing.py` once it has run against
    representative documents, not a guess made here.
    """

    def decide(self, profile: DocumentProfile, available: dict[str, ParserCapabilities]) -> RoutingDecision:
        if not available:
            raise NoParserAvailableError(profile.document_id, "no parsers are registered")

        candidates = {name: caps for name, caps in available.items() if profile.mime_type in caps.supported_mime_types}
        if not candidates:
            raise NoParserAvailableError(
                profile.document_id, f"no registered parser declares support for mime type {profile.mime_type!r}"
            )

        if not profile.has_text_layer:
            return self._route_scanned_document(profile, candidates)
        return self._route_text_document(profile, candidates)

    def _route_scanned_document(self, profile: DocumentProfile, candidates: dict[str, ParserCapabilities]) -> RoutingDecision:
        ocr_capable = sorted(name for name, caps in candidates.items() if caps.supports_ocr)
        if not ocr_capable:
            raise NoParserAvailableError(
                profile.document_id, "document has no text layer (appears scanned) but no OCR-capable parser is registered"
            )
        return RoutingDecision(
            parser_name=ocr_capable[0],
            reason="document has no text layer; routed to an OCR-capable parser",
            fallback_parser_names=tuple(ocr_capable[1:]),
        )

    def _route_text_document(self, profile: DocumentProfile, candidates: dict[str, ParserCapabilities]) -> RoutingDecision:
        fast = sorted(name for name, caps in candidates.items() if not caps.supports_tables)
        table_capable = sorted(name for name, caps in candidates.items() if caps.supports_tables)

        if profile.likely_has_tables and table_capable:
            return RoutingDecision(
                parser_name=table_capable[0],
                reason="preflight detected a likely table on a sampled page; routed to a table-capable parser",
                fallback_parser_names=tuple(table_capable[1:]) + tuple(fast),
            )

        if fast:
            return RoutingDecision(
                parser_name=fast[0],
                reason="document has a text layer; routed to a fast text parser",
                fallback_parser_names=tuple(fast[1:]) + tuple(table_capable),
            )

        return RoutingDecision(
            parser_name=table_capable[0],
            reason="document has a text layer; no fast text-only parser registered, routed to a table-capable parser",
            fallback_parser_names=tuple(table_capable[1:]),
        )
