"""Composition root for the full, multi-adapter Parser Engine.

This is the one place PyMuPDF and Docling get registered together. It is
deliberately separate from `parser_engine_integration.py`'s pinned
single-adapter (`pymupdf`-only) engine, which `TextExtractor` still calls
directly for the authoritative extraction result.

What this builds is used for shadow-routing execution and telemetry only —
nothing reads its output as the real answer yet. `ParserRouter` becomes the
real call site only after routing quality has been benchmarked and tuned
against representative FIES documents (see `scripts/benchmark_parser_routing.py`
and the migration notes in this module's callers).
"""

from __future__ import annotations

from fies_parser.adapters.docling_adapter import DoclingAdapter
from fies_parser.adapters.pymupdf_adapter import PyMuPDFAdapter
from fies_parser.engine.parser_engine import ParserEngine
from fies_parser.engine.registry import ParserRegistry
from fies_parser.routing.parser_router import ParserRouter
from fies_parser.routing.shadow_router import ShadowRouter
from fies_parser.routing.telemetry import RoutingTelemetry


def build_registry() -> ParserRegistry:
    registry = ParserRegistry()
    registry.register(PyMuPDFAdapter())
    registry.register(DoclingAdapter(ocr=False))
    return registry


def build_router(telemetry: RoutingTelemetry | None = None) -> ParserRouter:
    registry = build_registry()
    engine = ParserEngine(registry)
    return ParserRouter(registry, engine, telemetry=telemetry or RoutingTelemetry())


def build_shadow_router(sample_rate: float, execute: bool) -> ShadowRouter:
    """`sample_rate=0.0` (the default in `financial_pipeline.config.Settings`)
    makes every call a no-op — safe to construct unconditionally at import
    time in `parser_engine_integration.py`."""
    telemetry = RoutingTelemetry()
    router = build_router(telemetry=telemetry)
    return ShadowRouter(router=router, telemetry=telemetry, sample_rate=sample_rate, execute=execute)
