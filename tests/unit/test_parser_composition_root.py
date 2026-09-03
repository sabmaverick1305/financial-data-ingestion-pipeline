from __future__ import annotations

from fies_parser.routing.parser_router import ParserRouter
from fies_parser.routing.shadow_router import ShadowRouter
from financial_pipeline.processing.parser_composition_root import (
    build_registry,
    build_router,
    build_shadow_router,
)


def test_build_registry_registers_both_adapters() -> None:
    registry = build_registry()

    assert set(registry.list_parsers()) == {"pymupdf", "docling"}


def test_build_registry_returns_independent_instances() -> None:
    registry_a = build_registry()
    registry_b = build_registry()

    assert registry_a is not registry_b
    assert registry_a.get("pymupdf") is not registry_b.get("pymupdf")


def test_build_router_returns_a_working_router() -> None:
    router = build_router()

    assert isinstance(router, ParserRouter)


def test_build_shadow_router_wires_sample_rate_and_execute_flag() -> None:
    shadow = build_shadow_router(sample_rate=0.25, execute=True)

    assert isinstance(shadow, ShadowRouter)
    assert shadow._sample_rate == 0.25
    assert shadow._execute is True
