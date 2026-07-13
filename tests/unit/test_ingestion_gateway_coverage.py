"""Guards the "entity resolution is mandatory" invariant structurally, since
there's no shared base class every ingestion pipeline extends (see
services/lineage.py's module docstring) that could enforce this at import
time instead.

services/entity_ingestion.py's ingest_scheme_plan / ingest_amc / ingest_category
mint scheme/AMC/category identity. Every pipeline that needs one of them
should go through services/lineage.py's resolve_and_link(..., ingest_fn=...),
which pairs the call with a mandatory ingestion_lineage row — never call
these functions directly from a new ingestion entry point, or that pipeline's
entity-resolution writes become untraceable and easy to accidentally skip
under a future refactor.

The only legitimate direct calls are internal to entity_ingestion.py itself
(ingest_scheme_plan composes ingest_amc/ingest_category as implementation
detail) — everything else must reference these functions only as a value
(e.g. passed to resolve_and_link's ingest_fn= parameter), never invoke them.
"""
from __future__ import annotations

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GUARDED_FUNCTIONS = {"ingest_scheme_plan", "ingest_amc", "ingest_category"}
ALLOWED_DIRECT_CALL_FILES = {
    REPO_ROOT / "src" / "financial_pipeline" / "services" / "entity_ingestion.py",
}
SCAN_ROOTS = [
    REPO_ROOT / "src" / "financial_pipeline",
    REPO_ROOT / "scripts",
]


def _direct_calls_to_guarded_functions(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None
        if name in GUARDED_FUNCTIONS:
            hits.append(name)
    return hits


def test_guarded_ingest_functions_are_only_called_through_the_gateway() -> None:
    violations: dict[str, list[str]] = {}
    for root in SCAN_ROOTS:
        for path in root.rglob("*.py"):
            if path in ALLOWED_DIRECT_CALL_FILES:
                continue
            hits = _direct_calls_to_guarded_functions(path)
            if hits:
                violations[str(path.relative_to(REPO_ROOT))] = hits

    assert not violations, (
        "Found direct calls to entity_ingestion's ingest_* functions outside "
        "the mandatory gateway (services/lineage.py's resolve_and_link). "
        "Route these through resolve_and_link(..., ingest_fn=<the function>) "
        f"instead of calling directly: {violations}"
    )
