"""Relationship Service — loads and queries domain/entity_model/relationship_types.yaml
and entity_types.yaml.

Metadata-level only, matching what these YAML files actually describe:
which entity relates to which, cardinality, and whether Postgres enforces
it via a real FK today. This does NOT fetch related rows from the database
— e.g. it can tell you has_nav_history is a real 1:N FK from mf_scheme to
nav_observation, but it does not run
`SELECT * FROM mf_nav_history WHERE scheme_code = ...` for you. A live
DB-backed traversal layer is a natural next step but is out of scope here
(the same boundary domain/semantic/semantic_engine.py draws: definitions
vs. the code that consumes them are kept separate).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import yaml

from financial_pipeline.semantic.semantic_engine import DOMAIN_DIR


@lru_cache(maxsize=1)
def _relationship_docs() -> dict:
    path = DOMAIN_DIR / "entity_model" / "relationship_types.yaml"
    return yaml.safe_load(path.read_text())


@lru_cache(maxsize=1)
def _entity_docs() -> dict:
    path = DOMAIN_DIR / "entity_model" / "entity_types.yaml"
    return yaml.safe_load(path.read_text())


@dataclass
class EntityType:
    entity_id: str
    label: str
    primary_table: str | None
    primary_key: str | list[str] | None
    source_systems: list[str]
    has_lifecycle: bool


@dataclass
class RelationshipType:
    relationship_id: str
    subject: str
    object: str
    cardinality: str
    enforced: bool
    description: str


def entity_type(entity_id: str) -> EntityType | None:
    for e in _entity_docs()["entities"]:
        if e["entity_id"] == entity_id:
            return EntityType(
                entity_id=e["entity_id"],
                label=e["label"],
                primary_table=e.get("primary_table"),
                primary_key=e.get("primary_key"),
                source_systems=e.get("source_systems", []),
                has_lifecycle=e.get("has_lifecycle", False),
            )
    return None


def all_entity_types() -> list[EntityType]:
    return [entity_type(e["entity_id"]) for e in _entity_docs()["entities"]]


def relationship(relationship_id: str) -> RelationshipType | None:
    for r in _relationship_docs()["relationships"]:
        if r["relationship_id"] == relationship_id:
            return RelationshipType(
                relationship_id=r["relationship_id"],
                subject=r["subject"],
                object=r["object"],
                cardinality=r["cardinality"],
                enforced=r["enforced"],
                description=r["description"],
            )
    return None


def relationships_for(entity_id: str) -> list[RelationshipType]:
    """Every relationship where entity_id is the subject — e.g.
    relationships_for("mf_scheme") returns issued_by, has_nav_history,
    has_performance_snapshot, classified_as."""
    return [
        relationship(r["relationship_id"])
        for r in _relationship_docs()["relationships"]
        if r["subject"] == entity_id
    ]


def enforced_relationships() -> list[RelationshipType]:
    """Relationships backed by a real Postgres FK today — the subset safe
    to assume is always consistent, vs. resolved-at-query-time relationships
    like issued_by (see domain/resolution/entity_resolution_rules.yaml's
    amc_name_to_entity_id for why issued_by isn't FK-enforced)."""
    return [
        relationship(r["relationship_id"])
        for r in _relationship_docs()["relationships"]
        if r["enforced"]
    ]
