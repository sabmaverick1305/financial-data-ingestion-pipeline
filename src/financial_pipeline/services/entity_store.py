"""Entity Store — real DB-backed primitives for the "Lookup existing entity",
"Create entity if needed", "Create identifier mapping", and "Create
relationships" stages of the entity resolution pipeline.

    Normalize names -> Lookup existing entity -> Create entity if needed
        -> Create identifier mapping -> Create relationships
     (canonical_name_          (this module)         (this module)
      normalizer.py)

Distinct from services/entity_resolver.py: that module resolves a raw value
(amc_name, category) to a canonical taxonomy entity_id by matching against
domain/semantic/ YAML data — it never touches the database. This module
takes that resolved identity and reads/writes the actual
financial_entity_master / financial_entity_identifier /
financial_entity_relationship rows in Postgres.

Every write here is idempotent against the same unique constraints the
original bulk backfill (scratchpad's populate_entity_{master,identifier,
relationship}.py — see domain/resolution/entity_resolution_rules.yaml's
"scope" sections) relied on, so this is safe to call repeatedly, e.g. once
per scheme on every mf_ingestion sync run, not just as a one-off bulk load:
  - financial_entity_master:       uq_active_entity_identity
                                    (entity_type, entity_subtype, normalized_name)
  - financial_entity_identifier:   uq_source_identifier
                                    (source_system, identifier_type, identifier_value)
  - financial_entity_relationship: uq_active_entity_relationship
                                    (source_entity_id, relationship_type, target_entity_id)

Callers own the psycopg2 connection/transaction (pass a live cursor in) so
one caller can batch several entity/identifier/relationship writes into a
single commit — see services/entity_ingestion.py's ingest_scheme_plan for
the intended usage.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass

import psycopg2
from psycopg2.extensions import cursor as Psycopg2Cursor

from financial_pipeline.config import settings
from financial_pipeline.storage.checkpointer import to_psycopg_dsn


def connect() -> psycopg2.extensions.connection:
    """A plain psycopg2 connection to the same database
    settings.postgres_url points at (SQLAlchemy driver suffix stripped —
    see to_psycopg_dsn). Caller controls commit/rollback and closing."""
    return psycopg2.connect(to_psycopg_dsn(settings.postgres_url))


@dataclass
class EntityRef:
    entity_id: uuid.UUID
    created: bool  # True if this call inserted a new financial_entity_master row


def lookup_entity(
    cur: Psycopg2Cursor,
    *,
    entity_type: str,
    entity_subtype: str | None,
    normalized_name: str,
) -> uuid.UUID | None:
    """Look up an existing financial_entity_master row by the natural key
    uq_active_entity_identity enforces, restricted to the lifecycle statuses
    that index covers (pending/active/renamed) — a merged/inactive/closed/
    archived row with the same natural key does not count as "existing" for
    resolution purposes, matching the partial unique index's own scope."""
    cur.execute(
        """
        SELECT entity_id FROM financial_entity_master
        WHERE entity_type = %s
          AND entity_subtype IS NOT DISTINCT FROM %s
          AND normalized_name = %s
          AND lifecycle_status IN ('pending', 'active', 'renamed')
        """,
        (entity_type, entity_subtype, normalized_name),
    )
    row = cur.fetchone()
    return row[0] if row else None


def get_or_create_entity(
    cur: Psycopg2Cursor,
    *,
    entity_type: str,
    entity_subtype: str | None,
    canonical_name: str,
    normalized_name: str,
    canonical_source: str | None = None,
    source_confidence: float | None = None,
    metadata: dict | None = None,
) -> EntityRef:
    """Idempotent get-or-create against financial_entity_master.

    Looks up first; if missing, INSERTs. The INSERT itself is also
    ON CONFLICT DO NOTHING + a re-SELECT so a race against a concurrent
    caller inserting the same natural key between our lookup and insert
    still resolves to the winning row rather than raising or silently
    losing metadata.
    """
    existing = lookup_entity(
        cur, entity_type=entity_type, entity_subtype=entity_subtype, normalized_name=normalized_name
    )
    if existing is not None:
        return EntityRef(entity_id=existing, created=False)

    cur.execute(
        """
        INSERT INTO financial_entity_master
            (entity_type, entity_subtype, canonical_name, normalized_name,
             canonical_source, source_confidence, metadata)
        VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
        ON CONFLICT (entity_type, entity_subtype, normalized_name)
            WHERE lifecycle_status IN ('pending', 'active', 'renamed')
        DO NOTHING
        RETURNING entity_id
        """,
        (
            entity_type,
            entity_subtype,
            canonical_name,
            normalized_name,
            canonical_source,
            source_confidence,
            json.dumps(metadata or {}),
        ),
    )
    row = cur.fetchone()
    if row is not None:
        return EntityRef(entity_id=row[0], created=True)

    # Lost the race to a concurrent insert — the row exists now.
    winner = lookup_entity(
        cur, entity_type=entity_type, entity_subtype=entity_subtype, normalized_name=normalized_name
    )
    if winner is None:
        raise RuntimeError(
            f"get_or_create_entity: INSERT conflicted but no matching row found for "
            f"({entity_type!r}, {entity_subtype!r}, {normalized_name!r}) — unexpected."
        )
    return EntityRef(entity_id=winner, created=False)


def add_identifier(
    cur: Psycopg2Cursor,
    *,
    entity_id: uuid.UUID,
    source_system: str,
    identifier_type: str,
    identifier_value: str,
    is_primary_for_source: bool = True,
    match_method: str = "deterministic",
    match_confidence: float | None = None,
    source_record_name: str | None = None,
) -> uuid.UUID | None:
    """Idempotent insert into financial_entity_identifier. Returns the new
    identifier_id, or None if (source_system, identifier_type,
    identifier_value) already maps to some entity — matching
    uq_source_identifier, this never silently repoints an existing mapping
    to a different entity_id."""
    cur.execute(
        """
        INSERT INTO financial_entity_identifier
            (entity_id, source_system, identifier_type, identifier_value,
             is_primary_for_source, match_method, match_confidence, source_record_name)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_system, identifier_type, identifier_value) DO NOTHING
        RETURNING identifier_id
        """,
        (
            entity_id,
            source_system,
            identifier_type,
            identifier_value,
            is_primary_for_source,
            match_method,
            match_confidence,
            source_record_name,
        ),
    )
    row = cur.fetchone()
    return row[0] if row else None


def relationship_exists_for_source(
    cur: Psycopg2Cursor,
    *,
    source_entity_id: uuid.UUID,
    relationship_type: str,
) -> uuid.UUID | None:
    """Does source_entity_id already have an active relationship_type edge
    (to any target)? Correct for keeping a relationship single-valued *per
    source* — e.g. belongs_to (scheme -> category): a scheme is the source,
    and should have exactly one category. NOT correct for manages
    (organization -> scheme): an organization is the source there, and
    legitimately has many scheme targets (one AMC manages many schemes) —
    use relationship_exists_for_target instead to check "does this scheme
    already have a manager". Mixing these up is a real bug this codebase
    hit once already (see entity_reconciliation.py's dedupe function,
    scoped to belongs_to only for the same reason)."""
    cur.execute(
        """
        SELECT target_entity_id FROM financial_entity_relationship
        WHERE source_entity_id = %s AND relationship_type = %s AND is_active = TRUE
        LIMIT 1
        """,
        (source_entity_id, relationship_type),
    )
    row = cur.fetchone()
    return row[0] if row else None


def relationship_exists_for_target(
    cur: Psycopg2Cursor,
    *,
    target_entity_id: uuid.UUID,
    relationship_type: str,
) -> uuid.UUID | None:
    """Does target_entity_id already have an active incoming relationship_type
    edge (from any source)? The correct single-valued-per-target check for
    manages (organization -> scheme): "does this scheme already have a
    manager", not "does this organization already manage something"."""
    cur.execute(
        """
        SELECT source_entity_id FROM financial_entity_relationship
        WHERE target_entity_id = %s AND relationship_type = %s AND is_active = TRUE
        LIMIT 1
        """,
        (target_entity_id, relationship_type),
    )
    row = cur.fetchone()
    return row[0] if row else None


@dataclass
class ActiveRelationship:
    relationship_id: uuid.UUID
    target_entity_id: uuid.UUID
    source_system: str | None
    relationship_confidence: float | None


def active_relationship_for_source(
    cur: Psycopg2Cursor,
    *,
    source_entity_id: uuid.UUID,
    relationship_type: str,
) -> ActiveRelationship | None:
    """Like relationship_exists_for_source, but returns the full row
    (source_system/relationship_confidence included) — what
    services/entity_reconciliation.py needs to decide whether an existing
    edge should be superseded by a newer, higher-confidence one."""
    cur.execute(
        """
        SELECT relationship_id, target_entity_id, source_system, relationship_confidence
        FROM financial_entity_relationship
        WHERE source_entity_id = %s AND relationship_type = %s AND is_active = TRUE
        LIMIT 1
        """,
        (source_entity_id, relationship_type),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return ActiveRelationship(relationship_id=row[0], target_entity_id=row[1], source_system=row[2], relationship_confidence=row[3])


def lock_scheme_for_relationship_write(cur: Psycopg2Cursor, scheme_entity_id: uuid.UUID) -> None:
    """Transaction-scoped advisory lock, keyed on scheme_entity_id — serializes
    concurrent writers of a scheme's manages/belongs_to edges.

    relationship_exists_for_source's "does an edge already exist?" check is
    a plain SELECT, not otherwise atomic with the INSERT that follows it —
    two threads processing different scheme_plan siblings of the same base
    scheme (mf_ingestion/sync.py's ThreadPoolExecutor routinely does this,
    and any bulk backfill run with more than one worker will too) can both
    see "no edge yet" and both insert, producing two simultaneously-active
    edges with different targets for one scheme. uq_active_entity_relationship
    only blocks an *exact* repeat (source, type, target) — a second,
    different target for the same (source, type) sails right through it
    (see add_relationship's docstring). Verified as a real, not just
    theoretical, failure mode: a 20-worker backfill run produced 715
    schemes with duplicate active belongs_to edges and 76 with duplicate
    manages edges before this lock existed.

    Call once per ingest_scheme_plan() invocation, right after the scheme
    entity is resolved, before any relationship_exists_for_source check.
    Released automatically at the caller's next COMMIT/ROLLBACK — every
    call site already commits shortly after calling ingest_scheme_plan.
    """
    cur.execute("SELECT pg_advisory_xact_lock(hashtext(%s::text)::bigint)", (str(scheme_entity_id),))


def deactivate_relationship(cur: Psycopg2Cursor, relationship_id: uuid.UUID) -> None:
    """Soft-supersede: marks a relationship inactive rather than deleting it,
    preserving history. uq_active_entity_relationship is WHERE is_active =
    TRUE, so a deactivated row no longer blocks inserting a new active edge
    for the same (source, type) — see entity_reconciliation.py's upgrade
    path, the intended caller."""
    cur.execute(
        "UPDATE financial_entity_relationship SET is_active = FALSE, valid_to = CURRENT_DATE, updated_at = now() "
        "WHERE relationship_id = %s",
        (relationship_id,),
    )


def add_relationship(
    cur: Psycopg2Cursor,
    *,
    source_entity_id: uuid.UUID,
    relationship_type: str,
    target_entity_id: uuid.UUID,
    source_system: str | None = None,
    relationship_confidence: float | None = None,
    is_inferred: bool = False,
) -> uuid.UUID | None:
    """Idempotent insert into financial_entity_relationship. Returns the new
    relationship_id, or None if this exact (source, type, target) triple is
    already an active edge (uq_active_entity_relationship).

    Note this constraint only dedupes an *exact* repeat — it does NOT stop a
    source_entity_id from acquiring two different targets for the same
    relationship_type (e.g. a scheme "belonging to" two different
    categories). For relationship types that should be single-valued per
    source (manages, belongs_to), callers should check
    relationship_exists_for_source first — see entity_ingestion.py's
    ingest_scheme_plan for the intended pattern.
    """
    cur.execute(
        """
        INSERT INTO financial_entity_relationship
            (source_entity_id, relationship_type, target_entity_id,
             source_system, relationship_confidence, is_inferred)
        VALUES (%s, %s, %s, %s, %s, %s)
        ON CONFLICT (source_entity_id, relationship_type, target_entity_id) WHERE is_active = TRUE
        DO NOTHING
        RETURNING relationship_id
        """,
        (source_entity_id, relationship_type, target_entity_id, source_system, relationship_confidence, is_inferred),
    )
    row = cur.fetchone()
    return row[0] if row else None
