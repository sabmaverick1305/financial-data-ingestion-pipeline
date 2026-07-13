"""Entity Ingestion — orchestrates the full five-stage pipeline for one
scheme/AMC/category at a time:

    Normalize names -> Lookup existing entity -> Create entity if needed
        -> Create identifier mapping -> Create relationships

This is the reusable, incremental counterpart to scratchpad's
populate_entity_{master,identifier,relationship}.py, which ran this same
logic once, by hand, as a bulk backfill over all 37,659 mf_scheme_master
rows (see domain/resolution/entity_resolution_rules.yaml's "scope" sections
for that history — this module doesn't replace the bulk load, it's what
runs afterwards). The intended caller is one new scheme_code showing up on
a future mf_ingestion sync, or an operator re-running ingest_scheme_plan
over the full table to pick up rows the last bulk pass skipped (e.g. after
a taxonomy.yaml category gets added) — either way, every write is
idempotent, so calling this on an already-ingested scheme_code is a no-op.

Each ingest_* function commits its own transaction so a failure partway
through one scheme's ingestion (e.g. a bad category match) doesn't roll
back unrelated schemes already processed in the same batch.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass

import structlog

from financial_pipeline.semantic.semantic_engine import get_engine
from financial_pipeline.services import entity_store
from financial_pipeline.services.canonical_name_normalizer import normalize_name, parse_scheme_name
from financial_pipeline.services.entity_resolver import resolve_amc_name, resolve_category

log = structlog.get_logger()


def ingest_amc(cur, raw_amc_name: str) -> uuid.UUID | None:
    """Normalize -> lookup/create -> identifier for one AMC name.

    Resolves raw_amc_name to a canonical taxonomy.yaml amc_entity_id first
    (services/entity_resolver.resolve_amc_name — word-boundary synonym
    match). Returns None without writing anything if it doesn't resolve —
    per entity_resolution_rules.yaml's amc_name_to_entity_id known_gap, a
    newly-onboarded AMC with no taxonomy.yaml/thesaurus.yaml entry yet is a
    real, honest gap, not something to fabricate an entity for.
    """
    eng = get_engine()
    match = resolve_amc_name(raw_amc_name)
    if match.entity_id is None:
        log.warning("entity_ingestion.amc_unresolved", raw_amc_name=raw_amc_name)
        return None

    canonical_name = eng.amc_display_name(match.entity_id)
    if canonical_name is None:
        return None

    ref = entity_store.get_or_create_entity(
        cur,
        entity_type="organization",
        entity_subtype="amc",
        canonical_name=canonical_name,
        normalized_name=normalize_name(canonical_name),
        canonical_source="mfapi",
    )
    entity_store.add_identifier(
        cur,
        entity_id=ref.entity_id,
        source_system="fies_taxonomy",
        identifier_type="amc_entity_id",
        identifier_value=match.entity_id,
        match_method="deterministic",
        source_record_name=canonical_name,
    )
    return ref.entity_id


def ingest_category(cur, taxonomy_leaf_entity_id: str) -> uuid.UUID | None:
    """Normalize -> lookup/create -> identifier for one taxonomy.yaml
    mf_scheme_categories leaf. Unlike ingest_amc, the input here is already
    a resolved taxonomy entity_id (not a raw string needing resolution) —
    callers doing raw-category resolution should go through
    services/entity_resolver.resolve_category first and pass its
    match.entity_id here.
    """
    leaf = get_engine().mf_scheme_category(taxonomy_leaf_entity_id)
    if leaf is None:
        return None

    ref = entity_store.get_or_create_entity(
        cur,
        entity_type="category",
        entity_subtype=leaf["entity_id"],
        canonical_name=leaf["label"],
        normalized_name=normalize_name(leaf["label"]),
        canonical_source="amfi",
    )
    entity_store.add_identifier(
        cur,
        entity_id=ref.entity_id,
        source_system="fies_taxonomy",
        identifier_type="category_taxonomy_id",
        identifier_value=leaf["entity_id"],
        match_method="deterministic",
        source_record_name=leaf["label"],
    )
    return ref.entity_id


@dataclass
class SchemePlanIngestResult:
    scheme_entity_id: uuid.UUID
    scheme_plan_entity_id: uuid.UUID
    scheme_created: bool
    scheme_plan_created: bool
    amc_entity_id: uuid.UUID | None
    category_entity_id: uuid.UUID | None
    manages_relationship_created: bool
    belongs_to_relationship_created: bool


def ingest_scheme_plan(
    cur,
    *,
    scheme_code: str,
    scheme_name: str,
    amc_name: str | None,
    category: str | None,
    amfi_category: str | None = None,
) -> SchemePlanIngestResult:
    """Full five-stage pipeline for one mf_scheme_master row.

    1. Normalize   — parse_scheme_name() splits scheme_name into
                      base_fund_name/plan/option.
    2-3. Lookup/create — get_or_create the base "scheme" entity (keyed on
                      normalized base_fund_name, shared across all plan
                      variants of the same fund) and the "scheme_plan"
                      entity (keyed on the full normalized scheme_name).
    4. Identifier  — scheme_code as a mfapi identifier on the scheme_plan
                      entity.
    5. Relationships — has_plan (scheme -> scheme_plan, always); manages
                      (organization -> scheme) and belongs_to (scheme ->
                      category), each created only once per scheme via
                      relationship_exists_for_source, so a scheme's AMC/
                      category is fixed by whichever scheme_plan resolves
                      it first and later plans don't silently multiply
                      edges — see entity_store.add_relationship's
                      docstring. A later plan whose own resolution
                      disagrees with the scheme's already-set edge is
                      logged, not written (data-quality signal, matching
                      the batch script's "schemes with disagreeing child
                      AMCs" count).
    """
    eng = get_engine()
    parsed = parse_scheme_name(scheme_name)
    base_norm = normalize_name(parsed.base_fund_name)

    scheme_ref = entity_store.get_or_create_entity(
        cur,
        entity_type="scheme",
        entity_subtype="mutual_fund_scheme",
        canonical_name=parsed.base_fund_name,
        normalized_name=base_norm,
        canonical_source="mfapi",
    )

    # Serialize concurrent ingest_scheme_plan calls for sibling scheme_plans
    # of this same scheme — see lock_scheme_for_relationship_write's
    # docstring for the duplicate-active-edge race this prevents.
    entity_store.lock_scheme_for_relationship_write(cur, scheme_ref.entity_id)

    plan_subtype = f"{parsed.plan or 'unspecified'}_{parsed.option or 'unspecified'}"
    plan_ref = entity_store.get_or_create_entity(
        cur,
        entity_type="scheme_plan",
        entity_subtype=plan_subtype,
        canonical_name=scheme_name,
        normalized_name=normalize_name(scheme_name),
        canonical_source="mfapi",
        metadata={
            "scheme_code": scheme_code,
            "amc_name": amc_name,
            "category": category,
            "plan": parsed.plan,
            "option": parsed.option,
        },
    )

    entity_store.add_identifier(
        cur,
        entity_id=plan_ref.entity_id,
        source_system="mfapi",
        identifier_type="scheme_code",
        identifier_value=scheme_code,
        match_method="exact_identifier",
        source_record_name=scheme_name,
    )
    entity_store.add_relationship(
        cur,
        source_entity_id=scheme_ref.entity_id,
        relationship_type="has_plan",
        target_entity_id=plan_ref.entity_id,
        source_system="mfapi",
    )

    # amc_entity_id/category_entity_id always end up reflecting the scheme's
    # actual current belongs_to/manages target, whether that was already set
    # by an earlier sibling scheme_plan, newly set by this call, or still
    # unresolved — not just "what this one row's own fields resolved to".
    # A base fund's category/AMC is scheme-level, so a plan variant whose own
    # category text is garbage (e.g. mfapi's flat "Income" value) should
    # still report its scheme's already-known category rather than None.
    amc_entity_id = entity_store.relationship_exists_for_target(
        cur, target_entity_id=scheme_ref.entity_id, relationship_type="manages"
    )
    manages_created = False
    if amc_name:
        amc_match = resolve_amc_name(amc_name)
        if amc_match.entity_id:
            org_entity_id = ingest_amc(cur, amc_name)
            if org_entity_id:
                if amc_entity_id is None:
                    rel_id = entity_store.add_relationship(
                        cur,
                        source_entity_id=org_entity_id,
                        relationship_type="manages",
                        target_entity_id=scheme_ref.entity_id,
                        source_system="mfapi",
                    )
                    manages_created = rel_id is not None
                    amc_entity_id = org_entity_id
                elif amc_entity_id != org_entity_id:
                    log.warning(
                        "entity_ingestion.amc_disagreement",
                        scheme_entity_id=str(scheme_ref.entity_id),
                        scheme_code=scheme_code,
                        existing_amc_entity_id=str(amc_entity_id),
                        new_amc_entity_id=str(org_entity_id),
                    )
        else:
            log.warning("entity_ingestion.amc_unresolved", scheme_code=scheme_code, amc_name=amc_name)

    category_entity_id = entity_store.relationship_exists_for_source(
        cur, source_entity_id=scheme_ref.entity_id, relationship_type="belongs_to"
    )
    belongs_to_created = False
    cat_match = resolve_category(category, scheme_name=scheme_name, amfi_category=amfi_category)
    if cat_match.entity_id:
        cat_entity_id = ingest_category(cur, cat_match.entity_id)
        if cat_entity_id:
            if category_entity_id is None:
                rel_id = entity_store.add_relationship(
                    cur,
                    source_entity_id=scheme_ref.entity_id,
                    relationship_type="belongs_to",
                    target_entity_id=cat_entity_id,
                    source_system=cat_match.source,
                    relationship_confidence=cat_match.score,
                )
                belongs_to_created = rel_id is not None
                category_entity_id = cat_entity_id
            elif category_entity_id != cat_entity_id:
                log.warning(
                    "entity_ingestion.category_disagreement",
                    scheme_entity_id=str(scheme_ref.entity_id),
                    scheme_code=scheme_code,
                    existing_category_entity_id=str(category_entity_id),
                    new_category_entity_id=str(cat_entity_id),
                )
    else:
        log.warning("entity_ingestion.category_unresolved", scheme_code=scheme_code, category=category)

    return SchemePlanIngestResult(
        scheme_entity_id=scheme_ref.entity_id,
        scheme_plan_entity_id=plan_ref.entity_id,
        scheme_created=scheme_ref.created,
        scheme_plan_created=plan_ref.created,
        amc_entity_id=amc_entity_id,
        category_entity_id=category_entity_id,
        manages_relationship_created=manages_created,
        belongs_to_relationship_created=belongs_to_created,
    )
