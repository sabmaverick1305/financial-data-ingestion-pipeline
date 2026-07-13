"""Entity Reconciliation — the "Data-quality reconciliation" stage:

    Scheduled AMFI fetch -> S3 immutable snapshot -> AmfiCategoryResolver
        -> MFAPI live sync -> Entity resolution -> Canonical category relationship
        -> Data-quality reconciliation (this module)

entity_ingestion.ingest_scheme_plan()'s belongs_to logic only fills a GAP —
it creates a relationship if a scheme doesn't have one yet, but once one
exists it never revisits it, even when a stronger signal becomes available
later. That gap is real, not hypothetical: wiring services/
amfi_category_source.py into the live sync (see mf_ingestion/sync.py)
immediately made 2,895 previously-unresolvable scheme_plan rows resolvable,
but re-running ingest_scheme_plan across all of them created zero new
belongs_to edges — every one of those schemes already had a lower-confidence
edge set earlier via the scheme_name_fallback path (score 0.85), which
ingest_scheme_plan's "does an edge already exist?" check treats as fully
resolved and never upgrades.

This module is that upgrade path: for every scheme, re-derive the best
category signal available *right now* across all of its scheme_plan
children (majority vote, prioritized by source strength — amfi_navall >
primary > scheme_name_fallback, per resolve_category's own documented
signal ranking — not just by vote count), and supersede the active
belongs_to edge when a strictly higher-priority source is now available.

Superseding is soft (entity_store.deactivate_relationship + add_relationship),
never destructive — the old row is deactivated (is_active=False,
valid_to=today), not deleted, and a new active row takes over.

What this module deliberately does NOT do: auto-resolve a same-priority
disagreement (e.g. two amfi_navall-sourced votes across a scheme's plans
that disagree, or an existing amfi_navall edge that a fresh scheme_name_fallback
guess "disagrees" with). Those are real judgment calls, not upgrades — they
are counted and returned for a human to look at, matching this codebase's
established practice of surfacing disagreements rather than guessing.

This module also fixed a real, separate bug found while cleaning up after
the race condition above: entity_ingestion.ingest_scheme_plan's manages
check queried relationship_exists_for_source(source_entity_id=scheme, ...)
for a relationship stored as (organization -> scheme) — since scheme is
never the source of a manages edge, that check always returned None and
silently relied on add_relationship's exact-triple ON CONFLICT to avoid
duplicates, which only works when a scheme's sibling plans agree on the
AMC. Fixed by adding entity_store.relationship_exists_for_target and using
it for manages. dedupe_duplicate_active_relationships is deliberately
scoped to belongs_to only (see its own docstring) for the same
source-vs-target reason — it does NOT dedupe manages.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import structlog

from financial_pipeline.services import entity_store
from financial_pipeline.services.amfi_category_source import load_amfi_category_map
from financial_pipeline.services.entity_resolver import resolve_category

log = structlog.get_logger()

# Mirrors resolve_category's own documented signal-priority order (see
# services/entity_resolver.py's resolve_category docstring): AMFI's own
# NAVAll.txt category outranks mf_scheme_master.category (mfapi's own field,
# unreliable for a majority of rows) outranks the scheme_name pattern
# fallback. Both amfi_navall and primary score 1.0 in MatchResult — this
# ranking is what breaks that tie in the direction the domain rule intends,
# not raw score alone.
#
# "amfi" (not "amfi_navall") is a real, live value in production: 100% of
# the ~20,700 active belongs_to edges — every one of them — was written by
# the original one-off historical bulk backfill (scratchpad's
# populate_entity_relationship.py, predating this module and resolve_category
# entirely), which tagged its AMFI-sourced rows "amfi", not "amfi_navall".
# Discovered 2026-07-13 running reconcile_belongs_to for the first time
# against live data: _priority() didn't recognize "amfi", scored it 0 (below
# even scheme_name_fallback), and started silently "upgrading" every one of
# those 20,673 correctly-AMFI-sourced edges to the far less reliable
# "primary" (mfapi's own category field) tier — caught and killed after ~44
# rows via anomalous belongs_to_upgraded log lines (new_source=primary
# old_source=amfi is never a legitimate upgrade), nothing committed since
# reconcile_belongs_to's caller commits once at the end. "amfi" is kept as
# an explicit alias here rather than migrating the historical rows' data,
# so this function (and anyone reading it) has one place documenting both
# spellings mean the same thing.
_SOURCE_PRIORITY = {"amfi_navall": 3, "amfi": 3, "primary": 2, "scheme_name_fallback": 1}


def _priority(source: str | None) -> int:
    return _SOURCE_PRIORITY.get(source or "", 0)


@dataclass
class ReconciliationSummary:
    schemes_checked: int
    upgraded: int
    created: int
    disagreements: int
    unchanged: int


@dataclass
class DedupeSummary:
    sources_checked: int
    sources_with_duplicates: int
    edges_deactivated: int


def dedupe_duplicate_active_relationships(
    cur, relationship_type: str, *, conn=None, commit_every: int = 50
) -> DedupeSummary:
    """Collapse more than one simultaneously-active edge of relationship_type
    from the same source_entity_id down to exactly one — keeping the best
    by source priority (see _priority), then highest relationship_confidence,
    then earliest created_at (first-established wins ties).

    Only valid for belongs_to (scheme -> category): a scheme is the source
    there, and single-valued-per-source is the correct invariant to check
    ("does this scheme have exactly one category"). NOT valid for manages
    (organization -> scheme) — an organization is the source there and
    legitimately has many scheme targets (one AMC manages many schemes), so
    grouping by source_entity_id would treat an AMC's whole scheme
    portfolio as "duplicates" of each other. This function did exactly that
    once, live, against production data: 3,200 legitimate manages edges
    were deactivated in ~18 minutes before the bug was caught and every one
    of them restored. There was never a real manages duplication problem —
    checking the correct invariant (one manager per scheme, i.e. grouping
    by target_entity_id) found zero duplicates. If a real per-target dedupe
    is ever needed, it needs its own implementation, not this function with
    a different relationship_type argument.

    has_plan is also invalid here for the same reason as manages (legitimately
    multi-valued per scheme, one edge per plan variant) — nothing to dedupe.

    Pass conn to commit incrementally (every commit_every deactivated
    edges) rather than in one all-or-nothing transaction — this can be a
    few hundred individual UPDATE round trips over a cross-region RDS
    connection, and one long uncommitted transaction is fragile: a single
    network hiccup loses all progress and leaves a dangling idle-in-
    transaction connection holding locks (observed directly: a first
    attempt at this cleanup ran uninterrupted for 90+ minutes after a mid-
    run timeout, blocking every other writer, until the stale connection
    was manually terminated). Without conn, the caller owns committing
    (test/small-scale usage).

    A real, not hypothetical, one-time cleanup: uq_active_entity_relationship
    only blocks an *exact* (source, type, target) repeat, not a second
    different target for the same (source, type) — see add_relationship's
    docstring. entity_store.lock_scheme_for_relationship_write closes the
    race that causes this going forward; this function is for duplicates
    that already exist from before that lock was added (concretely: a
    20-worker parallel backfill produced 715 schemes with duplicate active
    belongs_to edges — see this docstring's manages warning above for why
    the originally-suspected 76 "duplicate manages edges" were never real).
    """
    if relationship_type != "belongs_to":
        raise ValueError(
            f"dedupe_duplicate_active_relationships groups by source_entity_id, "
            f"which is only a valid single-valued-per-source check for belongs_to. "
            f"Got {relationship_type!r} — see this function's docstring."
        )

    cur.execute(
        """
        SELECT relationship_id, source_entity_id, target_entity_id, source_system, relationship_confidence, created_at
        FROM financial_entity_relationship
        WHERE relationship_type = %s AND is_active = TRUE
        ORDER BY source_entity_id, created_at
        """,
        (relationship_type,),
    )
    by_source: dict = defaultdict(list)
    for relationship_id, source_entity_id, target_entity_id, source_system, relationship_confidence, created_at in cur.fetchall():
        by_source[source_entity_id].append((relationship_id, target_entity_id, source_system, relationship_confidence, created_at))

    edges_deactivated = 0
    sources_with_duplicates = 0

    for source_entity_id, edges in by_source.items():
        if len(edges) <= 1:
            continue
        sources_with_duplicates += 1

        def sort_key(edge):
            _, _, source_system, confidence, created_at = edge
            return (_priority(source_system), confidence or 0.0, -created_at.timestamp())

        edges_sorted = sorted(edges, key=sort_key, reverse=True)
        keep = edges_sorted[0]
        for edge in edges_sorted[1:]:
            entity_store.deactivate_relationship(cur, edge[0])
            edges_deactivated += 1
            if conn is not None and edges_deactivated % commit_every == 0:
                conn.commit()

        log.info(
            "entity_reconciliation.duplicate_active_relationship_deduped",
            source_entity_id=str(source_entity_id),
            relationship_type=relationship_type,
            kept_target=str(keep[1]),
            kept_source=keep[2],
            removed_count=len(edges) - 1,
        )

    if conn is not None:
        conn.commit()

    return DedupeSummary(
        sources_checked=len(by_source),
        sources_with_duplicates=sources_with_duplicates,
        edges_deactivated=edges_deactivated,
    )


def _load_scheme_plan_groups(cur) -> dict:
    """One query, not one-per-scheme: scheme_entity_id -> list of
    (scheme_code, scheme_name, category) for its scheme_plan children, via
    the has_plan relationship. Bulk-efficient — the same shape the original
    historical backfill used, now reused for reconciliation instead of
    only initial population."""
    cur.execute(
        """
        SELECT r.source_entity_id AS scheme_entity_id,
               m.metadata->>'scheme_code' AS scheme_code,
               m.canonical_name AS scheme_name,
               m.metadata->>'category' AS category
        FROM financial_entity_relationship r
        JOIN financial_entity_master m ON m.entity_id = r.target_entity_id
        WHERE r.relationship_type = 'has_plan' AND r.is_active = TRUE
        """
    )
    groups: dict = defaultdict(list)
    for scheme_entity_id, scheme_code, scheme_name, category in cur.fetchall():
        groups[scheme_entity_id].append((scheme_code, scheme_name, category))
    return groups


def _load_category_lookup(cur) -> dict[str, str]:
    """taxonomy leaf entity_id (entity_subtype) -> financial_entity_master
    category entity_id. One query for all ~39 leaves, reused across every
    scheme — calling entity_ingestion.ingest_category() (its own
    get_or_create_entity + add_identifier round trips) inside a per-scheme
    loop would repeat the same handful of lookups tens of thousands of
    times for no benefit; categories are already fully seeded."""
    cur.execute("SELECT entity_subtype, entity_id FROM financial_entity_master WHERE entity_type = 'category'")
    return {subtype: entity_id for subtype, entity_id in cur.fetchall()}


def _load_current_belongs_to(cur) -> dict:
    """scheme_entity_id -> entity_store.ActiveRelationship, one query for
    every active belongs_to edge — avoids a per-scheme round trip to check
    "what does this scheme currently have" before deciding whether to
    upgrade it."""
    cur.execute(
        """
        SELECT source_entity_id, relationship_id, target_entity_id, source_system, relationship_confidence
        FROM financial_entity_relationship
        WHERE relationship_type = 'belongs_to' AND is_active = TRUE
        """
    )
    result = {}
    for source_entity_id, relationship_id, target_entity_id, source_system, relationship_confidence in cur.fetchall():
        result[source_entity_id] = entity_store.ActiveRelationship(
            relationship_id=relationship_id,
            target_entity_id=target_entity_id,
            source_system=source_system,
            relationship_confidence=relationship_confidence,
        )
    return result


def _best_candidate(
    plans: list[tuple[str, str, str]],
    amfi_map: dict[str, str],
) -> tuple[str, str, float] | None:
    """The best (taxonomy_entity_id, source, score) across a scheme's plans,
    picking the highest-priority source tier first and, within that tier,
    the most common target (ties broken by encounter order)."""
    votes: Counter = Counter()
    best_by_target: dict[str, tuple[str, float]] = {}
    for scheme_code, scheme_name, category in plans:
        amfi_category = amfi_map.get(scheme_code)
        match = resolve_category(category, scheme_name=scheme_name, amfi_category=amfi_category)
        if match.entity_id is None:
            continue
        votes[(match.entity_id, match.source)] += 1
        # keep the highest score seen for this (entity_id, source) pair
        cur_best = best_by_target.get(match.entity_id)
        if cur_best is None or _priority(match.source) > _priority(cur_best[0]) or match.score > cur_best[1]:
            best_by_target[match.entity_id] = (match.source, match.score)

    if not votes:
        return None

    # Highest-priority source tier wins; within that tier, most votes wins.
    def sort_key(item):
        (entity_id, source), count = item
        return (_priority(source), count)

    (entity_id, source), _ = max(votes.items(), key=sort_key)
    score = best_by_target[entity_id][1]
    return entity_id, source, score


def reconcile_belongs_to(cur, amfi_map: dict[str, str] | None = None) -> ReconciliationSummary:
    """Re-derive and, where warranted, upgrade every scheme's belongs_to
    relationship using the best category signal available right now.

    Safe to run repeatedly (e.g. after any amfi_category_source snapshot
    refresh) — schemes whose current edge is already at the top priority
    tier, or where no stronger signal exists, are left untouched.
    """
    amfi_map = amfi_map if amfi_map is not None else load_amfi_category_map()
    groups = _load_scheme_plan_groups(cur)
    category_lookup = _load_category_lookup(cur)
    current_by_scheme = _load_current_belongs_to(cur)

    schemes_checked = upgraded = created = disagreements = unchanged = 0

    for scheme_entity_id, plans in groups.items():
        schemes_checked += 1
        candidate = _best_candidate(plans, amfi_map)
        if candidate is None:
            unchanged += 1
            continue
        candidate_entity_id, candidate_source, candidate_score = candidate

        cat_id = category_lookup.get(candidate_entity_id)
        if cat_id is None:
            unchanged += 1
            continue

        current = current_by_scheme.get(scheme_entity_id)

        if current is None:
            entity_store.add_relationship(
                cur,
                source_entity_id=scheme_entity_id,
                relationship_type="belongs_to",
                target_entity_id=cat_id,
                source_system=candidate_source,
                relationship_confidence=candidate_score,
            )
            created += 1
            continue

        if current.target_entity_id == cat_id:
            unchanged += 1
            continue

        if _priority(candidate_source) > _priority(current.source_system):
            entity_store.deactivate_relationship(cur, current.relationship_id)
            entity_store.add_relationship(
                cur,
                source_entity_id=scheme_entity_id,
                relationship_type="belongs_to",
                target_entity_id=cat_id,
                source_system=candidate_source,
                relationship_confidence=candidate_score,
            )
            log.info(
                "entity_reconciliation.belongs_to_upgraded",
                scheme_entity_id=str(scheme_entity_id),
                old_source=current.source_system,
                new_source=candidate_source,
            )
            upgraded += 1
        else:
            # Same or lower priority tier disagreeing with the existing
            # edge — a real judgment call, not an upgrade. Surface it,
            # don't guess.
            log.warning(
                "entity_reconciliation.belongs_to_disagreement",
                scheme_entity_id=str(scheme_entity_id),
                existing_source=current.source_system,
                existing_target=str(current.target_entity_id),
                candidate_source=candidate_source,
                candidate_target=str(cat_id),
            )
            disagreements += 1

    return ReconciliationSummary(
        schemes_checked=schemes_checked,
        upgraded=upgraded,
        created=created,
        disagreements=disagreements,
        unchanged=unchanged,
    )
