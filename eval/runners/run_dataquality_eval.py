"""Phase 6: Data Quality evaluation — scores the entity graph itself (Postgres
schema state), not model behavior against a query corpus like phases 1-5.

Each "query_id" here is a check id (DQ001..DQ009, defined in
eval/corpus/dataquality_checks.json) rather than a natural-language query.
Reuses the query patterns services/entity_reconciliation.py and
services/entity_store.py already established (duplicate detection grouped
by the correct side per relationship type, the is_active=TRUE filter
convention) instead of writing new ad-hoc SQL — see each _check_* function's
docstring for which existing module it mirrors.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CHECKS = ROOT / "eval/corpus/dataquality_checks.json"


def _check_duplicate_belongs_to(cur) -> tuple[float, int]:
    """Mirrors services/entity_reconciliation.py's
    dedupe_duplicate_active_relationships query, grouped by source_entity_id
    (the correct single-valued side for belongs_to), read-only."""
    cur.execute(
        """
        SELECT count(*) FROM (
          SELECT source_entity_id FROM financial_entity_relationship
          WHERE relationship_type = 'belongs_to' AND is_active = TRUE
          GROUP BY source_entity_id HAVING count(*) > 1
        ) dup
        """
    )
    n = cur.fetchone()[0]
    return float(n), n


def _check_duplicate_manages(cur) -> tuple[float, int]:
    """Same shape as _check_duplicate_belongs_to, but grouped by
    target_entity_id — the correct single-valued side for manages (an AMC
    legitimately manages many schemes; a scheme should have exactly one
    manager). See entity_reconciliation.py's module docstring for the
    incident this distinction exists to prevent."""
    cur.execute(
        """
        SELECT count(*) FROM (
          SELECT target_entity_id FROM financial_entity_relationship
          WHERE relationship_type = 'manages' AND is_active = TRUE
          GROUP BY target_entity_id HAVING count(*) > 1
        ) dup
        """
    )
    n = cur.fetchone()[0]
    return float(n), n


def _check_orphan_scheme_plan(cur) -> tuple[float, int]:
    """scheme_plan entities with no active has_plan edge pointing to them —
    should never happen given ingest_scheme_plan creates both atomically,
    but a partial/failed transaction could leave one."""
    cur.execute(
        """
        SELECT count(*) FROM financial_entity_master m
        WHERE m.entity_type = 'scheme_plan'
          AND NOT EXISTS (
            SELECT 1 FROM financial_entity_relationship r
            WHERE r.target_entity_id = m.entity_id
              AND r.relationship_type = 'has_plan' AND r.is_active = TRUE
          )
        """
    )
    n = cur.fetchone()[0]
    return float(n), n


def _ratio_missing_relationship(cur, *, relationship_type: str, side: str) -> tuple[float, int]:
    """side='target' checks entity_id as the relationship's target (manages);
    side='source' checks entity_id as the source (belongs_to) — same
    source-vs-target distinction entity_store.relationship_exists_for_source/
    _for_target encode."""
    column = "target_entity_id" if side == "target" else "source_entity_id"
    cur.execute(
        f"""
        SELECT
          count(*) FILTER (WHERE NOT EXISTS (
            SELECT 1 FROM financial_entity_relationship r
            WHERE r.{column} = m.entity_id
              AND r.relationship_type = %s AND r.is_active = TRUE
          )) AS missing,
          count(*) AS total
        FROM financial_entity_master m
        WHERE m.entity_type = 'scheme' AND m.lifecycle_status IN ('pending', 'active', 'renamed')
        """,
        (relationship_type,),
    )
    missing, total = cur.fetchone()
    return (missing / total if total else 0.0), missing


def _check_schemes_missing_manages(cur) -> tuple[float, int]:
    return _ratio_missing_relationship(cur, relationship_type="manages", side="target")


def _check_schemes_missing_belongs_to(cur) -> tuple[float, int]:
    return _ratio_missing_relationship(cur, relationship_type="belongs_to", side="source")


def _check_relationship_completeness(cur) -> tuple[float, int]:
    cur.execute(
        """
        SELECT
          count(*) FILTER (WHERE
            EXISTS (SELECT 1 FROM financial_entity_relationship r1
                    WHERE r1.target_entity_id = m.entity_id
                      AND r1.relationship_type = 'manages' AND r1.is_active = TRUE)
            AND EXISTS (SELECT 1 FROM financial_entity_relationship r2
                        WHERE r2.source_entity_id = m.entity_id
                          AND r2.relationship_type = 'belongs_to' AND r2.is_active = TRUE)
          ) AS complete,
          count(*) AS total
        FROM financial_entity_master m
        WHERE m.entity_type = 'scheme' AND m.lifecycle_status IN ('pending', 'active', 'renamed')
        """
    )
    complete, total = cur.fetchone()
    return (complete / total if total else 0.0), (total - complete)


def _check_identifier_mapping_coverage(cur) -> tuple[float, int]:
    cur.execute(
        """
        SELECT
          count(*) FILTER (WHERE EXISTS (
            SELECT 1 FROM financial_entity_identifier i WHERE i.entity_id = m.entity_id
          )) AS covered,
          count(*) AS total
        FROM financial_entity_master m
        WHERE m.entity_type = 'scheme_plan' AND m.lifecycle_status IN ('pending', 'active', 'renamed')
        """
    )
    covered, total = cur.fetchone()
    return (covered / total if total else 0.0), (total - covered)


def _check_nav_freshness(cur) -> tuple[float, int]:
    """Fraction of *recently-trading* schemes whose NAV has unexpectedly
    gone stale beyond 4 days.

    Population is deliberately restricted to schemes with a latest_nav_date
    within the last 90 days, not every is_active=TRUE scheme. A live
    incident investigation (2026-07-13) found ~77% of "active" scheme_codes
    in mf_scheme_master are long-dormant/closed-ended/matured funds mfapi.in
    still lists but hasn't published a NAV for in over a year (28,682 of
    29,019 no-new-data schemes were >1 year stale, not newly broken) — the
    daily sync attempts every one of them without fail (0% stale on
    last_attempt_at), it just correctly finds nothing new to report. Scoring
    freshness against that population made this check permanently red for a
    reason that had nothing to do with pipeline health. Restricting to
    "recently active" schemes turns this into what it should measure: has a
    scheme that WAS trading recently suddenly stopped updating — a real,
    actionable signal — instead of permanently flagging schemes that are
    simply dormant and were never going to look "fresh" again.
    """
    cur.execute(
        """
        SELECT
          count(*) FILTER (WHERE s.latest_nav_date < CURRENT_DATE - INTERVAL '4 days') AS stale,
          count(*) AS total
        FROM mf_scheme_master ms
        JOIN mf_scheme_sync_status s ON s.scheme_code = ms.scheme_code
        WHERE ms.is_active = TRUE
          AND s.latest_nav_date >= CURRENT_DATE - INTERVAL '90 days'
        """
    )
    stale, total = cur.fetchone()
    return (stale / total if total else 0.0), stale


def _check_sync_freshness(cur) -> tuple[float, int]:
    """Fraction of active schemes the daily sync hasn't *attempted* in the
    last 30 hours (one cron cycle plus buffer) — measures pipeline
    operational health, not data availability.

    Deliberately checks last_attempt_at, not last_success_at.
    mf_ingestion/repository.py's upsert_sync_status only stamps
    last_success_at when sync_status=='success' (mfapi returned new NAV
    rows); a scheme with no new data in the lookback window is correctly
    classified sync_status='no_data' and stamped to last_failed_at instead
    — which is not a failure, just nothing new to report. Checking
    last_success_at made this check flag ~77% of schemes as "not synced in
    30h" even though the 2026-07-13 CloudWatch log investigation confirmed
    every single one of them was attempted, every single day, without
    exception. last_attempt_at is the column that's actually true to "did
    the pipeline run for this scheme" — see DQ008 above for the sibling
    fix to the same root confusion (data availability vs. pipeline health).
    """
    cur.execute(
        """
        SELECT
          count(*) FILTER (WHERE s.last_attempt_at IS NULL OR s.last_attempt_at < NOW() - INTERVAL '30 hours') AS stale,
          count(*) AS total
        FROM mf_scheme_master ms
        JOIN mf_scheme_sync_status s ON s.scheme_code = ms.scheme_code
        WHERE ms.is_active = TRUE
        """
    )
    stale, total = cur.fetchone()
    return (stale / total if total else 0.0), stale


_CHECK_FUNCS = {
    "DQ001": _check_duplicate_belongs_to,
    "DQ002": _check_duplicate_manages,
    "DQ003": _check_orphan_scheme_plan,
    "DQ004": _check_schemes_missing_manages,
    "DQ005": _check_schemes_missing_belongs_to,
    "DQ006": _check_relationship_completeness,
    "DQ007": _check_identifier_mapping_coverage,
    "DQ008": _check_nav_freshness,
    "DQ009": _check_sync_freshness,
}


def _passes(kind: str, value: float, threshold: float) -> bool:
    if kind in ("count_max", "ratio_max"):
        return value <= threshold
    if kind == "ratio_min":
        return value >= threshold
    raise ValueError(f"Unknown check kind: {kind}")


def run(query_ids: list[str] | None = None) -> dict:
    from financial_pipeline.services import entity_store

    checks = json.loads(CHECKS.read_text())
    target_ids = [qid for qid in checks if (not query_ids or qid in query_ids)]

    results = []
    errors = []

    conn = entity_store.connect()
    try:
        cur = conn.cursor()
        for check_id in target_ids:
            spec = checks[check_id]
            func = _CHECK_FUNCS.get(check_id)
            if func is None:
                errors.append({"id": check_id, "error": "no check function registered"})
                continue
            try:
                value, detail_count = func(cur)
                results.append(
                    {
                        "id": check_id,
                        "description": spec["description"],
                        "kind": spec["kind"],
                        "threshold": spec["threshold"],
                        "measured_value": value,
                        "detail_count": detail_count,
                        "passed": _passes(spec["kind"], value, spec["threshold"]),
                    }
                )
            except Exception as exc:
                conn.rollback()
                errors.append({"id": check_id, "error": str(exc)})
        cur.close()
    finally:
        conn.close()

    from eval.metrics import dataquality_metrics

    metrics = dataquality_metrics.summary(results)

    return {
        "phase": "data_quality",
        "n_queries": len(results),
        "n_errors": len(errors),
        "metrics": metrics,
        "errors": errors,
        "results": results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 6: Data Quality eval")
    parser.add_argument("--ids", nargs="*", help="Subset of check IDs (e.g. DQ001 DQ006)")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run(args.ids)
    output = json.dumps(result, indent=2, default=str)
    print(output)
    if args.out:
        Path(args.out).write_text(output)
