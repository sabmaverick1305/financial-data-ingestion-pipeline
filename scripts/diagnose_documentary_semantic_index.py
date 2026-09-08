#!/usr/bin/env python3
"""Forensic diagnosis of closed-beta documentary semantic indexing.

Produces conclusive evidence for every gate used by find_semantic_evidence:
identity -> document type -> processing status -> chunk rows -> semantic terms.
"""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from sqlalchemy import create_engine, text

from financial_pipeline.config import settings
from financial_pipeline.documentary.evidence_policy import (
    DEFAULT_BETA_REQUIREMENTS,
    REQUIREMENT_BY_KEY,
)
from financial_pipeline.intelligence.closed_beta_policy import ClosedBetaUniversePolicy


def main() -> None:
    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is required")

    policy = ClosedBetaUniversePolicy.default()
    engine = create_engine(settings.postgres_url, pool_pre_ping=True)
    families = sorted(policy.allowed_scheme_families)

    sql = """
        SELECT
            dsi.scheme_family_key,
            CAST(dm.document_id AS text) AS document_id,
            dm.file_name,
            dm.document_type,
            dm.processing_status,
            dm.has_text_layer,
            dm.attempt_count,
            dm.last_error,
            COUNT(dc.chunk_id) AS chunk_count,
            BOOL_OR(LOWER(COALESCE(dc.text, '')) LIKE '%investment objective%') AS has_investment_objective,
            BOOL_OR(LOWER(COALESCE(dc.text, '')) LIKE '%asset allocation%') AS has_asset_allocation,
            BOOL_OR(LOWER(COALESCE(dc.text, '')) LIKE '%investment strategy%') AS has_investment_strategy,
            BOOL_OR(LOWER(COALESCE(dc.text, '')) LIKE '%investment approach%') AS has_investment_approach,
            BOOL_OR(LOWER(COALESCE(dc.text, '')) LIKE '%portfolio%') AS has_portfolio,
            BOOL_OR(LOWER(COALESCE(dc.text, '')) LIKE '%holding%') AS has_holding,
            BOOL_OR(LOWER(COALESCE(dc.text, '')) LIKE '%sector allocation%') AS has_sector_allocation
        FROM document_scheme_identity dsi
        JOIN document_metadata dm
          ON dm.document_id = dsi.document_id
        LEFT JOIN document_chunks dc
          ON dc.document_id = dm.document_id
        WHERE dsi.scheme_family_key = ANY(:families)
        GROUP BY
            dsi.scheme_family_key,
            dm.document_id,
            dm.file_name,
            dm.document_type,
            dm.processing_status,
            dm.has_text_layer,
            dm.attempt_count,
            dm.last_error
        ORDER BY dsi.scheme_family_key, dm.document_type, dm.file_name
    """

    with engine.connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(text(sql), {"families": families}).mappings().all()
        ]

    report = []
    for row in rows:
        normalized_type = str(row["document_type"]).lower().replace(" ", "_")
        gates = {
            "identity_present": True,
            "status_searchable": row["processing_status"] in ("embedded", "indexed"),
            "chunks_present": int(row["chunk_count"] or 0) > 0,
        }

        requirement_results = {}
        for key in DEFAULT_BETA_REQUIREMENTS:
            req = REQUIREMENT_BY_KEY[key]
            type_match = normalized_type in req.accepted_document_types
            term_flags = []
            for term in req.content_terms:
                attr = "has_" + term.replace(" ", "_")
                term_flags.append(bool(row.get(attr)))
            terms_match = (
                all(term_flags)
                if req.match_mode == "all"
                else any(term_flags)
            ) if term_flags else True
            requirement_results[key] = {
                "document_type_match": type_match,
                "semantic_terms_match": terms_match,
                "would_match_find_semantic_evidence": (
                    gates["status_searchable"]
                    and gates["chunks_present"]
                    and type_match
                    and terms_match
                ),
            }

        if not gates["status_searchable"]:
            root_cause = "processing_status_not_embedded_or_indexed"
        elif not gates["chunks_present"]:
            root_cause = "no_document_chunks"
        elif not any(
            item["document_type_match"]
            for item in requirement_results.values()
        ):
            root_cause = "document_type_not_accepted"
        elif not any(
            item["semantic_terms_match"]
            for item in requirement_results.values()
        ):
            root_cause = "required_semantic_terms_missing"
        else:
            root_cause = "searchable"

        report.append({
            **row,
            "gates": gates,
            "requirements": requirement_results,
            "root_cause": root_cause,
        })

    summary: dict[str, int] = {}
    for item in report:
        cause = item["root_cause"]
        summary[cause] = summary.get(cause, 0) + 1

    print(json.dumps({
        "families": families,
        "documents": len(report),
        "root_cause_counts": summary,
        "report": report,
        "conclusion": (
            "A document can satisfy semantic coverage only when identity, accepted "
            "document type, embedded/indexed status, chunk rows, and required content "
            "terms all pass. root_cause_counts identifies the first failed predicate."
        ),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
