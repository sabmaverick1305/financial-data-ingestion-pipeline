#!/usr/bin/env python3
"""Audit processing state of authoritative closed-beta documentary corpus."""
from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from sqlalchemy import create_engine, text

from financial_pipeline.config import settings
from financial_pipeline.intelligence.closed_beta_policy import ClosedBetaUniversePolicy


def main() -> None:
    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is required")

    policy = ClosedBetaUniversePolicy.default()
    engine = create_engine(settings.postgres_url, pool_pre_ping=True)

    sql = """
        SELECT
            dsi.scheme_family_key,
            dm.document_id,
            dm.file_name,
            dm.file_type,
            dm.document_type,
            dm.processing_status,
            dm.has_text_layer,
            dm.attempt_count,
            dm.last_error,
            dm.s3_raw_key,
            dm.s3_processed_key
        FROM document_scheme_identity dsi
        JOIN document_metadata dm
          ON dm.document_id = dsi.document_id
        WHERE dsi.scheme_family_key = ANY(:families)
        ORDER BY dsi.scheme_family_key, dm.document_type, dm.file_name
    """
    with engine.connect() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                text(sql),
                {"families": sorted(policy.allowed_scheme_families)},
            ).mappings().all()
        ]

    by_status: dict[str, int] = {}
    by_family: dict[str, list[dict]] = {}
    for row in rows:
        status = str(row.get("processing_status"))
        by_status[status] = by_status.get(status, 0) + 1
        by_family.setdefault(str(row["scheme_family_key"]), []).append(row)

    anomalies = []
    for row in rows:
        status = row.get("processing_status")
        file_type = str(row.get("file_type") or "").lower()
        has_text_layer = row.get("has_text_layer")
        if status == "text_extracted" and has_text_layer is None:
            anomalies.append({
                "document_id": str(row["document_id"]),
                "file_name": row["file_name"],
                "reason": "text_extracted_with_null_has_text_layer",
            })
        if status == "failed":
            anomalies.append({
                "document_id": str(row["document_id"]),
                "file_name": row["file_name"],
                "reason": "failed",
                "last_error": row.get("last_error"),
            })
        if file_type in ("html", "htm") and status == "text_extracted":
            anomalies.append({
                "document_id": str(row["document_id"]),
                "file_name": row["file_name"],
                "reason": "html_should_have_advanced_to_tables_extracted",
            })

    print(json.dumps({
        "document_count": len(rows),
        "status_counts": by_status,
        "anomalies": anomalies,
        "families": by_family,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
