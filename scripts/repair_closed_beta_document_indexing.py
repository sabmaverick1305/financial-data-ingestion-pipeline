#!/usr/bin/env python3
"""Repair authoritative beta documents stranded after successful text extraction.

Safe and idempotent: only promotes authoritative fund documents that already
have a usable text layer from text_extracted -> tables_extracted. Failed or
scanned documents are never promoted.
"""
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
        UPDATE document_metadata dm
           SET processing_status = 'tables_extracted',
               claim_expires_at = NULL,
               last_error = NULL,
               updated_at = NOW()
          FROM document_scheme_identity dsi
         WHERE dsi.document_id = dm.document_id
           AND dsi.scheme_family_key = ANY(:families)
           AND dm.processing_status = 'text_extracted'
           AND dm.has_text_layer IS TRUE
           AND dm.s3_raw_key LIKE 'bronze/fund_documents/%'
        RETURNING dm.document_id, dm.file_name, dsi.scheme_family_key
    """
    with engine.begin() as conn:
        rows = [
            dict(row)
            for row in conn.execute(
                text(sql),
                {"families": sorted(policy.allowed_scheme_families)},
            ).mappings().all()
        ]

    print(json.dumps({
        "promoted_to_tables_extracted": len(rows),
        "documents": rows,
        "next_step": (
            "Run process_chunk_worker.py --loop and then "
            "process_embed_worker.py --loop."
        ),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
