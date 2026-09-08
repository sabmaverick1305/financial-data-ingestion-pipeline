#!/usr/bin/env python3
"""Ingest authoritative mutual-fund documents from a reviewed JSON manifest.

Manifest format:
[
  {
    "scheme_family_key": "hsbc midcap fund",
    "scheme_code": "151036",
    "provider": "HSBC Mutual Fund",
    "source": "amc_official",
    "source_url": "https://<official-domain>/...",
    "authoritative_domain": "<official-domain>",
    "document_type": "scheme_information_document",
    "file_name": "hsbc-midcap-sid.pdf",
    "title": "HSBC Midcap Fund Scheme Information Document"
  }
]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from financial_pipeline.config import settings
from financial_pipeline.documentary.authoritative_ingestion import (
    AuthoritativeFundDocument,
    AuthoritativeFundDocumentIngestor,
)
from financial_pipeline.storage.document_repo import DocumentRepository


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is required")
    if not settings.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")

    payload = json.loads(args.manifest.read_text())
    if not isinstance(payload, list):
        raise ValueError("manifest must be a JSON array")

    repository = DocumentRepository(settings.postgres_url)
    repository.create_tables()
    ingestor = AuthoritativeFundDocumentIngestor(repository)

    results = []
    for item in payload:
        document = AuthoritativeFundDocument(**item)
        results.append(ingestor.ingest(document))

    print(json.dumps({
        "ingested": len(results),
        "results": results,
        "next_step": (
            "Run the existing text/table/chunk/embed workers until these document_ids "
            "reach embedded/indexed status, then rerun semantic documentary coverage."
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
