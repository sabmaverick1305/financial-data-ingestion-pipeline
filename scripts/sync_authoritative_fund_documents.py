#!/usr/bin/env python3
"""Discover + ingest authoritative mutual-fund documents from reviewed AMC sources.

Manifest format:
{
  "funds": [
    {
      "scheme_family_key": "hsbc midcap fund",
      "scheme_code": "151036",
      "provider": "HSBC Mutual Fund",
      "source": "amc_official",
      "authoritative_domain": "example-amc.com",
      "source_page_url": "https://example-amc.com/fund-page",
      "documents": [
        {
          "source_url": "https://example-amc.com/docs/hsbc-midcap-sid.pdf",
          "document_type": "scheme_information_document",
          "file_name": "hsbc-midcap-sid.pdf",
          "title": "HSBC Midcap Fund Scheme Information Document"
        }
      ]
    }
  ]
}

If "documents" is omitted or empty, links are discovered from source_page_url.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import urlparse

sys.path.insert(0, "src")

from financial_pipeline.config import settings
from financial_pipeline.documentary.authoritative_ingestion import (
    AuthoritativeFundDocument,
    AuthoritativeFundDocumentIngestor,
)
from financial_pipeline.documentary.source_discovery import (
    AuthoritativeSourcePageDiscoverer,
)
from financial_pipeline.storage.document_repo import DocumentRepository


def _document_from_item(fund: dict, item: dict) -> AuthoritativeFundDocument:
    source_url = str(item["source_url"])
    file_name = str(
        item.get("file_name")
        or PurePosixPath(urlparse(source_url).path).name
        or "fund-document.pdf"
    )
    return AuthoritativeFundDocument(
        scheme_family_key=str(fund["scheme_family_key"]),
        scheme_code=(
            str(fund["scheme_code"])
            if fund.get("scheme_code") is not None
            else None
        ),
        provider=str(fund["provider"]),
        source=str(fund.get("source") or "amc_official"),
        source_url=source_url,
        authoritative_domain=str(fund["authoritative_domain"]),
        document_type=str(item["document_type"]),
        file_name=file_name,
        title=item.get("title"),
        publication_date=item.get("publication_date"),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--discover-only",
        action="store_true",
        help="Discover official document links but do not download/ingest them.",
    )
    args = parser.parse_args()

    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is required")
    if not args.discover_only and not settings.s3_bucket:
        raise RuntimeError("S3_BUCKET is required")

    payload = json.loads(args.manifest.read_text())
    funds = payload.get("funds") if isinstance(payload, dict) else None
    if not isinstance(funds, list):
        raise ValueError("manifest must contain a 'funds' array")

    repository = DocumentRepository(settings.postgres_url)
    repository.create_tables()
    discoverer = AuthoritativeSourcePageDiscoverer()
    ingestor = AuthoritativeFundDocumentIngestor(repository)

    discovered_output = []
    ingested_output = []

    for fund in funds:
        documents = list(fund.get("documents") or [])
        if not documents:
            source_page = fund.get("source_page_url")
            if not source_page:
                raise ValueError(
                    f"{fund.get('scheme_family_key')}: source_page_url or documents is required"
                )
            discovered = discoverer.discover(
                source_page_url=str(source_page),
                authoritative_domain=str(fund["authoritative_domain"]),
                scheme_family_key=str(fund["scheme_family_key"]),
            )
            documents = [
                {
                    "source_url": item.url,
                    "document_type": item.document_type,
                    "title": item.anchor_text or None,
                }
                for item in discovered
            ]

        discovered_output.append({
            "scheme_family_key": fund["scheme_family_key"],
            "documents": documents,
        })

        if args.discover_only:
            continue

        for item in documents:
            result = ingestor.ingest(_document_from_item(fund, item))
            ingested_output.append(result)

    print(json.dumps({
        "funds": len(funds),
        "discovered": discovered_output,
        "ingested_count": len(ingested_output),
        "ingested": ingested_output,
        "next_steps": [] if args.discover_only else [
            "Run process_text_worker.py for uploaded documents.",
            "Run process_table_worker.py / process_ocr_worker.py as required.",
            "Run process_chunk_worker.py.",
            "Run process_embed_worker.py.",
            "Run audit_documentary_coverage.py and confirm semantic_documentary_ready=true.",
            "Run run_reasoning_production.py and confirm overall=GO.",
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
