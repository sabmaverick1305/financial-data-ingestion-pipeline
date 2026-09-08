#!/usr/bin/env python3
"""Build a reviewed ingestion manifest from official AMC source pages.

Input JSON format:
[
  {
    "scheme_family_key": "hsbc midcap fund",
    "scheme_code": "151036",
    "provider": "HSBC Mutual Fund",
    "source_page_url": "https://<official-amc-domain>/<fund-page>",
    "authoritative_domain": "<official-amc-domain>"
  }
]

The output is intentionally a manifest, not automatic ingestion. This keeps an
explicit review boundary between link discovery and evidence admission.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, "src")

from financial_pipeline.documentary.source_discovery import AuthoritativeSourcePageDiscoverer


def _file_name(url: str, scheme_family_key: str, document_type: str) -> str:
    name = Path(urlparse(url).path).name
    if name and "." in name:
        return name
    return f"{scheme_family_key.replace(' ', '-')}-{document_type}.pdf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sources", type=Path)
    parser.add_argument("--output", type=Path, default=Path("authoritative-fund-documents.json"))
    args = parser.parse_args()

    sources = json.loads(args.sources.read_text())
    if not isinstance(sources, list):
        raise ValueError("sources must be a JSON array")

    discoverer = AuthoritativeSourcePageDiscoverer()
    manifest = []
    unresolved = []
    for source in sources:
        discovered = discoverer.discover(
            source_page_url=source["source_page_url"],
            authoritative_domain=source["authoritative_domain"],
            scheme_family_key=source["scheme_family_key"],
        )
        if not discovered:
            unresolved.append(source["scheme_family_key"])
            continue
        for item in discovered:
            manifest.append({
                "scheme_family_key": source["scheme_family_key"],
                "scheme_code": source.get("scheme_code"),
                "provider": source["provider"],
                "source": "amc_official",
                "source_url": item.url,
                "authoritative_domain": source["authoritative_domain"],
                "document_type": item.document_type,
                "file_name": _file_name(item.url, source["scheme_family_key"], item.document_type),
                "title": item.anchor_text or None,
            })

    args.output.write_text(json.dumps(manifest, indent=2))
    print(json.dumps({
        "manifest": str(args.output),
        "documents_discovered": len(manifest),
        "schemes_without_discovered_documents": unresolved,
        "review_required": True,
        "next_step": f"Review {args.output} and pass it to scripts/ingest_authoritative_fund_documents.py",
    }, indent=2))


if __name__ == "__main__":
    main()
