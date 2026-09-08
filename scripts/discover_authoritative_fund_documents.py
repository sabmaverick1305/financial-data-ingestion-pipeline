#!/usr/bin/env python3
"""Discover candidate authoritative scheme-document links from an official AMC/AMFI page."""
from __future__ import annotations

import argparse
import json
import sys

sys.path.insert(0, "src")

from financial_pipeline.documentary.source_discovery import AuthoritativeSourcePageDiscoverer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--scheme-family-key", required=True)
    args = parser.parse_args()

    discovered = AuthoritativeSourcePageDiscoverer().discover(
        source_page_url=args.page,
        authoritative_domain=args.domain,
        scheme_family_key=args.scheme_family_key,
    )
    print(json.dumps([
        {
            "source_url": item.url,
            "document_type": item.document_type,
            "anchor_text": item.anchor_text,
        }
        for item in discovered
    ], indent=2))


if __name__ == "__main__":
    main()
