#!/usr/bin/env python3
"""table-worker — high memory (Docling layout + table-structure, CPU).

Picks documents at status=text_extracted with has_text_layer=True (native
PDFs) and runs Docling without OCR to extract tables, figures, and markdown.
Run this as its own ECS task with a large memory allocation (e.g. 16GB+) —
see docs/architecture.md for why this stage was split out from the rest.

Usage:
    python scripts/process_table_worker.py [--limit N]
"""

from __future__ import annotations

import sys

sys.path.insert(0, "src")

from _docling_worker import run  # noqa: E402

from financial_pipeline.processing.extractor import TableExtractor  # noqa: E402
from financial_pipeline.storage.document_repo import Status  # noqa: E402

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--loop", action="store_true", help="Keep processing until queue is empty, then exit")
    args = parser.parse_args()
    run(
        "table-worker",
        "table_extraction",
        TableExtractor,
        claim_status=Status.TABLE_PROCESSING,
        has_text_layer=True,
        limit=args.limit,
        loop=args.loop,
    )
