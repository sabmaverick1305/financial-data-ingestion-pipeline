#!/usr/bin/env python3
"""ocr-worker — high memory (Docling OCR + layout + table-structure, CPU).

Picks documents at status=text_extracted with has_text_layer=False (scanned
PDFs) and runs Docling with OCR enabled. Run this as its own ECS task with a
large memory allocation — see docs/architecture.md.

Usage:
    python scripts/process_ocr_worker.py [--limit N]
"""
from __future__ import annotations

import sys

sys.path.insert(0, "src")

from _docling_worker import run  # noqa: E402
from financial_pipeline.processing.extractor import OcrExtractor  # noqa: E402
from financial_pipeline.storage.document_repo import Status  # noqa: E402

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--loop", action="store_true",
                        help="Keep processing until queue is empty, then exit")
    args = parser.parse_args()
    run("ocr-worker", "ocr_extraction", OcrExtractor, claim_status=Status.OCR_PROCESSING,
        has_text_layer=False, limit=args.limit, loop=args.loop)
