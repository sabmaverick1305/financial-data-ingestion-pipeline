#!/usr/bin/env python3
"""Reset stale claim-state documents whose workers died without reverting.

Run this:
  - On every worker startup (already wired into _worker_common.init())
  - Every 5 minutes via AWS EventBridge Scheduler (recommended)
  - Manually when investigating a stuck pipeline

A document is considered stale when its processing_status is a transient
claim state (text_processing / table_processing / ocr_processing /
chunk_processing) AND its claim_expires_at timestamp is in the past.
claim_expires_at is set to NOW() + 30 minutes at claim time by
claim_documents(), so a worker has 30 minutes to either advance the status or
fail gracefully before the row is eligible for reset.

Stale rows are reverted to the appropriate pending status:
  text_processing  -> uploaded
  table_processing -> text_extracted
  ocr_processing   -> text_extracted
  chunk_processing -> tables_extracted

This script is safe to run concurrently — the UPDATE is atomic. Multiple
callers won't double-reset the same row.

Usage:
    python scripts/reset_stale_claims.py [--dry-run] [--report]

Exit codes:
    0 — completed (including "nothing to reset")
    1 — error connecting to DB
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from financial_pipeline.config import settings  # noqa: E402
from financial_pipeline.storage.document_repo import DocumentRepository  # noqa: E402


def main(dry_run: bool = False, report: bool = False) -> None:
    if not settings.postgres_url:
        print("POSTGRES_URL not set.")
        sys.exit(1)

    repo = DocumentRepository(settings.postgres_url)

    if report:
        # Show full pipeline health without making any changes
        depths = repo.queue_depths()
        print("=== Pipeline queue depths ===")
        for stage, count in depths.items():
            print(f"  {stage:<20} {count}")

        failed = repo.get_failed_documents(limit=20)
        if failed:
            print(f"\n=== Failed documents ({len(failed)} shown, most recent first) ===")
            for doc in failed:
                print(f"  {doc['file_name']:<40} attempts={doc['attempt_count']}  "
                      f"error={str(doc['last_error'])[:80]}")
        else:
            print("\nNo failed documents.")

    reset = repo.reset_stale_claims(dry_run=dry_run)

    if not reset:
        print("No stale claims found." + (" (dry-run)" if dry_run else ""))
        return

    action = "Would reset" if dry_run else "Reset"
    print(f"{action} {len(reset)} stale claim(s):")
    for row in reset:
        print(f"  {row['file_name']:<40} {row['processing_status']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be reset without modifying the DB")
    parser.add_argument("--report", action="store_true",
                        help="Also print queue depths and failed document summary")
    args = parser.parse_args()
    main(dry_run=args.dry_run, report=args.report)
