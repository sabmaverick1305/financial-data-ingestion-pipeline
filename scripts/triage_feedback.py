#!/usr/bin/env python3
"""Batch-triage query_log rows that haven't been triaged yet.

The /api/feedback endpoint already triages a row inline the moment a user
rates it (see feedback/triage.py + api/main.py's feedback()). This script
exists for the rows inline triage never reaches: a guardrail integrity flag
(hallucination_risk=high, invalid citation, inconsistent numbers) can make a
row worth reviewing even if the user never leaves feedback at all — most
beta users won't rate every answer. Run this periodically (cron / EventBridge,
same pattern as reset_stale_claims.py) to catch those.

Usage:
    python scripts/triage_feedback.py [--limit 500]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from financial_pipeline.config import settings  # noqa: E402
from financial_pipeline.feedback.triage import triage_pending  # noqa: E402
from financial_pipeline.storage.query_log import QueryLogRepository  # noqa: E402
from sqlalchemy import create_engine  # noqa: E402


def main(limit: int = 500) -> None:
    if not settings.postgres_url:
        print("POSTGRES_URL not set.")
        sys.exit(1)

    engine = create_engine(settings.postgres_url)
    query_log = QueryLogRepository(engine)
    query_log.create_tables()

    counts = triage_pending(query_log, limit=limit)

    if not counts:
        print("Nothing to triage.")
        return

    total = sum(counts.values())
    print(f"Triaged {total} row(s):")
    for category, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {category:30s} {n}")

    high_priority = query_log.fetch_for_review(priority="high", limit=5)
    if high_priority:
        print(f"\n{len(high_priority)} high-priority row(s) awaiting review (showing up to 5):")
        for row in high_priority:
            print(f"  [{row['query_id']}] {row['triage_category']}: {row['triage_reason']}")
            print(f"    Q: {row['question'][:100]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=500, help="Max rows to triage this run")
    args = parser.parse_args()
    main(limit=args.limit)
