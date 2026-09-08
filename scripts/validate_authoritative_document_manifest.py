#!/usr/bin/env python3
"""Validate structural semantic evidence coverage before authoritative ingestion."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, "src")

from financial_pipeline.documentary.evidence_policy import (
    DEFAULT_BETA_REQUIREMENTS,
    REQUIREMENT_BY_KEY,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args()

    items = json.loads(args.manifest.read_text())
    if not isinstance(items, list):
        raise ValueError("manifest must be a JSON array")

    by_fund: dict[str, list[dict]] = {}
    for item in items:
        by_fund.setdefault(item["scheme_family_key"], []).append(item)

    report: dict[str, dict] = {}
    all_ready = bool(by_fund)
    for fund_name, documents in sorted(by_fund.items()):
        types = {str(item["document_type"]) for item in documents}
        by_requirement = {}
        missing = []
        for key in DEFAULT_BETA_REQUIREMENTS:
            requirement = REQUIREMENT_BY_KEY[key]
            matching_types = sorted(types.intersection(requirement.accepted_document_types))
            covered = bool(matching_types)
            by_requirement[key] = {
                "covered_structurally": covered,
                "matching_document_types": matching_types,
                "accepted_document_types": list(requirement.accepted_document_types),
            }
            if not covered:
                missing.append(key)

        ready = not missing
        all_ready = all_ready and ready
        report[fund_name] = {
            "ready_for_ingestion": ready,
            "document_count": len(documents),
            "document_types": sorted(types),
            "missing_requirements": missing,
            "by_requirement": by_requirement,
        }

    print(json.dumps({
        "fund_count": len(by_fund),
        "semantic_requirements": list(DEFAULT_BETA_REQUIREMENTS),
        "structural_semantic_ready": all_ready,
        "funds": report,
        "note": (
            "Structural readiness only proves that acceptable authoritative document "
            "types exist. Final semantic coverage still requires indexed content "
            "to satisfy each requirement's content terms."
        ),
    }, indent=2))

    raise SystemExit(0 if all_ready else 2)


if __name__ == "__main__":
    main()
