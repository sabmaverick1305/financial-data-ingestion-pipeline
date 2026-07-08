"""Validates raw scheme records read from the S3 scheme-master JSON file.

The source file is expected to be a JSON array of objects with (at least)
`schemeCode` and `schemeName` — mirroring mfapi.in's own field naming.
`isInGrowth` / `isInDivReinvestment`, if present, are not currently persisted
(mf_scheme_master has no ISIN columns) — the authoritative amc_name/category/
scheme_type instead come from each scheme's own mfapi `meta` block, fetched
in sync.py.
"""

from __future__ import annotations

from typing import Any

from financial_pipeline.mf_ingestion.models import SchemeRecord


class InvalidSchemeRecord(Exception):
    def __init__(self, raw: dict[str, Any], reason: str) -> None:
        self.raw = raw
        self.reason = reason
        super().__init__(reason)


def parse_scheme_record(raw: dict[str, Any]) -> SchemeRecord:
    scheme_code = str(raw.get("schemeCode") or "").strip()
    scheme_name = str(raw.get("schemeName") or "").strip()

    if not scheme_code or not scheme_code.isdigit():
        raise InvalidSchemeRecord(raw, f"missing/non-numeric schemeCode: {raw.get('schemeCode')!r}")
    if not scheme_name:
        raise InvalidSchemeRecord(raw, "missing schemeName")

    return SchemeRecord(scheme_code=scheme_code, scheme_name=scheme_name)
