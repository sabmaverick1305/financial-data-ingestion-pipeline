"""Dataclasses for the mfapi.in scheme master / NAV history ingestion job."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class SchemeRecord:
    scheme_code: str
    scheme_name: str
    amc_name: str | None = None
    category: str | None = None
    scheme_type: str | None = None


@dataclass(frozen=True)
class NavRecord:
    scheme_code: str
    nav_date: date
    nav: float
