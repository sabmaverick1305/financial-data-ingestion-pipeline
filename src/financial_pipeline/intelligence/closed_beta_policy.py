"""Closed-beta product scope policy.

The beta universe is intentionally smaller than the production investment
mandate: only scheme families with complete structural documentary evidence
are admitted to candidate discovery.
"""
from __future__ import annotations

from dataclasses import dataclass


CLOSED_BETA_SCHEME_FAMILIES: tuple[str, ...] = (
    "bandhan small cap fund",
    "icici prudential large and mid cap fund",
    "invesco india large and mid cap fund",
    "invesco india midcap fund",
    "motilal oswal large and midcap fund",
    "motilal oswal midcap fund",
)


@dataclass(frozen=True)
class ClosedBetaUniversePolicy:
    allowed_scheme_families: frozenset[str]

    @classmethod
    def default(cls) -> "ClosedBetaUniversePolicy":
        return cls(frozenset(CLOSED_BETA_SCHEME_FAMILIES))

    def allows(self, scheme_family_key: str) -> bool:
        return scheme_family_key in self.allowed_scheme_families

    def filter_rows(self, rows: list[dict]) -> list[dict]:
        return [
            row
            for row in rows
            if str(row.get("scheme_family_key") or "") in self.allowed_scheme_families
        ]
