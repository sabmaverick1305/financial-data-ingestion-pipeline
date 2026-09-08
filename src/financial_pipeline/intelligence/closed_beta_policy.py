"""Closed-beta product scope policy.

The beta universe is intentionally smaller than the production investment
mandate: only scheme families with complete structural documentary evidence
are admitted to candidate discovery.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


CLOSED_BETA_SCHEME_FAMILIES: tuple[str, ...] = (
    "bandhan small cap fund",
    "icici prudential large and mid cap fund",
    "invesco india large and mid cap fund",
    "invesco india midcap fund",
    "motilal oswal large and midcap fund",
    "motilal oswal midcap fund",
)


def canonical_beta_family_key(value: str) -> str:
    """Normalize documentary/discovery spelling variants to one beta identity."""
    normalized = value.lower().replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Existing production scheme names use Midcap as a single token while
    # some AMC/source manifests use "Mid Cap". Keep one persistent identity.
    normalized = normalized.replace("mid cap", "midcap")
    return normalized


@dataclass(frozen=True)
class ClosedBetaUniversePolicy:
    allowed_scheme_families: frozenset[str]

    @classmethod
    def default(cls) -> "ClosedBetaUniversePolicy":
        return cls(
            frozenset(
                canonical_beta_family_key(value)
                for value in CLOSED_BETA_SCHEME_FAMILIES
            )
        )

    def canonicalize(self, scheme_family_key: str) -> str:
        return canonical_beta_family_key(scheme_family_key)

    def allows(self, scheme_family_key: str) -> bool:
        return self.canonicalize(scheme_family_key) in self.allowed_scheme_families

    def filter_rows(self, rows: list[dict]) -> list[dict]:
        filtered: list[dict] = []
        for row in rows:
            canonical = self.canonicalize(
                str(row.get("scheme_family_key") or "")
            )
            if canonical not in self.allowed_scheme_families:
                continue
            enriched = dict(row)
            enriched["scheme_family_key"] = canonical
            filtered.append(enriched)
        return filtered
