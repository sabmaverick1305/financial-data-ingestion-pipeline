"""Evidence lineage attached to computed/retrieved reasoning observations."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class EvidenceLineage:
    source_ids: tuple[str, ...] = ()
    date_range: tuple[str | None, str | None] = (None, None)
    transformation: str | None = None
    metric: str | None = None
    authority: str | None = None
