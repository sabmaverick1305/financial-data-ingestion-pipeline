"""Normalize mutual-fund share classes into one underlying scheme family."""
from __future__ import annotations
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class SchemeFamilyDecision:
    family_key: str
    preferred: bool
    preference_score: int

class SchemeFamilyPolicy:
    _STRIP = (
        r"\bdirect\b",
        r"\bregular\b",
        r"\bplan\b",
        r"\bgrowth\b",
        r"\bidcw\b",
        r"\bdividend\b",
        r"\boption\b",
        r"\bpayout\b",
        r"\breinvestment\b",
    )

    def family_key(self, name: str) -> str:
        value = name.lower()
        value = value.replace("&", " and ")
        for pattern in self._STRIP:
            value = re.sub(pattern, " ", value)
        value = re.sub(r"[^a-z0-9]+", " ", value)
        value = re.sub(r"\s+", " ", value).strip()
        return value

    def preference_score(self, name: str) -> int:
        value = name.lower()
        score = 0
        if "direct" in value:
            score += 100
        if "growth" in value:
            score += 20
        if "regular" in value:
            score -= 20
        if "idcw" in value or "dividend" in value:
            score -= 10
        return score

    def deduplicate(self, rows: list[dict]) -> list[dict]:
        selected: dict[str, dict] = {}
        scores: dict[str, int] = {}
        for row in rows:
            name = str(row.get("scheme_name") or "")
            key = self.family_key(name)
            if not key:
                key = str(row.get("scheme_code") or name)
            score = self.preference_score(name)
            current = selected.get(key)
            if current is None or score > scores[key]:
                enriched = dict(row)
                enriched["scheme_family_key"] = key
                enriched["share_class_preference_score"] = score
                selected[key] = enriched
                scores[key] = score
        return list(selected.values())
