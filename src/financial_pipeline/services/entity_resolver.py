"""Entity Resolver — implements domain/resolution/entity_resolution_rules.yaml's
amc_name_to_entity_id and category_resolution_with_scheme_name_fallback
rules as real, callable code.

Distinct from services/ontology_resolver.py: that module resolves ids
mentioned *inside free-form query text* ("what's HDFC's AUM" -> amc
"HDFC"), Stage 0 of query understanding. This module resolves *entity
field values* — e.g. a raw mf_scheme_master.amc_name string — to a
canonical amc_entity_id, which is the batch/programmatic use case
entity_resolution_rules.yaml's amc_name_to_entity_id rule flags as a known
gap ("no batch job that back-fills amc_entity_id onto existing scheme
rows"). Both reuse the same underlying synonym data (domain/semantic/
thesaurus.yaml's amc_synonyms) via SemanticEngine — they differ in mount
point (whole-query substring search vs. single-value exact match) and in
result shape (MatchResult carries a score/band per
domain/resolution/matching_thresholds.yaml, ontology_resolver's resolve()
does not).

Matching strategy is the "current_model" in matching_thresholds.yaml:
binary exact word-boundary match (score 1.0 or 0.0) — there is no fuzzy
matching in this codebase yet. MatchResult.band still uses the band
vocabulary defined for the (not-yet-implemented) future_model so callers
get a stable output shape regardless of which model is active later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import yaml

from financial_pipeline.semantic.semantic_engine import DOMAIN_DIR, get_engine


@lru_cache(maxsize=1)
def _resolution_rules() -> dict:
    path = DOMAIN_DIR / "resolution" / "entity_resolution_rules.yaml"
    return yaml.safe_load(path.read_text())


@lru_cache(maxsize=1)
def _matching_thresholds() -> dict:
    path = DOMAIN_DIR / "resolution" / "matching_thresholds.yaml"
    return yaml.safe_load(path.read_text())


def amc_name_to_entity_id_rule() -> dict:
    """The entity_resolution_rules.yaml rule this module implements, for
    callers that want to surface its documented on_no_match/scope text
    (e.g. in an admin UI) alongside a live MatchResult."""
    for rule in _resolution_rules()["rules"]:
        if rule["rule_id"] == "amc_name_to_entity_id":
            return rule
    raise KeyError("amc_name_to_entity_id rule missing from entity_resolution_rules.yaml")


@dataclass
class MatchResult:
    query: str
    entity_id: str | None
    score: float
    band: str  # "auto_accept" | "needs_review" | "reject" — see matching_thresholds.yaml
    source: str = "primary"  # which input field produced the match — "primary" | "scheme_name_fallback"


def _band_for_score(score: float) -> str:
    for band in _matching_thresholds()["future_model"]["thresholds"]:
        lo, hi = band["range"]
        if lo <= score <= hi:
            return band["band"]
    return "reject"


def resolve_amc_name(raw_amc_name: str) -> MatchResult:
    """Resolve a single raw amc_name value to its canonical amc_entity_id.

    Word-boundary, case-insensitive exact match against
    domain/semantic/thesaurus.yaml's amc_synonyms — same mechanism as
    services/ontology_resolver.py's _resolve_amcs, applied to one value
    instead of scanned across free-form query text.
    """
    eng = get_engine()
    q = raw_amc_name.strip().lower()
    for entity_id in eng.amc_ids:
        synonyms = sorted(eng.amc_synonyms(entity_id), key=len, reverse=True)
        for syn in synonyms:
            if re.search(r"\b" + re.escape(syn.lower()) + r"\b", q):
                return MatchResult(query=raw_amc_name, entity_id=entity_id, score=1.0, band=_band_for_score(1.0))
    return MatchResult(query=raw_amc_name, entity_id=None, score=0.0, band=_band_for_score(0.0))


def find_unresolved_amc_names(raw_amc_names: list[str]) -> list[str]:
    """Given distinct amc_name values (e.g. `SELECT DISTINCT amc_name FROM
    mf_scheme_master`), return those that don't resolve to any
    amc_entity_id — the concrete tool for the known_gap
    entity_resolution_rules.yaml documents: finding newly-onboarded AMCs
    that need a domain/semantic/taxonomy.yaml + thesaurus.yaml entry."""
    return [name for name in raw_amc_names if resolve_amc_name(name).entity_id is None]


def _like_to_regex(pattern: str) -> str:
    """SQL LIKE -> regex, preserving multi-wildcard patterns like
    "%banking%psu%" (two of taxonomy.yaml's 41 category_patterns use more
    than one '%'; a naive single-substring containment check would mishandle
    those)."""
    return ".*".join(re.escape(p) for p in pattern.split("%"))


@lru_cache(maxsize=1)
def _category_leaf_patterns() -> list[tuple[str, re.Pattern]]:
    return [
        (leaf["entity_id"], re.compile(_like_to_regex(leaf["category_pattern"]), re.IGNORECASE))
        for leaf in get_engine().mf_scheme_categories_leaf()
    ]


def category_resolution_rule() -> dict:
    for rule in _resolution_rules()["rules"]:
        if rule["rule_id"] == "category_resolution_with_scheme_name_fallback":
            return rule
    raise KeyError("category_resolution_with_scheme_name_fallback rule missing from entity_resolution_rules.yaml")


def resolve_category(
    raw_category: str | None,
    scheme_name: str | None = None,
    amfi_category: str | None = None,
) -> MatchResult:
    """Resolve a scheme's category to a domain/semantic/taxonomy.yaml
    mf_scheme_categories leaf entity_id, via real SQL-LIKE-semantics regex
    matching against each leaf's category_pattern (not naive substring
    containment). Tries three signals in priority order:

    1. amfi_category — the section-header category from AMFI's own
       NAVAll.txt ("Open Ended Schemes(Debt Scheme - Banking and PSU
       Fund)" -> "Debt Scheme - Banking and PSU Fund"), parsed by
       scripts/fetch_amfi_nav.py's parse_nav_text(). This is the real,
       authoritative SEBI category — highest confidence, when the caller
       has it (requires joining on scheme_code against a fetched
       NAVAll.txt; NOT looked up automatically inside this function).
    2. raw_category — mf_scheme_master.category (mfapi.in's own field).
       Unreliable for a large fraction of schemes — it holds plan/option/
       tenure text ("Growth", "1099 Days") or flatly wrong values
       ("Income", over half of all rows) instead of a real SEBI category
       (see domain/entity_model/source_trust.yaml's known_gaps).
    3. scheme_name fallback — matched against the same patterns, since
       scheme names often embed their category (e.g. "...Liquid Fund...").
       Scored into the "needs_review" band, not "auto_accept".

    See domain/resolution/entity_resolution_rules.yaml's
    category_resolution_with_scheme_name_fallback rule for the full
    rationale and coverage numbers.
    """
    if amfi_category:
        for entity_id, pattern in _category_leaf_patterns():
            if pattern.search(amfi_category):
                return MatchResult(query=amfi_category, entity_id=entity_id, score=1.0, band=_band_for_score(1.0), source="amfi_navall")

    if raw_category:
        for entity_id, pattern in _category_leaf_patterns():
            if pattern.search(raw_category):
                return MatchResult(query=raw_category, entity_id=entity_id, score=1.0, band=_band_for_score(1.0), source="primary")

    if scheme_name:
        for entity_id, pattern in _category_leaf_patterns():
            if pattern.search(scheme_name):
                return MatchResult(query=scheme_name, entity_id=entity_id, score=0.85, band=_band_for_score(0.85), source="scheme_name_fallback")

    return MatchResult(query=raw_category or "", entity_id=None, score=0.0, band=_band_for_score(0.0), source="primary")
