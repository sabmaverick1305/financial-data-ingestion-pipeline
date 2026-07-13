"""Canonical Name Normalizer — implements domain/entity_model/canonical_naming_rules.yaml.

Two independent normalizations, matching the two identifiers
canonical_naming_rules.yaml documents:

  scheme_name -> ParsedSchemeName (base_fund_name, plan, option)
    Previously undocumented as "parsing_status: Not implemented" — this is
    that implementation. scheme_name packs plan (Direct/Regular) and option
    (Growth/IDCW) into one free-text string with no dedicated columns; every
    consumer today (services/ontology_resolver.py, Vanna's ILIKE scheme
    lookup) treats it as one opaque string, so two variants of the same
    underlying fund are unrelated rows. This gives a first, honest cut at
    recovering that structure.

  amc_name -> canonical label (via services/entity_resolver.py + SemanticEngine)
    Thin composition: resolve to amc_entity_id, look up its
    domain/semantic/taxonomy.yaml label. Falls back to the raw input
    unchanged when unresolved — never fabricates a canonical form for an
    AMC this system doesn't know.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import yaml

from financial_pipeline.semantic.semantic_engine import DOMAIN_DIR, get_engine
from financial_pipeline.services.entity_resolver import resolve_amc_name


@lru_cache(maxsize=1)
def _naming_rules() -> dict:
    path = DOMAIN_DIR / "entity_model" / "canonical_naming_rules.yaml"
    return yaml.safe_load(path.read_text())


@lru_cache(maxsize=1)
def _plan_lookup() -> list[tuple[str, str]]:
    """(surface_form_lower, canonical) pairs, longest-first so "Direct Plan" matches before "Direct"."""
    pairs = [
        (surface.lower(), tok["canonical"])
        for tok in _naming_rules()["scheme_name"]["plan_tokens"]
        for surface in tok["surface_forms"]
    ]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


@lru_cache(maxsize=1)
def _option_lookup() -> list[tuple[str, str]]:
    pairs = [
        (surface.lower(), tok["canonical"])
        for tok in _naming_rules()["scheme_name"]["option_tokens"]
        for surface in tok["surface_forms"]
    ]
    return sorted(pairs, key=lambda p: len(p[0]), reverse=True)


@dataclass
class ParsedSchemeName:
    raw: str
    base_fund_name: str
    plan: str | None    # "direct" | "regular" | None
    option: str | None  # "growth" | "idcw" | None


def parse_scheme_name(scheme_name: str) -> ParsedSchemeName:
    """Split a scheme_name into base_fund_name + plan + option, per the
    "{base_fund_name} - {plan} - {option}" convention documented in
    canonical_naming_rules.yaml. Matching is case-insensitive, word-boundary;
    unmatched plan/option tokens are left as None rather than guessed.
    """
    remaining = scheme_name
    plan = None
    for surface_lower, canonical in _plan_lookup():
        m = re.search(r"\b" + re.escape(surface_lower) + r"\b", remaining, flags=re.IGNORECASE)
        if m:
            plan = canonical
            remaining = remaining[: m.start()] + remaining[m.end() :]
            break

    option = None
    for surface_lower, canonical in _option_lookup():
        m = re.search(r"\b" + re.escape(surface_lower) + r"\b", remaining, flags=re.IGNORECASE)
        if m:
            option = canonical
            remaining = remaining[: m.start()] + remaining[m.end() :]
            break

    # Strip leftover " - " / "-" separators and whitespace from removed tokens.
    base = re.sub(r"\s*-\s*-\s*", " - ", remaining)
    base = re.sub(r"\s*-\s*$", "", base)
    base = re.sub(r"\s{2,}", " ", base).strip(" -")

    return ParsedSchemeName(raw=scheme_name, base_fund_name=base, plan=plan, option=option)


def canonicalize_amc_name(amc_name: str) -> str:
    """Resolve amc_name to its canonical taxonomy.yaml label. Returns the
    original string unchanged if it doesn't resolve to any amc_entity_id —
    per canonical_naming_rules.yaml, never fabricates a canonical form."""
    match = resolve_amc_name(amc_name)
    if match.entity_id is None:
        return amc_name
    return get_engine().amc_display_name(match.entity_id) or amc_name


def normalize_name(s: str) -> str:
    """Collapse internal whitespace and lowercase — the natural-key
    normalization financial_entity_master.normalized_name expects (see
    uq_active_entity_identity). Distinct from parse_scheme_name/
    canonicalize_amc_name, which extract or resolve structure; this is
    just the plain string-identity normalization every entity_type uses
    before a lookup_entity/get_or_create_entity call."""
    return re.sub(r"\s{2,}", " ", s.strip().lower())
