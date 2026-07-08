"""Ontology Resolver — Stage 0 of query understanding.

    User Query -> Ontology Resolver -> Canonical Query -> Intent -> Planner

Resolves canonical metric/scheme_type/AMC ids and query-interpretation-rule
hints from a raw query, using the semantic/ knowledge stack (via
semantic_engine.SemanticEngine) as the single source of truth for synonym
data — the same stack fies/generator/query_generator.py compiles into the
eval corpus. Previously that synonym data was duplicated by hand across
SCHEME_TYPES / _METRIC_MAP / _AMC_NAMES in query_understanding.py, so
updating the ontology never changed live classifier behavior.

Additive, not authoritative: QueryAnalyzer's existing hand-tuned keyword
extraction remains the primary signal — proven against the eval suite through
many targeted fixes. The resolver's CanonicalQuery is consulted as a fallback
wherever the existing extractors find nothing (see _extract_metric /
_extract_entities), and its matched-rule hints are attached for
observability now, ahead of being wired into routing/aggregation decisions in
a later pass.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from financial_pipeline.semantic.semantic_engine import SemanticEngine, get_engine


@dataclass
class MatchedRule:
    rule_id: str
    actions: dict


@dataclass
class CanonicalQuery:
    """Structured, ontology-resolved view of a raw query."""

    raw_query: str
    metrics:      list[str] = field(default_factory=list)  # canonical metric_ids, best match first
    scheme_types: list[str] = field(default_factory=list)  # display tokens, same convention as intent.scheme_types
    scheme_type_ids: list[str] = field(default_factory=list)  # raw taxonomy.yaml entity_ids (pre-display-override)
    amcs:         list[str] = field(default_factory=list)  # display names, same convention as intent.amc_names
    matched_rules: list[MatchedRule] = field(default_factory=list)
    caveats:       list[str] = field(default_factory=list)


def _word_match(term: str, q: str) -> bool:
    return re.search(r"\b" + re.escape(term) + r"\b", q) is not None


@lru_cache(maxsize=1)
def _build_tables() -> dict:
    """Flatten SemanticEngine's per-id lookups into the sorted synonym
    tables _resolve_ids needs. Built once per process (lru_cache)."""
    eng: SemanticEngine = get_engine()

    metric_synonyms: list[tuple[str, str]] = []
    metric_negative: dict[str, list[str]] = {}
    for metric_id in eng.metric_ids:
        for syn in eng.metric_synonyms(metric_id):
            metric_synonyms.append((syn.lower(), metric_id))
        metric_negative[metric_id] = [s.lower() for s in eng.metric_negative_synonyms(metric_id)]
    metric_synonyms.sort(key=lambda t: len(t[0]), reverse=True)

    scheme_synonyms: list[tuple[str, str]] = []
    for entity_id in eng.scheme_type_ids:
        for syn in eng.scheme_type_synonyms(entity_id):
            scheme_synonyms.append((syn.lower(), entity_id))
    scheme_synonyms.sort(key=lambda t: len(t[0]), reverse=True)

    amc_synonyms: list[tuple[str, str]] = []
    for entity_id in eng.amc_ids:
        for syn in eng.amc_synonyms(entity_id):
            amc_synonyms.append((syn.lower(), entity_id))
    amc_synonyms.sort(key=lambda t: len(t[0]), reverse=True)

    return {
        "metric_synonyms": metric_synonyms,
        "metric_negative": metric_negative,
        "scheme_synonyms": scheme_synonyms,
        "amc_synonyms": amc_synonyms,
        "query_interpretation_rules": eng.query_interpretation_rules(),
    }


def _resolve_ids(q: str, synonym_table: list[tuple[str, str]]) -> list[str]:
    """Longest-synonym-first match, one entry per canonical id (first/most
    specific hit wins), order = order of first match."""
    ordered: list[str] = []
    for syn, canonical_id in synonym_table:
        if canonical_id in ordered:
            continue
        if _word_match(syn, q):
            ordered.append(canonical_id)
    return ordered


def _resolve_metrics(q: str, tables: dict) -> list[str]:
    resolved = []
    for metric_id in _resolve_ids(q, tables["metric_synonyms"]):
        negatives = tables["metric_negative"].get(metric_id, [])
        if any(_word_match(neg, q) for neg in negatives):
            continue
        resolved.append(metric_id)
    return resolved


def _resolve_scheme_types(q: str, tables: dict) -> list[str]:
    eng = get_engine()
    tokens: list[str] = []
    for entity_id in _resolve_ids(q, tables["scheme_synonyms"]):
        for tok in eng.scheme_type_display_tokens(entity_id):
            if tok not in tokens:
                tokens.append(tok)
    return tokens


def _resolve_amcs(q: str, tables: dict) -> list[str]:
    eng = get_engine()
    names: list[str] = []
    for entity_id in _resolve_ids(q, tables["amc_synonyms"]):
        # taxonomy.yaml's label is the full legal name ("HDFC Mutual Fund");
        # query_understanding.py's existing _AMC_NAMES list uses the short
        # form ("HDFC") — match that convention so the fallback is a drop-in
        # for intent.amc_names, not a second dialect.
        name = re.sub(r"\s+Mutual Fund$", "", eng.amc_display_name(entity_id) or entity_id)
        if name not in names:
            names.append(name)
    return names


def _resolve_matched_rules(q: str, tables: dict) -> list[MatchedRule]:
    matched = []
    for rule in tables["query_interpretation_rules"]:
        terms = [t.lower() for t in rule.get("triggers", {}).get("required_terms_any", [])]
        if terms and any(_word_match(t, q) for t in terms):
            matched.append(MatchedRule(rule_id=rule["rule_id"], actions=rule.get("actions", {})))
    return matched


def resolve(query: str) -> CanonicalQuery:
    """Resolve a raw query into canonical ids + matched interpretation rules."""
    q = query.lower().strip()
    tables = _build_tables()
    matched_rules = _resolve_matched_rules(q, tables)
    scheme_type_ids = _resolve_ids(q, tables["scheme_synonyms"])

    return CanonicalQuery(
        raw_query=query,
        metrics=_resolve_metrics(q, tables),
        scheme_types=_resolve_scheme_types(q, tables),
        scheme_type_ids=scheme_type_ids,
        amcs=_resolve_amcs(q, tables),
        matched_rules=matched_rules,
        caveats=[r.actions["caveat"] for r in matched_rules if "caveat" in r.actions],
    )
