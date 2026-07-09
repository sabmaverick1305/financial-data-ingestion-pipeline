"""Semantic Engine — loads and cross-validates the semantic/ knowledge stack.

    semantic/vocabulary.yaml           (layer 1: what concepts exist)
      -> taxonomy.yaml                  (layer 2: how entities classify)
      -> thesaurus.yaml                 (layer 3: what synonyms resolve to what)
      -> financial_ontology.yaml        (layer 4: what concepts formally mean)
      -> financial_relationships.yaml   (layer 5: how concepts causally relate)
      -> reasoning_rules.yaml           (layer 6: IF/THEN causal inference)

Single loader + cross-validator for the whole stack — semantic/ is the sole
source of truth (fies/ontology/entities.yaml, metrics.yaml, and the old
financial_ontology.yaml have been retired; fies/ontology/ now only holds
capabilities.yaml / templates.yaml / execution_labels.yaml, which are about
benchmark generation, not domain semantics).

Consumed by:
  - retrieval/ontology_resolver.py    (canonical entity/metric resolution)
  - retrieval/ontology_expansion.py   (retrieval query expansion)
  - retrieval/ontology_reranker.py    (reranking bonus)
  - reasoning/reasoning_engine.py     (causal "why" explanations)
  - fies/generator/query_generator.py (eval corpus generation)

Every layer may only reference ids registered in vocabulary.yaml or
taxonomy.yaml (layers 1-2) — validate_stack() enforces this so a typo in
layer 5 referencing a term that doesn't exist fails loudly at load time
(get_engine() raises), not silently at query time.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

SEMANTIC_DIR = Path(__file__).resolve().parents[3] / "semantic"

LAYER_FILES = [
    "vocabulary.yaml",
    "taxonomy.yaml",
    "thesaurus.yaml",
    "financial_ontology.yaml",
    "financial_relationships.yaml",
    "reasoning_rules.yaml",
]

_REL_TARGET_KEYS = ("depends_on", "influenced_by", "related_to", "indicates", "required_evidence")


@lru_cache(maxsize=1)
def load_stack(semantic_dir: Path = SEMANTIC_DIR) -> dict[str, dict]:
    docs = {}
    for name in LAYER_FILES:
        path = semantic_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required semantic layer: {path}")
        docs[name] = yaml.safe_load(path.read_text())
    return docs


@lru_cache(maxsize=1)
def all_term_ids() -> frozenset[str]:
    """Every id registered in vocabulary.yaml or taxonomy.yaml — the
    universe of ids every other layer is allowed to reference."""
    docs = load_stack()
    ids: set[str] = set()
    for term in docs["vocabulary.yaml"].get("metric_terms", []):
        ids.add(term["term_id"])
    for term in docs["vocabulary.yaml"].get("derived_concepts", []):
        ids.add(term["term_id"])
    tax = docs["taxonomy.yaml"]
    for group in ("scheme_types", "amcs", "document_types", "temporal_periods"):
        for e in tax.get(group, []):
            ids.add(e["entity_id"])
    return frozenset(ids)


def validate_stack() -> list[str]:
    """Cross-validate every layer only references ids that exist. Returns a
    list of human-readable problems; empty means the stack is consistent."""
    problems: list[str] = []
    docs = load_stack()
    known = all_term_ids()

    tax = docs["taxonomy.yaml"]
    scheme_ids = {e["entity_id"] for e in tax.get("scheme_types", [])}
    for e in tax.get("scheme_types", []):
        p = e.get("parent")
        if p not in (None, "all") and p not in scheme_ids:
            problems.append(f"taxonomy.scheme_types: {e['entity_id']!r} has unknown parent {p!r}")

    thes = docs["thesaurus.yaml"]
    metric_ids = {t["term_id"] for t in docs["vocabulary.yaml"].get("metric_terms", [])}
    for key in thes.get("metric_synonyms", {}):
        if key not in metric_ids:
            problems.append(f"thesaurus.metric_synonyms: {key!r} not in vocabulary.yaml metric_terms")
    amc_ids = {e["entity_id"] for e in tax.get("amcs", [])}
    for key in thes.get("scheme_type_synonyms", {}):
        if key not in scheme_ids:
            problems.append(f"thesaurus.scheme_type_synonyms: {key!r} not in taxonomy.yaml scheme_types")
    for key in thes.get("amc_synonyms", {}):
        if key not in amc_ids:
            problems.append(f"thesaurus.amc_synonyms: {key!r} not in taxonomy.yaml amcs")

    onto = docs["financial_ontology.yaml"]
    for concept_id in onto.get("concepts", {}):
        if concept_id not in known:
            problems.append(f"financial_ontology.concepts: {concept_id!r} not registered in vocabulary.yaml/taxonomy.yaml")

    rel = docs["financial_relationships.yaml"]
    for rel_id, r in rel.get("relationships", {}).items():
        subj = r.get("subject")
        if subj not in known:
            problems.append(f"financial_relationships.{rel_id}: subject {subj!r} not a known term")
        for key in _REL_TARGET_KEYS:
            for target in r.get(key) or []:
                if target not in known:
                    problems.append(f"financial_relationships.{rel_id}.{key}: {target!r} not a known term")
    for pat_id, p in rel.get("reasoning_patterns", {}).items():
        for target in p.get("required_evidence") or []:
            if target not in known:
                problems.append(f"financial_relationships.reasoning_patterns.{pat_id}.required_evidence: {target!r} not a known term")

    known_relationships = set(rel.get("relationships", {}))
    known_patterns = set(rel.get("reasoning_patterns", {}))
    rr = docs["reasoning_rules.yaml"]
    for rule in rr.get("rules", []):
        rid = rule["rule_id"]
        for cond in rule.get("conditions", []):
            if cond.get("metric") not in known:
                problems.append(f"reasoning_rules.{rid}: condition metric {cond.get('metric')!r} not a known term")
        for ev in rule.get("requires_evidence") or []:
            if ev not in known:
                problems.append(f"reasoning_rules.{rid}.requires_evidence: {ev!r} not a known term")
        sr = rule.get("supporting_relationship")
        if sr and sr not in known_relationships:
            problems.append(f"reasoning_rules.{rid}: supporting_relationship {sr!r} not in financial_relationships.yaml")
        sp = rule.get("supporting_pattern")
        if sp and sp not in known_patterns:
            problems.append(f"reasoning_rules.{rid}: supporting_pattern {sp!r} not in financial_relationships.yaml reasoning_patterns")

    return problems


@dataclass
class SemanticEngine:
    """Convenience façade over the raw layer docs — the lookup shapes
    consumer modules actually need, pre-built once per process."""

    docs: dict[str, dict]

    # ── layer 1: vocabulary ────────────────────────────────────────────
    @property
    def metric_ids(self) -> list[str]:
        return [t["term_id"] for t in self.docs["vocabulary.yaml"].get("metric_terms", [])]

    # ── layer 2: taxonomy ───────────────────────────────────────────────
    @property
    def scheme_type_ids(self) -> list[str]:
        return [e["entity_id"] for e in self.docs["taxonomy.yaml"].get("scheme_types", [])]

    @property
    def amc_ids(self) -> list[str]:
        return [e["entity_id"] for e in self.docs["taxonomy.yaml"].get("amcs", [])]

    def amc_display_name(self, entity_id: str) -> str | None:
        for e in self.docs["taxonomy.yaml"].get("amcs", []):
            if e["entity_id"] == entity_id:
                return e["label"]
        return None

    # mf_scheme_categories is a deliberately separate taxonomy from
    # scheme_types — see taxonomy.yaml's comment on why they aren't merged
    # (avoids a thesaurus synonym collision with the eval-verified
    # amfi_fund_stats scheme_type resolution). Not wired into
    # ontology_resolver.py; currently consumed by scripts/train_vanna.py
    # to give Vanna authoritative mf_scheme_master.category values.
    @property
    def mf_scheme_category_ids(self) -> list[str]:
        return [e["entity_id"] for e in self.docs["taxonomy.yaml"].get("mf_scheme_categories", [])]

    def mf_scheme_category(self, entity_id: str) -> dict | None:
        for e in self.docs["taxonomy.yaml"].get("mf_scheme_categories", []):
            if e["entity_id"] == entity_id:
                return e
        return None

    def mf_scheme_categories_leaf(self) -> list[dict]:
        """Entries with a category_pattern — excludes the four group headers
        (equity_scheme/debt_scheme/hybrid_scheme/other_scheme), which are
        parent nodes only, not real category_pattern-filterable leaves."""
        return [e for e in self.docs["taxonomy.yaml"].get("mf_scheme_categories", []) if "category_pattern" in e]

    # ── layer 3: thesaurus ──────────────────────────────────────────────
    def metric_synonyms(self, metric_id: str) -> list[str]:
        return self.docs["thesaurus.yaml"].get("metric_synonyms", {}).get(metric_id, {}).get("synonyms", [])

    def metric_negative_synonyms(self, metric_id: str) -> list[str]:
        return self.docs["thesaurus.yaml"].get("metric_synonyms", {}).get(metric_id, {}).get("negative_synonyms", [])

    def scheme_type_synonyms(self, entity_id: str) -> list[str]:
        return self.docs["thesaurus.yaml"].get("scheme_type_synonyms", {}).get(entity_id, [])

    def amc_synonyms(self, entity_id: str) -> list[str]:
        return self.docs["thesaurus.yaml"].get("amc_synonyms", {}).get(entity_id, [])

    def scheme_type_display_tokens(self, entity_id: str) -> list[str]:
        overrides = self.docs["thesaurus.yaml"].get("scheme_type_display_overrides", {})
        if entity_id in overrides:
            return overrides[entity_id]
        return [entity_id.replace("_", " ")]

    # ── layer 4: financial_ontology ─────────────────────────────────────
    def concept(self, concept_id: str) -> dict | None:
        return self.docs["financial_ontology.yaml"].get("concepts", {}).get(concept_id)

    def concept_canonical_name(self, concept_id: str) -> str | None:
        """Prefer financial_ontology.yaml's explicit canonical_name (set for
        FinancialMetric concepts). Fall back to taxonomy.yaml's label (set
        for FundCategory/AMC/DocumentType concepts), then a generic
        underscore-to-title-case display form for anything else (e.g. the
        derived_concepts in vocabulary.yaml, which have no formal concept
        entry at all)."""
        c = self.concept(concept_id)
        if c and c.get("canonical_name"):
            return c["canonical_name"]
        for group in ("scheme_types", "amcs", "document_types"):
            for e in self.docs["taxonomy.yaml"].get(group, []):
                if e["entity_id"] == concept_id:
                    return e["label"]
        return concept_id.replace("_", " ").title()

    def disallowed_template_families(self, metric_id: str) -> list[str]:
        c = self.concept(metric_id) or {}
        return c.get("disallowed_template_families", [])

    def metric_resolution_rules(self) -> list[dict]:
        return self.docs["financial_ontology.yaml"].get("metric_resolution_rules", [])

    def query_interpretation_rules(self) -> list[dict]:
        return self.docs["financial_ontology.yaml"].get("query_interpretation_rules", [])

    # ── layer 5: financial_relationships ────────────────────────────────
    def relationships_for(self, subject_id: str) -> list[dict]:
        return [
            {"relationship_id": rid, **r}
            for rid, r in self.docs["financial_relationships.yaml"].get("relationships", {}).items()
            if r.get("subject") == subject_id
        ]

    def relationship(self, relationship_id: str) -> dict | None:
        return self.docs["financial_relationships.yaml"].get("relationships", {}).get(relationship_id)

    def reasoning_pattern(self, pattern_id: str) -> dict | None:
        return self.docs["financial_relationships.yaml"].get("reasoning_patterns", {}).get(pattern_id)

    def related_concept_terms(self, metric_id: str) -> list[str]:
        """Flatten every relationship target for a metric into display terms
        — the vocabulary used by ontology_expansion.py's `related_terms`."""
        terms: list[str] = []
        for rel in self.relationships_for(metric_id):
            for key in _REL_TARGET_KEYS:
                for target in rel.get(key) or []:
                    if target == metric_id:
                        continue
                    label = self.concept_canonical_name(target) or target.replace("_", " ")
                    if label not in terms:
                        terms.append(label)
        return terms

    # ── layer 6: reasoning_rules ─────────────────────────────────────────
    def rules(self) -> list[dict]:
        return self.docs["reasoning_rules.yaml"].get("rules", [])

    def direction_thresholds(self) -> dict:
        return self.docs["reasoning_rules.yaml"].get("direction_thresholds", {})

    def no_match_response(self) -> dict:
        return self.docs["reasoning_rules.yaml"].get("no_match_response", {})


@lru_cache(maxsize=1)
def get_engine() -> SemanticEngine:
    problems = validate_stack()
    if problems:
        raise ValueError("semantic/ stack validation failed:\n" + "\n".join(f"  - {p}" for p in problems))
    return SemanticEngine(docs=load_stack())
