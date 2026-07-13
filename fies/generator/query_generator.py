"""FIES Query Generator — compiles fies/ontology/*.yaml + domain/semantic/*.yaml into
a versioned query corpus.

Responsibility split:
  fies/ontology/templates.yaml          -> How to ask questions (patterns, variants, placeholders).
  fies/ontology/capabilities.yaml       -> What cognitive capability a template exercises.
  fies/ontology/execution_labels.yaml   -> What route/execution_plan a query should resolve to.
  domain/semantic/vocabulary.yaml, taxonomy.yaml
    -> What concepts exist (canonical metric/scheme_type/AMC ids), via SemanticEngine.
  domain/semantic/financial_ontology.yaml
    -> What those concepts mean (definitions, canonical names) and which
       metric/template-family combinations are meaningful
       (disallowed_template_families), via SemanticEngine.
  query_generator.py (this file) -> Combines all of the above into a VALID benchmark
                               corpus: expands templates, and uses domain/semantic/financial_ontology.yaml
                               to reject metric/template combinations that are syntactically
                               fine but semantically nonsensical (e.g. "total AUM from 2020
                               to 2024" — AUM is a stock/snapshot metric, summing it over a
                               range is not a meaningful operation).

entities.yaml, metrics.yaml, and the old financial_ontology.yaml previously lived in
fies/ontology/ — that content has been migrated into domain/semantic/ (the single source of
truth for domain semantics, shared with the live retrieval/reranking/reasoning code).
fies/ontology/ now only holds benchmark-generation concerns: capabilities, templates,
execution labels.

Each run writes a NEW version under eval/corpus/generated/v{N}/ — it never touches the
hand-authored eval/corpus/*.json files, and eval/run_eval.py is not wired to read generated
versions (a deliberate, separate follow-up). Re-run whenever the specs change; that's what
makes this a "compiler" rather than a one-off script — the YAML specs are the source of
truth, this file is the build step.

sql_labels.json / retrieval_labels.json / expected_answers.json are NOT generated here: they
require real DB values or retrieved chunk IDs that the YAML specs don't encode. Fabricating
those would silently corrupt eval accuracy, so promoting a generated version to a full
eval-ready corpus is a manual/separate step.

Usage:
  python fies/generator/query_generator.py
  python fies/generator/query_generator.py --per-template 15 --seed 7
  python fies/generator/query_generator.py --per-template 5 --dry-run
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

GENERATOR_DIR = Path(__file__).resolve().parent
FIES_DIR = GENERATOR_DIR.parent
ONTOLOGY_DIR = FIES_DIR / "ontology"
REPO_ROOT = FIES_DIR.parent
CORPUS_DIR = REPO_ROOT / "eval" / "corpus"
GENERATED_DIR = CORPUS_DIR / "generated"

sys.path.insert(0, str(REPO_ROOT / "src"))
from financial_pipeline.semantic.semantic_engine import (  # noqa: E402
    SEMANTIC_DIR,
    LAYER_FILES as SEMANTIC_LAYER_FILES,
    SemanticEngine,
    get_engine,
)

# fies/ontology/ files consumed to build the template pools and validate
# cross-references. entities.yaml/metrics.yaml/financial_ontology.yaml used
# to live here too — that domain-semantics content is now in domain/semantic/,
# sourced via SemanticEngine (see build_lookup_tables).
SOURCE_FILES = [
    "capabilities.yaml",
    "templates.yaml",
    "execution_labels.yaml",
]

_PLACEHOLDER_RE = re.compile(r"\{(\w+)\}")

# Maps a template placeholder name to the pool it draws from. Suffixed
# variants (_a/_b, start_/end_) share the same pool but are resolved as a
# pair so distinctness/ordering constraints can be enforced together.
_VAR_POOL = {
    "metric": "metric", "metric_a": "metric", "metric_b": "metric",
    "metric_snapshot": "metric_snapshot",
    "scheme_type": "scheme_type", "scheme_type_a": "scheme_type", "scheme_type_b": "scheme_type",
    "amc": "amc",
    "year": "year", "start_year": "year", "end_year": "year",
    "year_a": "year", "year_b": "year",
    "month": "month", "quarter": "quarter", "top_n": "top_n",
    "future_year": "future_year",
}

# Placeholder pairs that must be resolved together, with the constraint that
# applies. "distinct" = the two values must differ; "ordered" = the first
# must be strictly less than the second (only meaningful for years).
_PAIR_CONSTRAINTS = {
    ("start_year", "end_year"): "ordered",
    ("year_a", "year_b"): "distinct",
    ("scheme_type_a", "scheme_type_b"): "distinct",
    ("metric_a", "metric_b"): "distinct",
}

_FUTURE_YEARS = [2030, 2035, 2050]

# Maps a template's execution_plan to the "template family" vocabulary used
# by financial_ontology.yaml's valid_metric_template_constraints. Plans with
# no metric-family semantics (routing/guardrail/metadata plans) are omitted
# on purpose — _filter_metric_pool treats a missing mapping as "no
# constraint applies".
FAMILY_BY_EXECUTION_PLAN = {
    "latest_snapshot": "lookup_snapshot",
    "point_in_time_lookup": "point_in_time",
    "range_sum": "sum_over_range",
    "range_average": "average_over_range",
    "entity_comparison": "comparison",
    "time_comparison": "comparison",
    "ranking_top_n": "ranking",
    "ranking_bottom_n": "ranking",
    "time_series_trend": "trend",
    "derived_growth_ranking": "trend",
    "empty_result_future_or_out_of_range": "point_in_time",
}


# ── Loading & validation ───────────────────────────────────────────────────

def load_yaml_files(ontology_dir: Path) -> dict[str, dict]:
    docs = {}
    for name in SOURCE_FILES:
        path = ontology_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Missing required spec: {path}")
        docs[name] = yaml.safe_load(path.read_text())
    return docs


def _file_hash(ontology_dir: Path, name: str) -> str:
    return hashlib.sha256((ontology_dir / name).read_bytes()).hexdigest()[:12]


def build_lookup_tables(docs: dict) -> dict:
    capabilities_by_id = {}
    for domain in docs["capabilities.yaml"]["capability_domains"]:
        for cap in domain["capabilities"]:
            capabilities_by_id[cap["capability_id"]] = {**cap, "domain_id": domain["domain_id"],
                                                          "domain_name": domain["domain_name"]}

    execution_plans_by_id = {p["plan_id"]: p for p in docs["execution_labels.yaml"]["execution_plans"]}
    routes_by_id = {r["route_id"]: r for r in docs["execution_labels.yaml"]["routes"]}

    # Domain semantics (concept ids, canonical names, template-family
    # constraints) now live in domain/semantic/, not fies/ontology/ — sourced
    # through SemanticEngine so this generator and the live retrieval code
    # share one vocabulary. entities_by_group/metrics_by_id are rebuilt in
    # the shape the rest of this file already expects, so _build_pools/
    # _display_text/_filter_metric_pool need no further changes.
    eng: SemanticEngine = get_engine()
    entities_by_group: dict[str, list[dict]] = {
        "scheme_type": [
            {"entity_id": eid, "canonical_name": eng.concept_canonical_name(eid) or eid}
            for eid in eng.scheme_type_ids
        ],
        "amc": [
            {"entity_id": eid, "canonical_name": eng.amc_display_name(eid) or eid}
            for eid in eng.amc_ids
        ],
    }
    metrics_by_id = {
        mid: {"canonical_name": eng.concept_canonical_name(mid) or mid}
        for mid in eng.metric_ids
    }
    valid_metric_template_constraints = {
        mid: {"disallowed_template_families": eng.disallowed_template_families(mid)}
        for mid in eng.metric_ids
    }

    return {
        "capabilities_by_id": capabilities_by_id,
        "entities_by_group": entities_by_group,
        "metrics_by_id": metrics_by_id,
        "execution_plans_by_id": execution_plans_by_id,
        "routes_by_id": routes_by_id,
        "valid_metric_template_constraints": valid_metric_template_constraints,
    }


def validate_cross_references(docs: dict, lookups: dict) -> list[str]:
    """Return a list of human-readable problems; empty means the specs are consistent."""
    problems = []
    tv = docs["templates.yaml"]["template_variables"]

    for tmpl in docs["templates.yaml"]["templates"]:
        tid = tmpl["template_id"]
        for cap_id in tmpl.get("capabilities", []):
            if cap_id not in lookups["capabilities_by_id"]:
                problems.append(f"{tid}: unknown capability_id {cap_id!r}")
        if tmpl.get("expected_route") not in lookups["routes_by_id"]:
            problems.append(f"{tid}: unknown expected_route {tmpl.get('expected_route')!r}")
        if tmpl.get("execution_plan") not in lookups["execution_plans_by_id"]:
            problems.append(f"{tid}: unknown execution_plan {tmpl.get('execution_plan')!r}")

        for text in [tmpl["pattern"], *tmpl.get("variants", [])]:
            for ph in _PLACEHOLDER_RE.findall(text):
                if ph not in _VAR_POOL:
                    problems.append(f"{tid}: placeholder {{{ph}}} has no known pool mapping")
                elif _VAR_POOL[ph] not in ("scheme_type", "amc", "future_year") and _VAR_POOL[ph] not in tv:
                    problems.append(f"{tid}: pool {_VAR_POOL[ph]!r} for {{{ph}}} missing from template_variables")

    for entity_id in tv.get("scheme_type", []):
        if not any(e["entity_id"] == entity_id for e in lookups["entities_by_group"].get("scheme_type", [])):
            problems.append(f"template_variables.scheme_type: {entity_id!r} not defined in domain/semantic/taxonomy.yaml")

    for pool_name in ("metric", "metric_snapshot"):
        for metric_id in tv.get(pool_name, []):
            if metric_id not in lookups["metrics_by_id"]:
                problems.append(f"template_variables.{pool_name}: {metric_id!r} not defined in domain/semantic/vocabulary.yaml")

    return problems


def validate_ontology_constraints(docs: dict, lookups: dict) -> list[str]:
    """Check financial_ontology.yaml's valid_metric_template_constraints don't
    leave any implemented template with zero usable metric values once
    _filter_metric_pool is applied — that would mean the spec is
    unsatisfiable (a real authoring error), not just a narrowing.
    """
    problems = []
    tv = docs["templates.yaml"]["template_variables"]

    for tmpl in docs["templates.yaml"]["templates"]:
        if tmpl.get("status") == "unimplemented":
            continue
        text_pool = [tmpl["pattern"], *tmpl.get("variants", [])]
        placeholders = {ph for text in text_pool for ph in _PLACEHOLDER_RE.findall(text)}
        for pool_name in ("metric", "metric_snapshot"):
            if not any(_VAR_POOL.get(ph) == pool_name for ph in placeholders):
                continue
            base_pool = tv.get(pool_name, [])
            filtered = _filter_metric_pool(tmpl, pool_name, base_pool, lookups)
            if not filtered:
                problems.append(
                    f"{tmpl['template_id']}: every {pool_name} value is disallowed for "
                    f"execution_plan {tmpl['execution_plan']!r} per financial_ontology.yaml "
                    f"valid_metric_template_constraints — spec is unsatisfiable"
                )
    return problems


def _filter_metric_pool(template: dict, pool_name: str, base_pool: list, lookups: dict) -> list:
    """Drop metric ids that financial_ontology.yaml explicitly marks incompatible
    with this template's execution-plan family.

    Only disallowed_template_families is treated as authoritative.
    allowed_template_families is NOT used as a strict allowlist — it is not
    exhaustively maintained (e.g. funds_mobilized's allowed list omits
    average_over_range even though "average funds mobilized in 2023" is a
    perfectly valid query), so honoring it as exhaustive would silently
    reject valid metric/template combinations.
    """
    if pool_name not in ("metric", "metric_snapshot"):
        return base_pool
    family = FAMILY_BY_EXECUTION_PLAN.get(template["execution_plan"])
    if family is None:
        return base_pool
    constraints = lookups["valid_metric_template_constraints"]
    return [
        metric_id for metric_id in base_pool
        if family not in (constraints.get(metric_id) or {}).get("disallowed_template_families", [])
    ]


# ── Variable resolution ─────────────────────────────────────────────────────

def _build_pools(docs: dict, lookups: dict) -> dict[str, list]:
    tv = docs["templates.yaml"]["template_variables"]
    return {
        "metric": tv["metric"],
        "metric_snapshot": tv["metric_snapshot"],
        "scheme_type": [e["entity_id"] for e in lookups["entities_by_group"].get("scheme_type", [])],
        "amc": [e["entity_id"] for e in lookups["entities_by_group"].get("amc", [])],
        "year": tv["year"],
        "month": tv["month"],
        "quarter": tv["quarter"],
        "top_n": tv["top_n"],
        "future_year": _FUTURE_YEARS,
    }


def _display_text(pool_name: str, value, lookups: dict) -> str:
    if pool_name in ("metric", "metric_snapshot"):
        return lookups["metrics_by_id"][value]["canonical_name"]
    if pool_name == "scheme_type":
        entity = next(e for e in lookups["entities_by_group"]["scheme_type"] if e["entity_id"] == value)
        # Every template appends its own "fund(s)"/"category" noun after
        # {scheme_type}, so an entity whose canonical_name already ends in
        # "Fund(s)" (e.g. "Index Funds") would otherwise read "Index Funds
        # funds". Strip the redundant suffix at substitution time only.
        name = entity["canonical_name"]
        name = re.sub(r"\s+Funds?$", "", name)
        return name
    if pool_name == "amc":
        entity = next(e for e in lookups["entities_by_group"]["amc"] if e["entity_id"] == value)
        return entity["canonical_name"]
    return str(value)


def _extract_placeholders(text: str) -> list[str]:
    return _PLACEHOLDER_RE.findall(text)


def _resolve_one(rng: random.Random, pools: dict, placeholder: str):
    pool_name = _VAR_POOL[placeholder]
    return rng.choice(pools[pool_name])


def _resolve_pair(rng: random.Random, pools: dict, a: str, b: str, constraint: str, attempts: int = 30):
    pool_name = _VAR_POOL[a]
    for _ in range(attempts):
        va = rng.choice(pools[pool_name])
        vb = rng.choice(pools[pool_name])
        if constraint == "distinct" and va != vb:
            return va, vb
        if constraint == "ordered" and va < vb:
            return va, vb
    # Fall back to a deterministic valid pair drawn from the pool's extremes.
    ordered_pool = sorted(pools[pool_name])
    if constraint == "ordered":
        return ordered_pool[0], ordered_pool[-1]
    return ordered_pool[0], ordered_pool[-1] if len(ordered_pool) > 1 else ordered_pool[0]


def resolve_instance(rng: random.Random, pools: dict, placeholders: list[str]) -> dict[str, tuple]:
    """Return {placeholder: (raw_value, pool_name)} for one sampled combination."""
    resolved: dict[str, tuple] = {}
    remaining = set(placeholders)

    for (pa, pb), constraint in _PAIR_CONSTRAINTS.items():
        if pa in remaining and pb in remaining:
            va, vb = _resolve_pair(rng, pools, pa, pb, constraint)
            resolved[pa] = (va, _VAR_POOL[pa])
            resolved[pb] = (vb, _VAR_POOL[pb])
            remaining -= {pa, pb}

    # Sorted, not raw set iteration: string hashing is randomized per-process,
    # so iterating `remaining` directly would draw from `rng` in a different
    # order each run and silently break seed reproducibility.
    for ph in sorted(remaining):
        resolved[ph] = (_resolve_one(rng, pools, ph), _VAR_POOL[ph])

    return resolved


# ── Corpus / label building ─────────────────────────────────────────────────

def _derive_period(resolved: dict[str, tuple]) -> str | None:
    years = [v for ph, (v, pool) in resolved.items() if pool == "year"]
    if not years:
        return None
    lo, hi = min(years), max(years)
    if hi < 2020:
        return "pre_2020"
    if lo >= 2020:
        return "post_2020"
    return "cross_period"


def _derive_category(template: dict, lookups: dict) -> str:
    cap_ids = template.get("capabilities") or []
    if not cap_ids:
        return "uncategorized"
    domain_name = lookups["capabilities_by_id"][cap_ids[0]]["domain_name"]
    return re.sub(r"[^a-z0-9]+", "_", domain_name.lower()).strip("_")


def generate_template_instances(
    template: dict,
    pools: dict,
    lookups: dict,
    rng: random.Random,
    count: int,
) -> list[dict]:
    # Per-template pools: financial_ontology.yaml's valid_metric_template_constraints
    # can narrow the metric/metric_snapshot pool for this specific execution_plan
    # (e.g. AUM/folios/schemes are dropped from range_sum templates — you can't
    # meaningfully SUM a stock or count metric across periods).
    template_pools = dict(pools)
    template_pools["metric"] = _filter_metric_pool(template, "metric", pools["metric"], lookups)
    template_pools["metric_snapshot"] = _filter_metric_pool(
        template, "metric_snapshot", pools["metric_snapshot"], lookups
    )

    text_pool = [template["pattern"], *template.get("variants", [])]
    seen: set[tuple] = set()
    instances = []
    attempts = 0
    max_attempts = max(count * 20, 40)

    while len(instances) < count and attempts < max_attempts:
        attempts += 1
        text_template = text_pool[len(instances) % len(text_pool)]
        placeholders = _extract_placeholders(text_template)
        resolved = resolve_instance(rng, template_pools, placeholders)

        combo_key = (text_template, tuple(sorted((k, v[0]) for k, v in resolved.items())))
        if combo_key in seen:
            continue
        seen.add(combo_key)

        display_values = {ph: _display_text(pool, val, lookups) for ph, (val, pool) in resolved.items()}
        query_text = text_template.format(**display_values)

        instances.append({
            "query": query_text,
            "template_id": template["template_id"],
            "resolved": resolved,
            "period": _derive_period(resolved),
        })

    return instances


def build_corpus_entry(qid: str, inst: dict, template: dict, lookups: dict) -> dict:
    tags = list(template.get("capabilities") or [])
    for ph, (val, pool) in inst["resolved"].items():
        tags.append(str(val))
    tags.append(inst["template_id"])

    return {
        "id": qid,
        "query": inst["query"],
        "category": _derive_category(template, lookups),
        "subcategory": template["execution_plan"],
        "period": inst["period"],
        "path": template["expected_route"],
        "difficulty": template["difficulty"],
        "tags": tags,
    }


def build_intent_label(inst: dict, template: dict) -> dict:
    resolved = inst["resolved"]

    def _first(pool_name):
        for val, pool in resolved.values():
            if pool == pool_name:
                return val
        return None

    year = None
    year_from = year_to = None
    years = [v for v, pool in resolved.values() if pool == "year"]
    future_years = [v for v, pool in resolved.values() if pool == "future_year"]
    if "start_year" in resolved and "end_year" in resolved:
        year_from, year_to = resolved["start_year"][0], resolved["end_year"][0]
    elif "year_a" in resolved and "year_b" in resolved:
        # Time-comparison templates (e.g. T_COMPARE_002) don't have a single
        # "year" slot — represent the compared pair as a range for downstream
        # consumers that key off year_from/year_to.
        year_from, year_to = sorted((resolved["year_a"][0], resolved["year_b"][0]))
    elif len(years) == 1:
        year = years[0]
    elif future_years:
        year = future_years[0]

    # Verified against the live classifier's actual (established, hand-authored-
    # corpus-consistent) extraction behavior, not just a generic id->text guess:
    # "Sectoral/Thematic" text extracts as two separate tags (SCHEME_TYPES has
    # no combined entry), and "All Categories"/industry-wide mentions extract
    # as no scheme_type at all (no filter needed for an industry-wide query).
    _SCHEME_TYPE_LABEL_OVERRIDES = {
        "sectoral_thematic": ["sectoral", "thematic"],
        "all_categories": [],
    }
    scheme_types = []
    for ph, (val, pool) in resolved.items():
        if pool == "scheme_type":
            scheme_types.extend(_SCHEME_TYPE_LABEL_OVERRIDES.get(val, [val.replace("_", " ")]))

    metric = _first("metric") or _first("metric_snapshot")
    month = _first("month")

    return {
        "intent_type": template["expected_intent"],
        "metric": metric,
        "year": year,
        "month": month,
        "year_from": year_from,
        "year_to": year_to,
        "needs_analytical": template["expected_route"] == "plan_years",
        "scheme_types": scheme_types,
        "routing": template["expected_route"],
    }


# ── Versioning & output ─────────────────────────────────────────────────────

def next_version_dir(base_dir: Path) -> tuple[Path, int]:
    base_dir.mkdir(parents=True, exist_ok=True)
    existing = [int(m.group(1)) for p in base_dir.iterdir() if p.is_dir()
                for m in [re.fullmatch(r"v(\d+)", p.name)] if m]
    n = max(existing, default=0) + 1
    return base_dir / f"v{n}", n


def generate_corpus(
    ontology_dir: Path = ONTOLOGY_DIR,
    per_template: int = 10,
    seed: int = 42,
    include_unimplemented: bool = False,
) -> tuple[dict, dict, dict]:
    """Return (query_corpus, intent_labels, manifest) without writing anything."""
    docs = load_yaml_files(ontology_dir)
    lookups = build_lookup_tables(docs)

    problems = validate_cross_references(docs, lookups)
    problems += validate_ontology_constraints(docs, lookups)
    if problems:
        raise ValueError("Spec validation failed:\n" + "\n".join(f"  - {p}" for p in problems))

    pools = _build_pools(docs, lookups)
    rng = random.Random(seed)

    queries = []
    intent_labels = {}
    categories: dict[str, list[int]] = {}
    skipped_templates = []
    metric_filtering: dict[str, list[str]] = {}
    counter = 0

    for template in docs["templates.yaml"]["templates"]:
        if template.get("status") == "unimplemented" and not include_unimplemented:
            skipped_templates.append(template["template_id"])
            continue

        placeholders = {ph for text in [template["pattern"], *template.get("variants", [])]
                        for ph in _PLACEHOLDER_RE.findall(text)}
        for pool_name in ("metric", "metric_snapshot"):
            if not any(_VAR_POOL.get(ph) == pool_name for ph in placeholders):
                continue
            excluded = sorted(set(pools[pool_name]) - set(_filter_metric_pool(template, pool_name, pools[pool_name], lookups)))
            if excluded:
                metric_filtering[f"{template['template_id']}.{pool_name}"] = excluded

        instances = generate_template_instances(template, pools, lookups, rng, per_template)
        for inst in instances:
            counter += 1
            qid = f"G{counter:04d}"
            entry = build_corpus_entry(qid, inst, template, lookups)
            queries.append(entry)
            intent_labels[qid] = build_intent_label(inst, template)
            categories.setdefault(f"{entry['category']}/{entry['subcategory']}", []).append(counter)

    query_corpus = {
        "_meta": {
            "version": "1.0",
            "generator": "fies/generator/query_generator.py",
            "seed": seed,
            "per_template": per_template,
            "total": len(queries),
            "categories": {
                k: f"G{v[0]:04d}-G{v[-1]:04d}" if len(v) > 1 else f"G{v[0]:04d}"
                for k, v in categories.items()
            },
        },
        "queries": queries,
    }

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "per_template": per_template,
        "total_queries": len(queries),
        "total_templates": len(docs["templates.yaml"]["templates"]),
        "skipped_templates": skipped_templates,
        "source_files": {
            **{f"fies/ontology/{name}": _file_hash(ontology_dir, name) for name in SOURCE_FILES},
            **{f"domain/semantic/{name}": _file_hash(SEMANTIC_DIR, name) for name in SEMANTIC_LAYER_FILES},
        },
        "ontology_metric_filtering": metric_filtering,
        "not_generated": [
            "sql_labels.json — requires real DB row counts/values",
            "retrieval_labels.json — requires real retrieved chunk IDs",
            "expected_answers.json — requires real expected facts/citations",
        ],
    }

    return query_corpus, intent_labels, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile fies/ontology/*.yaml into a versioned query corpus")
    parser.add_argument("--per-template", type=int, default=10,
                         help="Sampled instances per template (default: 10)")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed for reproducibility")
    parser.add_argument("--ontology-dir", type=Path, default=ONTOLOGY_DIR)
    parser.add_argument("--out-base", type=Path, default=GENERATED_DIR)
    parser.add_argument("--dry-run", action="store_true", help="Validate and report counts without writing")
    parser.add_argument("--include-unimplemented", action="store_true",
                         help="Also generate queries for templates marked status: unimplemented "
                              "(off by default — those capabilities have no working route yet)")
    args = parser.parse_args()

    query_corpus, intent_labels, manifest = generate_corpus(
        ontology_dir=args.ontology_dir, per_template=args.per_template, seed=args.seed,
        include_unimplemented=args.include_unimplemented,
    )

    print(f"[query_generator] {manifest['total_templates']} templates -> "
          f"{manifest['total_queries']} queries (seed={args.seed}, per_template={args.per_template})")
    for cat, span in query_corpus["_meta"]["categories"].items():
        print(f"  {cat:<45} {span}")
    if manifest["skipped_templates"]:
        print(f"[query_generator] Skipped (status: unimplemented): {', '.join(manifest['skipped_templates'])}")
    if manifest["ontology_metric_filtering"]:
        print("[query_generator] domain/semantic/financial_ontology.yaml excluded these metric/template combinations:")
        for key, excluded in manifest["ontology_metric_filtering"].items():
            print(f"  {key:<35} excluded: {', '.join(excluded)}")

    if args.dry_run:
        print("[query_generator] --dry-run: nothing written")
        return

    version_dir, n = next_version_dir(args.out_base)
    version_dir.mkdir(parents=True, exist_ok=True)

    (version_dir / "query_corpus.json").write_text(json.dumps(query_corpus, indent=2))
    (version_dir / "intent_labels.json").write_text(json.dumps(intent_labels, indent=2))
    (version_dir / "_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[query_generator] Wrote v{n} -> {version_dir}")


if __name__ == "__main__":
    main()
