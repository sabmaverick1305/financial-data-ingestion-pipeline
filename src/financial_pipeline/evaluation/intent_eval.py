"""Intent Classification Accuracy Evaluator.

Tests the 5-stage intent pipeline (QueryAnalyzer) against a labeled
query set and reports:

  1. Per-class accuracy  (factual / trend / definition / comparison / …)
  2. Stage distribution  (% queries resolved at rule / embedding / llm)
  3. needs_analytical accuracy for year-range queries
  4. Metric extraction accuracy (aum / nav / folios / sip / …)
  5. Confidence calibration (are high-confidence predictions correct more often?)

Usage:
    python -m financial_pipeline.evaluation.intent_eval
    python -m financial_pipeline.evaluation.intent_eval --no-embedding   # rule only
    python -m financial_pipeline.evaluation.intent_eval --no-llm         # rule + embedding
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

log = structlog.get_logger()


# ── Labeled test cases ────────────────────────────────────────────────────────

@dataclass
class IntentCase:
    query:              str
    expected_intent:    str                  # factual / trend / definition / comparison / regulatory / lookup / tabular
    expected_metric:    str | None = None    # aum / nav / folios / funds_mobilized / sip / redemption
    expected_analytical:bool = False         # needs_analytical should be True
    expected_year_from: int | None = None
    expected_year_to:   int | None = None
    notes:              str = ""


INTENT_CASES: list[IntentCase] = [
    # ── Definition ────────────────────────────────────────────────────────────
    IntentCase("What is a large cap fund?",                     "definition"),
    IntentCase("Define expense ratio",                          "definition"),
    IntentCase("What does NAV stand for?",                      "definition"),
    IntentCase("Explain the difference between equity and debt funds", "definition"),

    # ── Factual (value queries — "what is THE <metric>") ─────────────────────
    IntentCase("What is the NAV of HDFC Large Cap Fund?",       "factual",  "nav"),
    IntentCase("What is the AUM of SBI Bluechip?",              "factual",  "aum"),
    IntentCase("What is the total AUM in FY2024?",              "factual",  "aum"),
    IntentCase("What is the total AUM as of June 2025?",        "factual",  "aum"),
    IntentCase("What is the current NAV of Mirae Asset?",       "factual",  "nav"),

    # ── Trend / analytical (year-range) ──────────────────────────────────────
    # Year-range queries → intent_type=analytical (stage 1 sets this explicitly)
    IntentCase(
        "What was the total funds mobilized from 2020 to 2023?",
        "analytical", "funds_mobilized",
        expected_analytical=True, expected_year_from=2020, expected_year_to=2023,
    ),
    IntentCase(
        "Show me the AUM trend from 2021 to 2024",
        "analytical", "aum",
        expected_analytical=True, expected_year_from=2021, expected_year_to=2024,
    ),
    IntentCase(
        "How did SIP contributions change between 2019 and 2022?",
        "analytical", "sip",
        expected_analytical=True, expected_year_from=2019, expected_year_to=2022,
        notes="stage1 sets analytical for year-range; SIP metric may escalate to embedding",
    ),
    IntentCase(
        "What were the total folios from 2022 to 2025?",
        "analytical", "folios",
        expected_analytical=True, expected_year_from=2022, expected_year_to=2025,
    ),
    IntentCase(
        "funds mobilized trend 2018 to 2020",
        "analytical", "funds_mobilized",
        expected_analytical=True, expected_year_from=2018, expected_year_to=2020,
    ),

    # ── Single-year trend (not analytical, just factual/tabular) ─────────────
    IntentCase("What is the AUM for 2023?",                     "factual",  "aum"),
    IntentCase("Total funds mobilized in March 2024",           "factual",  "funds_mobilized"),

    # ── Comparison ────────────────────────────────────────────────────────────
    IntentCase("Compare equity vs debt AUM",                    "comparison", "aum"),
    IntentCase("How does HDFC AUM compare to SBI AUM?",         "comparison", "aum"),
    IntentCase("Equity versus debt fund performance",           "comparison"),

    # ── Tabular / ranking ─────────────────────────────────────────────────────
    IntentCase("Which fund has the highest AUM?",               "tabular",  "aum"),
    IntentCase("Top 5 funds by AUM",                            "tabular",  "aum"),
    IntentCase("Lowest NAV among large cap funds",              "tabular",  "nav"),
    IntentCase("Rank mutual funds by folios",                   "tabular",  "folios"),

    # ── Regulatory ────────────────────────────────────────────────────────────
    IntentCase("What SEBI guidelines were issued for derivatives?", "regulatory"),
    IntentCase("What are the AMFI regulations for expense ratio?",  "regulatory"),

    # ── Lookup (point-in-time) ────────────────────────────────────────────────
    # Stage 1 classifies these as "factual"; embedding/LLM promotes to "lookup".
    # Correct rule-level expected_intent is "factual"; needs_analytical=False.
    IntentCase("Total folios as of May 2026",                   "factual",  "folios",
               notes="Promoted to lookup at embedding/LLM stage; factual at rule stage"),
    IntentCase("Total AUM for January 2025",                    "factual",  "aum",
               notes="Promoted to lookup at embedding/LLM stage; factual at rule stage"),
    IntentCase("Funds mobilized in December 2023",              "factual",  "funds_mobilized",
               notes="Promoted to lookup at embedding/LLM stage; factual at rule stage"),

    # ── Tricky edge cases ──────────────────────────────────────────────────────
    IntentCase(
        "what is the nav of hdfc large cap",                    "factual",  "nav",
        notes="'what is the' + metric present → factual, not definition",
    ),
    IntentCase(
        "what is the total aum in fy2024",                      "factual",  "aum",
        notes="'what is the' + FY year → factual, not definition",
    ),
    IntentCase(
        "tell me about the aum growth",                         "trend",    "aum",
        notes="search_query stripping: 'tell me about' removed",
    ),
    IntentCase(
        "show me the total number of folios",                   "tabular",  "folios",
        notes="'number of' triggers table signal; tabular at rule stage",
    ),
]


# ── Results ───────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    query:              str
    expected_intent:    str
    predicted_intent:   str
    expected_metric:    str | None
    predicted_metric:   str | None
    expected_analytical:bool
    predicted_analytical:bool
    expected_year_from: int | None
    predicted_year_from:int | None
    expected_year_to:   int | None
    predicted_year_to:  int | None
    confidence:         float
    stage:              str
    intent_correct:     bool
    metric_correct:     bool    # True if expected_metric is None or matches predicted
    analytical_correct: bool
    notes:              str = ""


@dataclass
class IntentEvalReport:
    total:              int
    intent_accuracy:    float
    metric_accuracy:    float
    analytical_accuracy:float
    per_class:          dict[str, dict] = field(default_factory=dict)  # intent → {correct, total, acc}
    stage_dist:         dict[str, int]  = field(default_factory=dict)  # stage → count
    conf_calibration:   list[tuple[float, float]] = field(default_factory=list)  # (avg_conf, acc) per bucket
    results:            list[CaseResult] = field(default_factory=list)

    def print_report(self) -> None:
        grade = "A" if self.intent_accuracy >= 0.90 else "B" if self.intent_accuracy >= 0.75 else "C"
        print(f"\n{'═'*60}")
        print(f"  Intent Classification Evaluation")
        print(f"  Overall intent accuracy : {self.intent_accuracy:.1%}  ({grade})")
        print(f"  Metric extraction acc   : {self.metric_accuracy:.1%}")
        print(f"  needs_analytical acc    : {self.analytical_accuracy:.1%}")
        print(f"  Total cases             : {self.total}")
        print(f"{'═'*60}")

        print(f"\n  Per-class accuracy:")
        for intent, stats in sorted(self.per_class.items()):
            bar = "█" * stats["correct"] + "░" * (stats["total"] - stats["correct"])
            print(f"    {intent:<14} {stats['acc']:.0%}  [{bar}]  {stats['correct']}/{stats['total']}")

        print(f"\n  Stage distribution:")
        total = sum(self.stage_dist.values()) or 1
        for stage, count in sorted(self.stage_dist.items()):
            pct = count / total
            bar = "█" * int(pct * 20)
            print(f"    {stage:<12} {pct:.0%}  {bar}  ({count})")

        print(f"\n  Confidence calibration (buckets):")
        for avg_conf, acc in self.conf_calibration:
            gap = abs(avg_conf - acc)
            flag = "⚠" if gap > 0.15 else " "
            print(f"    conf~{avg_conf:.2f}  acc={acc:.0%}  gap={gap:.2f} {flag}")

        print(f"\n  Failures:")
        failures = [r for r in self.results if not r.intent_correct]
        if not failures:
            print("    none — all intent types correct")
        for r in failures:
            print(f"    ✗ [{r.expected_intent}→{r.predicted_intent}]  conf={r.confidence:.2f}  stage={r.stage}")
            print(f"      Q: {r.query[:70]}")
            if r.notes:
                print(f"      note: {r.notes}")
        print(f"{'═'*60}\n")


# ── Evaluator ─────────────────────────────────────────────────────────────────

class IntentEvaluator:
    """Runs QueryAnalyzer against INTENT_CASES and produces IntentEvalReport."""

    def __init__(
        self,
        embed_fn:  Callable[[str], list[float]] | None = None,
        generator=None,
    ) -> None:
        from financial_pipeline.retrieval.query_understanding import QueryAnalyzer
        self._analyzer = QueryAnalyzer(embed_fn=embed_fn, generator=generator)

    def run(self, cases: list[IntentCase] | None = None) -> IntentEvalReport:
        cases = cases or INTENT_CASES
        results: list[CaseResult] = []

        for case in cases:
            intent = self._analyzer.analyze(case.query)
            predicted_metric = getattr(intent, "metric", None)
            predicted_year_from = getattr(intent, "year_from", None)
            predicted_year_to   = getattr(intent, "year_to", None)
            needs_analytical    = getattr(intent, "needs_analytical", False)
            confidence          = getattr(intent, "confidence", 0.0)
            stage               = getattr(intent, "stage_used", "rule")

            intent_ok     = intent.intent_type == case.expected_intent
            metric_ok     = (case.expected_metric is None or
                             predicted_metric == case.expected_metric)
            analytical_ok = needs_analytical == case.expected_analytical

            results.append(CaseResult(
                query               = case.query,
                expected_intent     = case.expected_intent,
                predicted_intent    = intent.intent_type,
                expected_metric     = case.expected_metric,
                predicted_metric    = predicted_metric,
                expected_analytical = case.expected_analytical,
                predicted_analytical= needs_analytical,
                expected_year_from  = case.expected_year_from,
                predicted_year_from = predicted_year_from,
                expected_year_to    = case.expected_year_to,
                predicted_year_to   = predicted_year_to,
                confidence          = confidence,
                stage               = stage,
                intent_correct      = intent_ok,
                metric_correct      = metric_ok,
                analytical_correct  = analytical_ok,
                notes               = case.notes,
            ))

        return self._aggregate(results)

    def _aggregate(self, results: list[CaseResult]) -> IntentEvalReport:
        n = len(results)
        intent_acc    = sum(r.intent_correct    for r in results) / n
        metric_cases  = [r for r in results if r.expected_metric is not None]
        metric_acc    = sum(r.metric_correct    for r in metric_cases) / len(metric_cases) if metric_cases else 1.0
        analytical_acc= sum(r.analytical_correct for r in results) / n

        # Per-class accuracy
        per_class: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
        for r in results:
            pc = per_class[r.expected_intent]
            pc["total"] += 1
            if r.intent_correct:
                pc["correct"] += 1
        for v in per_class.values():
            v["acc"] = v["correct"] / v["total"] if v["total"] else 0.0

        # Stage distribution
        stage_dist: dict[str, int] = defaultdict(int)
        for r in results:
            stage_dist[r.stage] += 1

        # Confidence calibration — 5 buckets by confidence
        buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
        for r in results:
            bucket = min(4, int(r.confidence * 5))
            buckets[bucket].append((r.confidence, r.intent_correct))

        calibration = []
        for b in range(5):
            if buckets[b]:
                avg_conf = sum(c for c, _ in buckets[b]) / len(buckets[b])
                acc      = sum(1 for _, ok in buckets[b] if ok) / len(buckets[b])
                calibration.append((round(avg_conf, 3), round(acc, 3)))

        return IntentEvalReport(
            total               = n,
            intent_accuracy     = round(intent_acc, 4),
            metric_accuracy     = round(metric_acc, 4),
            analytical_accuracy = round(analytical_acc, 4),
            per_class           = dict(per_class),
            stage_dist          = dict(stage_dist),
            conf_calibration    = calibration,
            results             = results,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Intent classification accuracy evaluator")
    parser.add_argument("--no-embedding", action="store_true", help="Disable embedding classifier (rule only)")
    parser.add_argument("--no-llm",       action="store_true", help="Disable LLM extractor (rule + embedding)")
    args = parser.parse_args()

    from financial_pipeline.config import settings

    embed_fn  = None
    generator = None

    if not args.no_embedding and settings.postgres_url:
        try:
            from financial_pipeline.retrieval.retriever import Retriever
            from financial_pipeline.storage.document_repo import DocumentRepository
            repo      = DocumentRepository(settings.postgres_url)
            retriever = Retriever(repo, settings.embed_model)
            embed_fn  = retriever._encode
            print(f"Embedding classifier active  (model: {settings.embed_model})")
        except Exception as exc:
            print(f"Could not load embedding model: {exc} — running rule-only")

    if not args.no_llm and embed_fn and settings.openai_api_key:
        try:
            from financial_pipeline.augmentation.generator import AnswerGenerator
            generator = AnswerGenerator()
            print("LLM extractor active")
        except Exception as exc:
            print(f"Could not load LLM generator: {exc} — skipping LLM stage")

    if not embed_fn:
        print("Running rule-only mode (no DB / embedding)")

    evaluator = IntentEvaluator(embed_fn=embed_fn, generator=generator)
    report    = evaluator.run()
    report.print_report()


if __name__ == "__main__":
    main()
