"""Evaluation — Stage 6 of the augmentation layer.

Provides:
  - EvalQuestion  : a ground-truth QA pair with metadata
  - EvalResult    : outcome for one question
  - EvalRunner    : runs the full augmentation pipeline against the dataset
  - Metrics       : faithfulness, relevance, hit-rate, abstention accuracy

The dataset is seeded with real questions derivable from the AMFI corpus.
Run the evaluator to measure pipeline quality before/after changes:

    python -m financial_pipeline.augmentation.evaluation --limit 5

Metrics reported:
  - hit_rate          : fraction of questions where ≥1 relevant source retrieved
  - faithfulness      : average guardrail faithfulness score (0–1)
  - citation_rate     : fraction of answers with valid citations
  - abstention_acc    : "not in docs" response when expected (correct abstention)
  - avg_latency_ms    : end-to-end latency
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger()

# ── Ground-truth dataset ──────────────────────────────────────────────────────
# Questions are calibrated to the AMFI corpus (2009–2025 monthly + quarterly).
# expected_in_docs=False means the corpus cannot answer this → model should
# abstain, not hallucinate.

EVAL_DATASET: list[dict] = [
    # ── Regulatory (quarterly journals have prose) ──────────────────────────
    {
        "id": "reg_001",
        "question": "What SEBI guidelines were issued for derivatives trading by mutual funds?",
        "category": "quarterly",
        "intent": "regulatory",
        "expected_keywords": ["derivatives", "hedging", "stock exchange"],
        "expected_in_docs": True,
        "difficulty": "easy",
    },
    {
        "id": "reg_002",
        "question": "What changes did SEBI make to expense ratio regulations for mutual funds?",
        "category": "quarterly",
        "intent": "regulatory",
        "expected_keywords": ["expense", "TER", "regulations"],
        "expected_in_docs": True,
        "difficulty": "medium",
    },
    {
        "id": "reg_003",
        "question": "What investor education initiatives did AMFI undertake?",
        "category": "quarterly",
        "intent": "factual",
        "expected_keywords": ["investor", "education", "brochure"],
        "expected_in_docs": True,
        "difficulty": "easy",
    },
    # ── Definition (AMFI data docs DON'T define fund types) ─────────────────
    {
        "id": "def_001",
        "question": "What is the difference between a growth plan and a dividend plan?",
        "category": None,
        "intent": "definition",
        "expected_keywords": [],
        "expected_in_docs": False,  # model should abstain
        "difficulty": "hard",
    },
    # ── Trend (quarterly journals have narrative) ────────────────────────────
    {
        "id": "trend_001",
        "question": "How has the mutual fund industry grown over the last decade?",
        "category": "quarterly",
        "intent": "trend",
        "expected_keywords": ["AUM", "growth", "schemes"],
        "expected_in_docs": True,
        "difficulty": "medium",
    },
    # ── Factual from monthly data ────────────────────────────────────────────
    {
        "id": "fact_001",
        "question": "What are the names of balanced advantage funds listed in AMFI data?",
        "category": None,
        "intent": "factual",
        "expected_keywords": ["Balanced Advantage"],
        "expected_in_docs": True,
        "difficulty": "easy",
    },
    {
        "id": "fact_002",
        "question": "Who is the Prime Minister of India?",
        "category": None,
        "intent": "factual",
        "expected_keywords": [],
        "expected_in_docs": False,  # completely out-of-domain, must abstain
        "difficulty": "easy",
    },
]


# ── Dataclasses ───────────────────────────────────────────────────────────────


@dataclass
class EvalQuestion:
    id: str
    question: str
    category: str | None
    intent: str
    expected_keywords: list[str]
    expected_in_docs: bool
    difficulty: str = "medium"


@dataclass
class EvalResult:
    question_id: str
    question: str
    answer: str
    sources_count: int
    hit: bool  # ≥1 expected keyword found in retrieved chunks
    faithfulness: float  # guardrail faithfulness score
    citation_present: bool
    abstention_correct: bool  # model abstained correctly when expected
    guardrail_passed: bool
    latency_ms: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class EvalSummary:
    total: int
    hit_rate: float
    faithfulness_avg: float
    citation_rate: float
    abstention_acc: float
    avg_latency_ms: float
    results: list[EvalResult] = field(default_factory=list)

    def print_report(self) -> None:
        print(f"\n{'=' * 60}")
        print("  Augmentation Pipeline — Evaluation Report")
        print(f"{'=' * 60}")
        print(f"  Questions evaluated : {self.total}")
        print(f"  Hit rate            : {self.hit_rate:.1%}")
        print(f"  Avg faithfulness    : {self.faithfulness_avg:.3f}")
        print(f"  Citation rate       : {self.citation_rate:.1%}")
        print(f"  Abstention accuracy : {self.abstention_acc:.1%}")
        print(f"  Avg latency         : {self.avg_latency_ms:.0f} ms")
        print(f"{'=' * 60}")
        for r in self.results:
            status = "✓" if r.guardrail_passed else "✗"
            print(f"  {status} [{r.question_id}] {r.question[:55]}")
            print(f"    hit={r.hit}  faith={r.faithfulness:.3f}  cite={r.citation_present}  {r.latency_ms}ms")
            if r.warnings:
                print(f"    ⚠ {r.warnings[0]}")
        print()


# ── Runner ────────────────────────────────────────────────────────────────────


class EvalRunner:
    """Runs the augmentation pipeline against the evaluation dataset."""

    def __init__(self, augmentation_pipeline: Any) -> None:
        self._pipeline = augmentation_pipeline

    def run(
        self,
        questions: list[EvalQuestion] | None = None,
        limit: int | None = None,
    ) -> EvalSummary:
        qs = questions or [EvalQuestion(**q) for q in EVAL_DATASET]
        if limit:
            qs = qs[:limit]

        results: list[EvalResult] = []

        for q in qs:
            log.info("eval.running", id=q.id, question=q.question[:60])
            t0 = time.perf_counter()
            try:
                resp = self._pipeline.run(
                    question=q.question,
                    category=q.category,
                )
                latency = int((time.perf_counter() - t0) * 1000)

                # Keyword hit check
                all_text = " ".join(c.excerpt for c in resp.citations).lower()
                hit = not q.expected_keywords or any(kw.lower() in all_text for kw in q.expected_keywords)

                # Abstention check
                abstained = resp.guardrail.abstention_detected
                abs_correct = (q.expected_in_docs and not abstained) or (not q.expected_in_docs and abstained)

                results.append(
                    EvalResult(
                        question_id=q.id,
                        question=q.question,
                        answer=resp.answer[:200],
                        sources_count=len(resp.citations),
                        hit=hit,
                        faithfulness=resp.guardrail.faithfulness_score,
                        citation_present=resp.guardrail.citation_present,
                        abstention_correct=abs_correct,
                        guardrail_passed=resp.guardrail.passed,
                        latency_ms=latency,
                        warnings=resp.guardrail.warnings,
                    )
                )

            except Exception as exc:
                log.warning("eval.question_failed", id=q.id, error=str(exc))
                results.append(
                    EvalResult(
                        question_id=q.id,
                        question=q.question,
                        answer="ERROR",
                        sources_count=0,
                        hit=False,
                        faithfulness=-1.0,
                        citation_present=False,
                        abstention_correct=False,
                        guardrail_passed=False,
                        latency_ms=0,
                        warnings=[str(exc)],
                    )
                )

        def avg(vals: list[float]) -> float:
            valid = [v for v in vals if v >= 0]
            return sum(valid) / len(valid) if valid else 0.0

        summary = EvalSummary(
            total=len(results),
            hit_rate=avg([float(r.hit) for r in results]),
            faithfulness_avg=avg([r.faithfulness for r in results]),
            citation_rate=avg([float(r.citation_present) for r in results]),
            abstention_acc=avg([float(r.abstention_correct) for r in results]),
            avg_latency_ms=avg([float(r.latency_ms) for r in results]),
            results=results,
        )
        return summary


if __name__ == "__main__":
    import argparse
    import sys

    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    from financial_pipeline.augmentation.pipeline import AugmentationPipeline
    from financial_pipeline.config import settings
    from financial_pipeline.retrieval.pipeline import RetrievalPipeline
    from financial_pipeline.retrieval.retriever import Retriever
    from financial_pipeline.storage.document_repo import DocumentRepository

    repo = DocumentRepository(settings.postgres_url)
    retriever = Retriever(repo)
    ret_pipe = RetrievalPipeline(repo, retriever)
    aug_pipe = AugmentationPipeline(ret_pipe)

    runner = EvalRunner(aug_pipe)
    summary = runner.run(limit=args.limit)
    summary.print_report()
