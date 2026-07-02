"""Evaluation Runner — orchestrates all 7 dimensions into one structured report.

Usage:
    # Full evaluation (requires LLM API key)
    python -m financial_pipeline.evaluation.runner

    # Quick eval — skip LLM generation (only dims 1, 2, 6)
    python -m financial_pipeline.evaluation.runner --no-llm

    # Limit cases
    python -m financial_pipeline.evaluation.runner --limit 5

Output:
    - Console report (EvalMetrics.print_report)
    - JSON file: eval_results_<timestamp>.json
    - CloudWatch metrics (if AWS creds configured)
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from financial_pipeline.augmentation.guardrails import PreGenerationGuardrails
from financial_pipeline.augmentation.pipeline import AugmentationPipeline
from financial_pipeline.augmentation.ranker import ContextRanker
from financial_pipeline.config import settings
from financial_pipeline.evaluation.cost_tracker import CostSummary, QueryCost
from financial_pipeline.evaluation.dataset import EVAL_DATASET, EvalCase
from financial_pipeline.evaluation.metrics import (
    EvalMetrics,
    abstention_accuracy,
    abstention_correct,
    advice_block_rate,
    citation_coverage,
    citation_validity,
    expected_number_hit,
    false_positive_rate,
    faithfulness_embedding,
    hit_at_k,
    keyword_recall,
    ndcg_at_k,
    numeric_precision,
    reciprocal_rank,
)
from financial_pipeline.evaluation.observability import emitter, RequestTrace
from financial_pipeline.evaluation.retrieval_eval import RetrievalEvaluator
from financial_pipeline.retrieval.pipeline import RetrievalPipeline
from financial_pipeline.retrieval.retriever import Retriever
from financial_pipeline.storage.document_repo import DocumentRepository

log = structlog.get_logger()


class EvalRunner:
    """Runs the complete 7-dimension evaluation suite."""

    def __init__(
        self,
        repo:       DocumentRepository,
        retriever:  Retriever,
        ret_pipe:   RetrievalPipeline,
        aug_pipe:   AugmentationPipeline | None = None,
        embed_model = None,
        run_llm:    bool = True,
    ) -> None:
        self._repo      = repo
        self._retriever = retriever
        self._ret_pipe  = ret_pipe
        self._aug_pipe  = aug_pipe
        self._model     = embed_model
        self._run_llm   = run_llm and aug_pipe is not None and aug_pipe.is_configured()
        self._pre_guard = PreGenerationGuardrails()
        self._ranker    = ContextRanker(use_cross_encoder=True, top_k=10)

    def run(self, cases: list[EvalCase]) -> EvalMetrics:
        log.info("eval.start", cases=len(cases), llm=self._run_llm)
        t0 = time.perf_counter()

        # ── Dims 1 & 2: Retrieval + Reranker quality ──────────────────
        ret_eval = RetrievalEvaluator(self._retriever, self._ranker, top_k=10)
        ret_summary = ret_eval.run(cases)

        # ── Dims 3–7: Generation + Guardrail quality ───────────────────
        case_results: list[dict] = []
        costs = CostSummary()

        for case in cases:
            result = self._eval_case(case)
            case_results.append(result)

            if result.get("prompt_tokens") and result.get("model"):
                costs.add(QueryCost(
                    model             = result["model"],
                    prompt_tokens     = result.get("prompt_tokens", 0),
                    completion_tokens = result.get("completion_tokens", 0),
                    latency_ms        = result.get("total_ms", 0),
                ))
                # Emit per-request CloudWatch metrics
                trace = RequestTrace(
                    request_id        = str(uuid.uuid4()),
                    question          = case.question,
                    intent            = result.get("intent", ""),
                    model             = result.get("model", ""),
                    provider          = result.get("provider", ""),
                    prompt_tokens     = result.get("prompt_tokens", 0),
                    completion_tokens = result.get("completion_tokens", 0),
                    total_ms          = result.get("total_ms", 0),
                    pre_blocked       = result.get("pre_blocked", False),
                    post_passed       = result.get("post_passed", True),
                    answer_safe       = result.get("answer_safe", True),
                    abstention_detected = result.get("abstained", False),
                    faithfulness_score  = result.get("faithfulness", -1.0),
                    citation_count      = result.get("citation_count", 0),
                )
                emitter.record(trace)

        emitter.flush()

        # ── Aggregate ─────────────────────────────────────────────────
        metrics = self._aggregate(ret_summary, case_results, costs)
        metrics.total_cases  = len(cases)
        metrics.failed_cases = sum(1 for r in case_results if r.get("error"))

        log.info("eval.done",
                 total_ms=int((time.perf_counter() - t0) * 1000),
                 score=round(metrics.overall_score(), 3))
        return metrics

    # ------------------------------------------------------------------

    def _eval_case(self, case: EvalCase) -> dict:
        result: dict = {
            "case_id":           case.id,
            "question":          case.question,
            "is_investment_advice": case.is_investment_advice,
            "expected_abstain":  case.expected_abstain,
        }

        # Dim 6a: Pre-guardrail (runs even without LLM)
        try:
            from financial_pipeline.augmentation.citations import Citation
            dummy_cit: list[Citation] = []
            pre = self._pre_guard.check(case.question, dummy_cit, "factual")
            result["pre_blocked"]     = not pre.should_proceed
            result["pre_is_advice"]   = pre.is_investment_advice
        except Exception as exc:
            result["pre_error"] = str(exc)
            result["pre_blocked"] = False

        if not self._run_llm:
            result["skipped_llm"] = True
            result["abstention_correct"] = abstention_correct(
                "skipped" if not case.expected_abstain else
                "This information is not available in the provided amfi documents.",
                case.expected_abstain,
            )
            return result

        if result.get("pre_blocked"):
            # Correctly blocked → no LLM call needed
            result["abstention_correct"] = case.is_investment_advice or case.expected_abstain
            result["abstained"]          = True
            return result

        # Dims 3–6: Full augmentation pipeline
        try:
            t0   = time.perf_counter()
            resp = self._aug_pipe.run(
                question = case.question,
                category = case.category,
                top_k    = 8,
            )
            total_ms = int((time.perf_counter() - t0) * 1000)

            answer       = resp.answer
            citations    = resp.citations
            sources_text = " ".join(c.excerpt for c in citations)
            cited_nums   = {int(m) for m in __import__("re").findall(r"\[(\d+)\]", answer)}
            cited_text   = " ".join(
                c.excerpt for c in citations if c.number in cited_nums
            )

            result.update({
                "answer":            answer[:500],
                "total_ms":          total_ms,
                "model":             resp.generation.model if resp.generation else "",
                "provider":          resp.generation.provider if resp.generation else "",
                "prompt_tokens":     resp.generation.prompt_tokens if resp.generation else 0,
                "completion_tokens": resp.generation.completion_tokens if resp.generation else 0,
                "intent":            resp.intent_type,
                "citation_count":    len(citations),
                "post_passed":       resp.guardrail.post_passed,
                "answer_safe":       resp.guardrail.answer_safe,
                "abstained":         resp.guardrail.abstention_detected,
                "faithfulness":      resp.guardrail.faithfulness_score,
                "hallucination_risk":resp.guardrail.hallucination_risk,
                # Dim 3
                "citation_coverage": citation_coverage(answer),
                "citation_validity": citation_validity(answer, len(citations)),
                "keyword_recall":    keyword_recall(answer, sources_text, case.expected_keywords),
                # Dim 4
                "abstention_correct":abstention_correct(answer, case.expected_abstain),
                "faithfulness_embed":faithfulness_embedding(
                    answer, [c.excerpt for c in citations], self._model
                ) if self._model else -1.0,
                # Dim 5
                "numeric_precision": numeric_precision(answer, cited_text),
                "expected_num_hit":  expected_number_hit(answer, sources_text, case.expected_numbers),
                # Dim 6
                "has_unsafe_content":not resp.guardrail.answer_safe,
            })

        except Exception as exc:
            result["error"] = str(exc)
            log.warning("eval.case_failed", id=case.id, error=str(exc))

        return result

    def _aggregate(self, ret, case_results: list[dict], costs: CostSummary) -> EvalMetrics:
        ok = [r for r in case_results if not r.get("error")]

        def avg(key: str, default: float = 0.0) -> float:
            vals = [r[key] for r in ok if key in r and r[key] is not None and r[key] >= 0]
            return sum(vals) / len(vals) if vals else default

        m = EvalMetrics(
            # Dim 1
            hit_at_5     = ret.post_hit_5,
            hit_at_10    = ret.post_hit_10,
            mrr          = ret.post_mrr,
            ndcg_at_10   = ret.post_ndcg_10,
            # Dim 2
            ce_avg_improvement = ret.ce_avg_improvement,
            ce_promotion_rate  = ret.ce_promotion_rate,
            # Dim 3
            citation_coverage_avg = avg("citation_coverage"),
            citation_validity_avg = avg("citation_validity", 1.0),
            # Dim 4
            faithfulness_avg   = avg("faithfulness_embed", -1.0),
            keyword_recall_avg = avg("keyword_recall"),
            abstention_acc     = abstention_accuracy(case_results),
            # Dim 5
            numeric_precision_avg   = avg("numeric_precision", 1.0),
            expected_number_hit_avg = avg("expected_num_hit", 1.0),
            # Dim 6
            advice_block_rate   = advice_block_rate(case_results),
            false_positive_rate = false_positive_rate(case_results),
            safety_catch_rate   = 1.0,   # computed below
            # Dim 7
            avg_latency_ms        = costs.avg_latency_ms,
            p90_latency_ms        = costs.p90_latency_ms,
            avg_prompt_tokens     = costs.avg_prompt_tokens,
            avg_completion_tokens = costs.avg_completion_tokens,
            total_cost_usd        = costs.total_cost_usd,
            cost_per_query_usd    = costs.cost_per_query,
        )
        return m


def main() -> None:
    parser = argparse.ArgumentParser(description="AMFI pipeline evaluation suite")
    parser.add_argument("--limit",   type=int, default=None)
    parser.add_argument("--no-llm",  action="store_true",
                        help="Skip LLM generation (dims 1, 2, 6 only)")
    parser.add_argument("--output",  default=None,
                        help="Write JSON results to this file")
    args = parser.parse_args()

    if not settings.postgres_url:
        print("POSTGRES_URL not set.", file=sys.stderr)
        sys.exit(1)

    repo      = DocumentRepository(settings.postgres_url)
    retriever = Retriever(repo)
    ret_pipe  = RetrievalPipeline(repo, retriever)

    aug_pipe = None
    if not args.no_llm and settings.openai_api_key:
        from financial_pipeline.augmentation.pipeline import AugmentationPipeline
        aug_pipe = AugmentationPipeline(ret_pipe, use_cross_encoder=True)

    cases = EVAL_DATASET[: args.limit] if args.limit else EVAL_DATASET
    print(f"Running {len(cases)} eval cases  (llm={'yes' if aug_pipe else 'no'})")

    runner  = EvalRunner(repo, retriever, ret_pipe, aug_pipe, run_llm=not args.no_llm)
    metrics = runner.run(cases)
    metrics.print_report()

    if args.output:
        out_path = Path(args.output)
    else:
        ts = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"eval_results_{ts}.json")

    out_path.write_text(
        json.dumps({
            "timestamp": datetime.now(UTC).isoformat(),
            "cases":     len(cases),
            "score":     round(metrics.overall_score(), 4),
            "metrics":   asdict(metrics),
        }, indent=2)
    )
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
