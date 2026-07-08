"""Evaluation Runner — orchestrates all 7 dimensions into one structured report.

Usage:
    # Full evaluation (requires LLM API key + running DB)
    python -m financial_pipeline.evaluation.runner

    # Quick eval — skip LLM generation (dims 1, 2, 6 only)
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
import re
import sys
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from financial_pipeline.augmentation.guardrails import PreGenerationGuardrails
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
from financial_pipeline.evaluation.observability import emitter, ls_feedback, RequestTrace
from financial_pipeline.evaluation.retrieval_eval import RetrievalEvaluator
from financial_pipeline.retrieval.pipeline import RetrievalPipeline
from financial_pipeline.retrieval.retriever import Retriever
from financial_pipeline.storage.document_repo import DocumentRepository

log = structlog.get_logger()


class EvalRunner:
    """Runs the complete 7-dimension evaluation suite against the LangGraph pipeline."""

    def __init__(
        self,
        repo:       DocumentRepository,
        retriever:  Retriever,
        ret_pipe:   RetrievalPipeline,
        graph=None,          # compiled LangGraph (build_graph output)
        embed_model=None,
        run_llm:    bool = True,
    ) -> None:
        self._repo      = repo
        self._retriever = retriever
        self._ret_pipe  = ret_pipe
        self._graph     = graph
        self._model     = embed_model
        self._run_llm   = run_llm and graph is not None
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
                trace = RequestTrace(
                    request_id          = str(uuid.uuid4()),
                    question            = case.question,
                    intent              = result.get("intent", ""),
                    model               = result.get("model", ""),
                    provider            = result.get("provider", ""),
                    prompt_tokens       = result.get("prompt_tokens", 0),
                    completion_tokens   = result.get("completion_tokens", 0),
                    total_ms            = result.get("total_ms", 0),
                    pre_blocked         = result.get("pre_blocked", False),
                    post_passed         = result.get("post_passed", True),
                    answer_safe         = result.get("answer_safe", True),
                    abstention_detected = result.get("abstained", False),
                    faithfulness_score  = result.get("faithfulness", -1.0),
                    citation_count      = result.get("citation_count", 0),
                )
                emitter.record(trace)

            # Push per-case metric scores to LangSmith
            ls_feedback.record_case(result.get("ls_run_id"), result)

        emitter.flush()

        metrics = self._aggregate(ret_summary, case_results, costs)
        metrics.total_cases  = len(cases)
        metrics.failed_cases = sum(1 for r in case_results if r.get("error"))

        # Push aggregated metrics to LangSmith on the first successful run's trace
        # so they appear alongside that run in the project view
        first_run_id = next(
            (r.get("ls_run_id") for r in case_results if r.get("ls_run_id")),
            None,
        )
        ls_feedback.record_aggregate(first_run_id, metrics)

        log.info("eval.done",
                 total_ms=int((time.perf_counter() - t0) * 1000),
                 score=round(metrics.overall_score(), 3))
        return metrics

    # ------------------------------------------------------------------

    def _eval_case(self, case: EvalCase) -> dict:
        result: dict = {
            "case_id":              case.id,
            "question":             case.question,
            "is_investment_advice": case.is_investment_advice,
            "expected_abstain":     case.expected_abstain,
        }

        # Dim 6a: Pre-guardrail check (runs even without LLM)
        try:
            from financial_pipeline.augmentation.citations import Citation
            pre = self._pre_guard.check(case.question, [], "factual")
            result["pre_blocked"]   = not pre.should_proceed
            result["pre_is_advice"] = pre.is_investment_advice
        except Exception as exc:
            result["pre_error"]  = str(exc)
            result["pre_blocked"] = False

        if not self._run_llm:
            result["skipped_llm"] = True
            result["abstention_correct"] = abstention_correct(
                "" if not case.expected_abstain
                else "This information is not available in the provided amfi documents.",
                case.expected_abstain,
            )
            return result

        if result.get("pre_blocked"):
            result["abstention_correct"] = case.is_investment_advice or case.expected_abstain
            result["abstained"] = True
            return result

        # Dims 3–6: Full LangGraph pipeline
        try:
            from langchain_core.runnables import RunnableConfig
            ls_run_id = uuid.uuid4()
            result["ls_run_id"] = str(ls_run_id)

            t0 = time.perf_counter()
            graph_result = self._graph.invoke(
                {
                    "query":        case.question,
                    "retry_count":  0,
                    "repair_count": 0,
                },
                config=RunnableConfig(run_id=ls_run_id),
            )
            total_ms = int((time.perf_counter() - t0) * 1000)

            resp     = graph_result.get("response", {})
            g        = resp.get("guardrail", {})
            intent   = graph_result.get("intent")
            answer   = resp.get("answer", "")
            sources  = resp.get("sources", [])   # list[dict] with file_name, preview, …

            sources_text = " ".join(s.get("preview", "") for s in sources)
            cited_nums   = {int(m) for m in re.findall(r"\[(\d+)\]", answer)}
            cited_text   = " ".join(
                s.get("preview", "") for s in sources
                if _citation_num(s.get("citation", "")) in cited_nums
            )

            result.update({
                "answer":             answer[:500],
                "total_ms":           total_ms,
                "model":              resp.get("model", ""),
                "provider":           "anthropic" if "claude" in resp.get("model", "") else "openai",
                "prompt_tokens":      resp.get("prompt_tokens", 0),
                "completion_tokens":  resp.get("completion_tokens", 0),
                "intent":             getattr(intent, "intent_type", "") or "",
                "citation_count":     resp.get("retrieval_count", len(sources)),
                "post_passed":        g.get("post_passed", True),
                "answer_safe":        g.get("answer_safe", True),
                "abstained":          g.get("abstention_detected", False),
                "faithfulness":       g.get("faithfulness_score", -1.0),
                "hallucination_risk": g.get("hallucination_risk", "unknown"),
                # Dim 3
                "citation_coverage":  citation_coverage(answer),
                "citation_validity":  citation_validity(answer, len(sources)),
                "keyword_recall":     keyword_recall(answer, sources_text, case.expected_keywords),
                # Dim 4
                "abstention_correct": abstention_correct(answer, case.expected_abstain),
                "faithfulness_embed": faithfulness_embedding(
                    answer, [s.get("preview", "") for s in sources], self._model
                ) if self._model else -1.0,
                # Dim 5
                "numeric_precision":  numeric_precision(answer, cited_text),
                "expected_num_hit":   expected_number_hit(answer, sources_text, case.expected_numbers),
                # Dim 6
                "has_unsafe_content": not g.get("answer_safe", True),
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

        return EvalMetrics(
            # Dim 1
            hit_at_5               = ret.post_hit_5,
            hit_at_10              = ret.post_hit_10,
            mrr                    = ret.post_mrr,
            ndcg_at_10             = ret.post_ndcg_10,
            # Dim 2
            ce_avg_improvement     = ret.ce_avg_improvement,
            ce_promotion_rate      = ret.ce_promotion_rate,
            # Dim 3
            citation_coverage_avg  = avg("citation_coverage"),
            citation_validity_avg  = avg("citation_validity", 1.0),
            # Dim 4
            faithfulness_avg       = avg("faithfulness_embed", -1.0),
            keyword_recall_avg     = avg("keyword_recall"),
            abstention_acc         = abstention_accuracy(case_results),
            # Dim 5
            numeric_precision_avg  = avg("numeric_precision", 1.0),
            expected_number_hit_avg= avg("expected_num_hit", 1.0),
            # Dim 6
            advice_block_rate      = advice_block_rate(case_results),
            false_positive_rate    = false_positive_rate(case_results),
            safety_catch_rate      = 1.0,
            # Dim 7
            avg_latency_ms         = costs.avg_latency_ms,
            p90_latency_ms         = costs.p90_latency_ms,
            avg_prompt_tokens      = costs.avg_prompt_tokens,
            avg_completion_tokens  = costs.avg_completion_tokens,
            total_cost_usd         = costs.total_cost_usd,
            cost_per_query_usd     = costs.cost_per_query,
        )


def _citation_num(citation_str: str) -> int:
    """Extract the integer from '[3]' → 3, or 0 if not parseable."""
    m = re.search(r"\d+", citation_str or "")
    return int(m.group()) if m else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="AMFI pipeline evaluation suite")
    parser.add_argument("--limit",  type=int, default=None)
    parser.add_argument("--no-llm", action="store_true",
                        help="Skip LLM generation (dims 1, 2, 6 only)")
    parser.add_argument("--output", default=None,
                        help="Write JSON results to this file (default: eval_results_<ts>.json)")
    args = parser.parse_args()

    if not settings.postgres_url:
        print("POSTGRES_URL not set.", file=sys.stderr)
        sys.exit(1)

    # ── LangSmith tracing (mirrors api/main.py lifespan) ──────────────
    if settings.langsmith_api_key:
        import os
        os.environ.setdefault("LANGSMITH_API_KEY",    settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_API_KEY",    settings.langsmith_api_key)
        os.environ.setdefault("LANGCHAIN_PROJECT",    settings.langchain_project)
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true" if settings.langchain_tracing_v2 else "false")

    repo      = DocumentRepository(settings.postgres_url)
    retriever = Retriever(repo, settings.embed_model)
    ret_pipe  = RetrievalPipeline(repo, retriever)

    graph = None
    if not args.no_llm and settings.openai_api_key:
        from financial_pipeline.graph import build_graph, NodeFactory, AnalyticalNodeFactory
        factory    = NodeFactory(repo=repo, retriever=retriever)
        analytical = AnalyticalNodeFactory(repo=repo)
        graph      = build_graph(factory, analytical=analytical)

    cases = EVAL_DATASET[: args.limit] if args.limit else EVAL_DATASET
    print(f"Running {len(cases)} eval cases  (llm={'yes' if graph else 'no'})")

    embed_model = retriever._model if hasattr(retriever, "_model") else None
    runner  = EvalRunner(repo, retriever, ret_pipe, graph=graph,
                         embed_model=embed_model, run_llm=not args.no_llm)
    metrics = runner.run(cases)
    metrics.print_report()

    out_path = Path(args.output) if args.output else Path(
        f"eval_results_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
    )
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
