"""Phase 5: Guardrail effectiveness evaluation using failure corpus."""
import json
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "eval/corpus/query_corpus.json"
FAILURE_LABELS = ROOT / "eval/corpus/failure_corpus.json"
ANSWER_LABELS = ROOT / "eval/corpus/expected_answers.json"

FAILURE_IDS = {f"F{i:03d}" for i in range(1, 32)}


_vanna_instance = None
log = structlog.get_logger()

def _get_vanna():
    global _vanna_instance
    if _vanna_instance is None:
        from financial_pipeline.text_to_sql.vanna_agent import build_vanna_agent
        from financial_pipeline.config import settings
        _vanna_instance = build_vanna_agent(
            anthropic_api_key=settings.openai_api_key,
            postgres_url=settings.postgres_url,
        )
    return _vanna_instance


def _build_graph_with_sql():
    """Mirrors api/main.py's graph construction (factory + analytical + sql).
    Without the sql factory, any guardrail case that production routes to
    Vanna/SQL silently degrades to a RAG-only answer here instead — that's
    the gap that made F009/F023 score low fact_coverage even after the
    has_identified_fund guardrail fix, since the block itself was gone but
    this harness had no way to give the same real answer production does.
    fund_performance/reasoning aren't wired here because api/main.py doesn't
    wire them into the production graph either — matching current reality,
    not the full graph module's capability.
    """
    from financial_pipeline.graph import build_graph, NodeFactory, AnalyticalNodeFactory
    from financial_pipeline.graph.nodes_sql import SQLNodeFactory
    from financial_pipeline.retrieval.retriever import Retriever
    from financial_pipeline.storage.document_repo import DocumentRepository
    from financial_pipeline.config import settings

    repo = DocumentRepository(settings.postgres_url)
    retriever = Retriever(repo, settings.embed_model)
    factory = NodeFactory(repo=repo, retriever=retriever)
    analytical = AnalyticalNodeFactory(repo=repo)

    sql_factory = None
    try:
        sql_factory = SQLNodeFactory(vanna=_get_vanna())
    except Exception as exc:
        log.warning("guardrail.sql_factory_unavailable", error=str(exc))

    return build_graph(factory, analytical=analytical, sql=sql_factory)


def _run_pipeline_with_guard_context(query: str, label: dict) -> dict:
    """Run pipeline and capture guardrail-relevant state."""
    import re

    routing = label.get("routing", "route")
    result = {
        "hard_blocked": False,
        "timed_out": False,
        "scope_rejected": False,
        "blocked_at_layer": None,
        "policy_warnings": [],
        "sql": None,
        "answer": None,
    }

    if routing == "query_sql":
        try:
            from financial_pipeline.text_to_sql import vanna_agent
            vn = _get_vanna()
            sql, df, answer, warnings = vanna_agent.ask(vn, query)
            result["sql"] = sql
            result["answer"] = answer
            result["policy_warnings"] = warnings or []
            # Detect hard block from answer string (policy blocks return None df + block message)
            if df is None and answer and "SQL blocked by safety policy" in answer:
                result["hard_blocked"] = True
                result["blocked_at_layer"] = "sql_validator"
            elif df is None and answer and "timeout" in answer.lower():
                result["timed_out"] = True
                result["blocked_at_layer"] = "sql_timeout"
        except ModuleNotFoundError as exc:
            log.warning("guardrail.sql_unavailable", error=str(exc))
            graph = _build_graph_with_sql()
            final_state = graph.invoke({"query": query})
            response = final_state.get("response", {})
            result["answer"] = response.get("answer", "")
            result["sql"] = final_state.get("generation_meta", {}).get("sql")
            result["sql_unavailable"] = True
    else:
        try:
            from financial_pipeline.augmentation.guardrails import PreGenerationGuardrails
            from financial_pipeline.retrieval.query_understanding import QueryAnalyzer

            # Mirrors graph/nodes.py's analyze_query: run intent extraction
            # first so has_identified_fund reflects the same signal
            # production uses (amc_names populated by stage1, no embed_fn/
            # generator needed) — otherwise this eval path tests a stricter,
            # unrealistic guardrail than what users actually hit.
            intent = QueryAnalyzer().analyze(query)
            pre = PreGenerationGuardrails().check(
                query, [], "factual", has_identified_fund=bool(intent.amc_names),
            )
            if not pre.should_proceed:
                result["answer"] = pre.block_reason
                result["blocked_at_layer"] = "pre_guardrail"
                if pre.is_out_of_scope:
                    result["scope_rejected"] = True
                else:
                    result["hard_blocked"] = True
                return result

            graph = _build_graph_with_sql()
            final_state = graph.invoke({"query": query})
            response = final_state.get("response", {})
            result["answer"] = response.get("answer", "")
            result["sql"] = final_state.get("generation_meta", {}).get("sql")
            guardrail = response.get("guardrail", {})
            if guardrail.get("blocked"):
                result["blocked_at_layer"] = "pre_guardrail"
                if guardrail.get("is_out_of_scope"):
                    result["scope_rejected"] = True
                else:
                    result["hard_blocked"] = True
        except Exception as e:
            result["error"] = str(e)

    return result


def run(query_ids: list[str] | None = None) -> dict:
    corpus_raw = json.loads(CORPUS.read_text())
    corpus = corpus_raw["queries"] if isinstance(corpus_raw, dict) and "queries" in corpus_raw else corpus_raw
    failure_labels = json.loads(FAILURE_LABELS.read_text())
    answer_labels = json.loads(ANSWER_LABELS.read_text())

    target_ids = set(query_ids) if query_ids else FAILURE_IDS

    results = []
    label_list = []
    raw_results = []
    errors = []

    for item in corpus:
        qid = item["id"]
        if qid not in target_ids:
            continue
        label = failure_labels.get(qid)
        if label is None:
            continue
        try:
            r = _run_pipeline_with_guard_context(item["query"], label)
            r["id"] = qid
            results.append(r)
            label_list.append(label)
            raw_results.append(r)
        except Exception as exc:
            errors.append({"id": qid, "error": str(exc)})

    from eval.metrics import guardrail_metrics, answer_metrics

    guardrail_m = guardrail_metrics.summary(results, label_list)

    answer_label_list = [answer_labels.get(r["id"]) for r in raw_results]
    answers = [r.get("answer", "") or "" for r in raw_results]
    answer_m = answer_metrics.summary(answers, answer_label_list)

    return {
        "phase": "guardrail",
        "n_queries": len(results),
        "n_errors": len(errors),
        "metrics": {**guardrail_m, "answer": answer_m},
        "errors": errors,
        "results": [
            {k: v for k, v in r.items() if k not in ("df",)}
            for r in raw_results
        ],
    }


if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="Run Phase 5: Guardrail eval")
    parser.add_argument("--ids", nargs="*", help="Subset of failure query IDs")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run(args.ids)
    output = _json.dumps(result, indent=2, default=str)
    print(output)
    if args.out:
        Path(args.out).write_text(output)
