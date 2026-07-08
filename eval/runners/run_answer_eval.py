"""Phase 4: Full pipeline answer quality evaluation."""
import json
import sys
from pathlib import Path

import structlog

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "eval/corpus/query_corpus.json"
ANSWER_LABELS = ROOT / "eval/corpus/expected_answers.json"

ANSWER_QUERY_IDS = {
    "Q001","Q002","Q003","Q004","Q005",
    "Q011","Q013","Q021","Q022",
    "Q041","Q042",
    "Q049","Q050","Q052","Q057","Q058","Q059",
}

_graph = None
log = structlog.get_logger()


def _get_graph():
    global _graph
    if _graph is None:
        from financial_pipeline.graph import build_graph, NodeFactory, AnalyticalNodeFactory
        from financial_pipeline.retrieval.retriever import Retriever
        from financial_pipeline.storage.document_repo import DocumentRepository
        from financial_pipeline.config import settings

        repo = DocumentRepository(settings.postgres_url)
        retriever = Retriever(repo, settings.embed_model)
        factory = NodeFactory(repo=repo, retriever=retriever)
        analytical = AnalyticalNodeFactory(repo=repo)
        try:
            from financial_pipeline.text_to_sql.vanna_agent import build_vanna_agent
            from financial_pipeline.graph.nodes_sql import SQLNodeFactory

            vn = build_vanna_agent(
                anthropic_api_key=settings.openai_api_key,
                postgres_url=settings.postgres_url,
            )
            sql_factory = SQLNodeFactory(vanna=vn)
            _graph = build_graph(factory, analytical=analytical, sql=sql_factory)
        except ModuleNotFoundError as exc:
            log.warning("answer.sql_unavailable", error=str(exc))
            _graph = build_graph(factory, analytical=analytical)
    return _graph


def _run_pipeline(query: str) -> dict:
    """Run the full LangGraph pipeline and return the final answer."""
    graph = _get_graph()
    final_state = graph.invoke({"query": query})
    citations = final_state.get("citations", [])
    return {
        "answer": final_state.get("response", {}).get("answer", ""),
        "contexts": [c.excerpt for c in citations if c.excerpt],
        "route": final_state.get("route"),
    }


def run(
    query_ids: list[str] | None = None,
    run_ragas: bool = False,
    ragas_metrics: list[str] | None = None,
) -> dict:
    corpus_raw = json.loads(CORPUS.read_text())
    corpus = corpus_raw["queries"] if isinstance(corpus_raw, dict) and "queries" in corpus_raw else corpus_raw
    answer_labels = json.loads(ANSWER_LABELS.read_text())

    target_ids = set(query_ids) if query_ids else ANSWER_QUERY_IDS

    questions = []
    answers = []
    contexts_list = []
    label_list = []
    raw_results = []
    errors = []

    for item in corpus:
        qid = item["id"]
        if qid not in target_ids:
            continue
        label = answer_labels.get(qid)
        if label is None:
            continue
        try:
            r = _run_pipeline(item["query"])
            r["id"] = qid
            questions.append(item["query"])
            answers.append(r.get("answer", ""))
            contexts_list.append(r.get("contexts", []))
            label_list.append(label)
            raw_results.append(r)
        except Exception as exc:
            errors.append({"id": qid, "error": str(exc)})

    from eval.metrics import answer_metrics
    heuristic_metrics = answer_metrics.summary(answers, label_list)

    result = {
        "phase": "answer",
        "n_queries": len(answers),
        "n_errors": len(errors),
        "metrics": heuristic_metrics,
        "errors": errors,
        "results": [
            {"id": r["id"], "answer": r["answer"], "route": r.get("route")}
            for r in raw_results
        ],
    }

    if run_ragas and answers:
        from eval.metrics.ragas_metrics import run_ragas as _run_ragas
        print("[RAGAS] Running LLM-based evaluation …")
        ragas_result = _run_ragas(
            questions=questions,
            answers=answers,
            contexts_list=contexts_list,
            metric_names=ragas_metrics,
        )
        result["ragas"] = ragas_result

    return result


if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="Run Phase 4: Answer quality eval")
    parser.add_argument("--ids", nargs="*", help="Subset of query IDs")
    parser.add_argument("--ragas", action="store_true", help="Run RAGAS LLM-based metrics")
    parser.add_argument(
        "--ragas-metrics",
        nargs="*",
        default=None,
        choices=["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
        help="Which RAGAS metrics to run (default: faithfulness + answer_relevancy)",
    )
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run(args.ids, run_ragas=args.ragas, ragas_metrics=args.ragas_metrics)
    output = _json.dumps(result, indent=2, default=str)
    print(output)
    if args.out:
        Path(args.out).write_text(output)
