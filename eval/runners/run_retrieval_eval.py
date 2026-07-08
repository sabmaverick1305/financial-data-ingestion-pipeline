"""Phase 3: Retrieval quality evaluation for RAG-path queries."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "eval/corpus/query_corpus.json"
RETRIEVAL_LABELS = ROOT / "eval/corpus/retrieval_labels.json"

RAG_QUERY_IDS = {f"Q0{i:02d}" for i in range(49, 61)}

_factory = None


def _get_factory():
    global _factory
    if _factory is not None:
        return _factory

    from financial_pipeline.config import settings
    from financial_pipeline.graph.nodes import NodeFactory
    from financial_pipeline.retrieval.retriever import Retriever
    from financial_pipeline.storage.document_repo import DocumentRepository

    repo      = DocumentRepository(settings.postgres_url)
    retriever = Retriever(repo, settings.embed_model)
    _factory  = NodeFactory(repo=repo, retriever=retriever)
    return _factory


def _run_retrieval(query: str) -> dict:
    from financial_pipeline.retrieval.query_understanding import QueryAnalyzer
    from financial_pipeline.graph.state import RAGState

    factory = _get_factory()
    analyzer = QueryAnalyzer()
    intent   = analyzer.analyze(query)

    # Analyze then route to pick the right retrieval branches
    state: RAGState = {"query": query, "retry_count": 0, "repair_count": 0, "intent": intent}
    route_result = factory.route(state)
    state.update(route_result)

    # Run parallel retrieval branches
    fused: list[dict] = []
    for branch in state.get("active_branches", ["dense", "sparse"]):
        try:
            fn = getattr(factory, f"retrieve_{branch}", None)
            if fn:
                r = fn(state)
                fused.extend(r.get(f"{branch}_results", []))
        except Exception:
            pass

    state["fused_results"] = fused
    rr = factory.rerank(state)
    state.update(rr)
    co = factory.context_optimizer(state)
    state.update(co)
    aug = factory.augment(state)

    citations = aug.get("citations", [])
    return {
        "citations": [
            {
                "period_year":  getattr(c, "period_year", None),
                "period_month": getattr(c, "period_month", None),
                "source_type":  getattr(c, "source_type", None),
                "excerpt":      getattr(c, "excerpt", ""),
                "confidence":   getattr(c, "confidence", 0.0),
            }
            for c in citations
        ]
    }


def run(query_ids: list[str] | None = None, k: int = 8) -> dict:
    corpus_raw = json.loads(CORPUS.read_text())
    corpus     = corpus_raw["queries"] if isinstance(corpus_raw, dict) and "queries" in corpus_raw else corpus_raw
    retrieval_labels = json.loads(RETRIEVAL_LABELS.read_text())

    target_ids = set(query_ids) if query_ids else RAG_QUERY_IDS

    results    = []
    label_list = []
    errors     = []

    for item in corpus:
        qid = item["id"]
        if qid not in target_ids:
            continue
        label = retrieval_labels.get(qid)
        if label is None:
            continue
        try:
            r = _run_retrieval(item["query"])
            r["id"] = qid
            results.append(r)
            label_list.append(label)
        except Exception as exc:
            errors.append({"id": qid, "error": str(exc)})

    from eval.metrics import retrieval_metrics
    metrics = retrieval_metrics.summary(results, label_list, k=k)

    return {
        "phase":     "retrieval",
        "n_queries": len(results),
        "n_errors":  len(errors),
        "k":         k,
        "metrics":   metrics,
        "errors":    errors,
        "results":   results,
    }


if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="Run Phase 3: Retrieval eval")
    parser.add_argument("--ids", nargs="*", help="Subset of query IDs")
    parser.add_argument("--k", type=int, default=8, help="Top-k cutoff for metrics")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run(args.ids, k=args.k)
    output = _json.dumps(result, indent=2, default=str)
    print(output)
    if args.out:
        Path(args.out).write_text(output)
