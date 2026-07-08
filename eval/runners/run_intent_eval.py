"""Phase 1: QueryAnalyzer intent & routing evaluation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "eval/corpus/query_corpus.json"
LABELS = ROOT / "eval/corpus/intent_labels.json"


def _predict_routing(intent) -> str:
    """Mirror is_range_query()'s precedence (edges_analytical.py) so the eval
    reflects actual graph routing, not just intent_type."""
    if "investment_advice" in getattr(intent, "labels", []):
        return "route"
    if getattr(intent, "needs_analytical", False):
        return "plan_years"
    if getattr(intent, "intent_type", None) in ("tabular", "trend"):
        return "query_sql"
    if getattr(intent, "year_from", None) and getattr(intent, "year_to", None):
        return "plan_years"
    return "route"


def run(query_ids: list[str] | None = None) -> dict:
    corpus_raw = json.loads(CORPUS.read_text())
    corpus = corpus_raw["queries"] if isinstance(corpus_raw, dict) and "queries" in corpus_raw else corpus_raw
    labels = json.loads(LABELS.read_text())

    from financial_pipeline.retrieval.query_understanding import QueryAnalyzer
    analyzer = QueryAnalyzer()

    predictions = []
    label_list  = []
    errors      = []

    for item in corpus:
        qid = item["id"]
        if query_ids and qid not in query_ids:
            continue
        if qid not in labels:
            continue
        label = labels[qid]
        try:
            intent = analyzer.analyze(item["query"])
            predictions.append({
                "id":              qid,
                "intent_type":     intent.intent_type,
                "routing":         _predict_routing(intent),
                "year":            intent.year,
                "year_from":       intent.year_from,
                "year_to":         intent.year_to,
                "month":           intent.month,
                "needs_analytical": intent.needs_analytical,
                "metric":          intent.metric,
                "scheme_types":    intent.scheme_types,
                "confidence":      intent.confidence,
            })
            label_list.append({**label, "id": qid})
        except Exception as exc:
            errors.append({"id": qid, "error": str(exc)})

    from eval.metrics import intent_metrics
    metrics = intent_metrics.summary(predictions, label_list)
    return {
        "phase":      "intent",
        "n_queries":  len(predictions),
        "n_errors":   len(errors),
        "metrics":    metrics,
        "errors":     errors,
        "predictions": predictions,
    }


if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="Run Phase 1: Intent eval")
    parser.add_argument("--ids", nargs="*", help="Subset of query IDs to evaluate")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run(args.ids)
    output = _json.dumps(result, indent=2)
    print(output)
    if args.out:
        Path(args.out).write_text(output)
