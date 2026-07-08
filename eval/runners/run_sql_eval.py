"""Phase 2: Vanna SQL generation & policy evaluation."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "eval/corpus/query_corpus.json"
SQL_LABELS = ROOT / "eval/corpus/sql_labels.json"
EXPECTED = ROOT / "eval/corpus/expected_results.json"

SQL_QUERY_IDS = {
    "Q001","Q002","Q003","Q004","Q005","Q006","Q007","Q008","Q009","Q010",
    "Q011","Q012","Q013","Q014","Q015","Q016","Q017","Q018","Q019","Q020",
    "Q021","Q022","Q023","Q024","Q025","Q026","Q027","Q028","Q029",
    "Q034","Q035","Q036","Q037","Q038","Q039","Q040",
}

FAILURE_IDS = {
    "F001","F002","F003","F004","F005","F006","F009","F010","F011","F012",
    "F013","F014","F016","F018",
}


_vanna_instance = None

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


def _run_vanna(query: str) -> dict:
    from financial_pipeline.text_to_sql import vanna_agent
    vn = _get_vanna()
    sql, df, answer, policy_warnings = vanna_agent.ask(vn, query)
    hard_blocked = df is None and answer and "SQL blocked by safety policy" in answer
    timed_out = df is None and answer and "timeout" in (answer or "").lower()
    return {
        "sql": sql,
        "df": df,
        "answer": answer,
        "policy_warnings": policy_warnings or [],
        "hard_blocked": bool(hard_blocked),
        "timed_out": bool(timed_out),
    }


def run(query_ids: list[str] | None = None, include_failures: bool = True) -> dict:
    corpus_raw = json.loads(CORPUS.read_text())
    corpus = corpus_raw["queries"] if isinstance(corpus_raw, dict) and "queries" in corpus_raw else corpus_raw
    sql_labels = json.loads(SQL_LABELS.read_text())
    expected = json.loads(EXPECTED.read_text())

    target_ids = SQL_QUERY_IDS | (FAILURE_IDS if include_failures else set())
    if query_ids:
        target_ids = set(query_ids) & target_ids

    results = []
    label_list = []
    expected_list = []
    errors = []

    for item in corpus:
        qid = item["id"]
        if qid not in target_ids:
            continue
        try:
            r = _run_vanna(item["query"])
            r["id"] = qid
            results.append(r)
            label_list.append(sql_labels.get(qid))
            expected_list.append(expected.get(qid))
        except Exception as exc:
            errors.append({"id": qid, "error": str(exc)})

    from eval.metrics import sql_metrics
    valid_labels   = [l for l in label_list if l is not None]
    valid_results  = [r for r, l in zip(results, label_list) if l is not None]
    valid_expected = [e for e, l in zip(expected_list, label_list) if l is not None]
    metrics = sql_metrics.summary(valid_results, valid_labels, valid_expected)

    return {
        "phase": "sql",
        "n_queries": len(results),
        "n_errors": len(errors),
        "metrics": metrics,
        "errors": errors,
        "results": [
            {k: v for k, v in r.items() if k != "df"} for r in results
        ],
    }


if __name__ == "__main__":
    import argparse, json as _json

    parser = argparse.ArgumentParser(description="Run Phase 2: SQL eval")
    parser.add_argument("--ids", nargs="*", help="Subset of query IDs")
    parser.add_argument("--no-failures", action="store_true", help="Skip failure corpus")
    parser.add_argument("--out", default=None, help="Output JSON path")
    args = parser.parse_args()

    result = run(args.ids, include_failures=not args.no_failures)
    output = _json.dumps(result, indent=2, default=str)
    print(output)
    if args.out:
        Path(args.out).write_text(output)
