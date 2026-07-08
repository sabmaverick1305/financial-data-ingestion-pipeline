"""Seed expected_results.json with exact DB values for SQL-path queries.

Runs each SQL-path query against the live DB and records actual values.
Only populates entries where seeded=false. Safe to re-run.

Usage:
  python scripts/seed_expected_results.py
  python scripts/seed_expected_results.py --ids Q001 Q002
  python scripts/seed_expected_results.py --force   # re-seed all, even if seeded=true
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

EXPECTED_PATH = ROOT / "eval/corpus/expected_results.json"
CORPUS_PATH = ROOT / "eval/corpus/query_corpus.json"

SQL_QUERY_IDS = {
    "Q001","Q002","Q003","Q004","Q005","Q006","Q007","Q008","Q009","Q010",
    "Q011","Q012","Q013","Q014","Q015","Q016","Q017","Q018","Q019","Q020",
    "Q021","Q022","Q023","Q024","Q025","Q026","Q027","Q028","Q029",
    "Q034","Q035","Q036","Q037","Q038","Q039","Q040",
    # Q061-Q097: expanded tabular/query_sql queries
    "Q061","Q062","Q063","Q064","Q065","Q066","Q067","Q068","Q069","Q070",
    "Q071","Q072","Q073","Q074","Q075","Q076","Q077","Q078","Q079","Q080",
    "Q081","Q082","Q083","Q084","Q085","Q086","Q087",
    "Q088","Q089","Q090","Q091","Q092",
    "Q093","Q094","Q095","Q096","Q097",
}


_vn = None

def _get_vn():
    global _vn
    if _vn is None:
        from financial_pipeline.config import settings
        from financial_pipeline.text_to_sql.vanna_agent import build_vanna_agent
        _vn = build_vanna_agent(
            anthropic_api_key=settings.openai_api_key,
            postgres_url=settings.postgres_url,
        )
    return _vn


def _run_query(query: str) -> tuple[str | None, object | None]:
    from financial_pipeline.text_to_sql.vanna_agent import ask
    try:
        sql, df, _, _ = ask(_get_vn(), query)
        return sql, df
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, None


def _extract_key_values(df) -> list[dict]:
    """Extract actual numeric values from the first row of a DataFrame.

    PostgreSQL NUMERIC columns arrive as Python Decimal (object dtype in pandas),
    so we try every column and cast rather than using select_dtypes.
    """
    if df is None or len(df) == 0:
        return []

    key_values = []
    for col in list(df.columns)[:6]:
        raw = df[col].iloc[0]
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        # Use min/max so bounds are always (lo, hi) regardless of sign
        lo = round(min(val * 0.98, val * 1.02), 2)
        hi = round(max(val * 0.98, val * 1.02), 2)
        key_values.append({
            "col": col,
            "min": lo,
            "max": hi,
            "actual": round(val, 2),
        })
        if len(key_values) == 4:
            break
    return key_values


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed expected_results.json from live DB")
    parser.add_argument("--ids", nargs="*", help="Specific query IDs to seed")
    parser.add_argument("--force", action="store_true", help="Re-seed even if already seeded")
    args = parser.parse_args()

    expected = json.loads(EXPECTED_PATH.read_text())
    corpus_raw = json.loads(CORPUS_PATH.read_text())
    corpus = corpus_raw["queries"] if isinstance(corpus_raw, dict) and "queries" in corpus_raw else corpus_raw
    corpus_map = {item["id"]: item for item in corpus}

    target_ids = set(args.ids) if args.ids else SQL_QUERY_IDS

    seeded_count = 0
    skipped_count = 0
    error_count = 0

    for qid in sorted(target_ids):
        if qid not in expected:
            print(f"[{qid}] Not in expected_results.json — skipping")
            continue
        if expected[qid].get("seeded") and not args.force:
            skipped_count += 1
            continue
        if qid not in corpus_map:
            print(f"[{qid}] Not in query corpus — skipping")
            continue

        query = corpus_map[qid]["query"]
        print(f"[{qid}] Querying: {query[:80]}…")

        sql, df = _run_query(query)
        if df is None:
            print(f"  → No result")
            error_count += 1
            continue

        row_count = len(df)
        key_values = _extract_key_values(df)

        expected[qid]["row_count_actual"] = row_count
        expected[qid]["key_values"] = key_values
        expected[qid]["seeded"] = True
        expected[qid]["sql_used"] = sql

        if "row_count" not in expected[qid] and "row_count_min" not in expected[qid]:
            expected[qid]["row_count"] = row_count

        print(f"  → {row_count} rows, key values: {[kv['col'] for kv in key_values]}")
        seeded_count += 1

    EXPECTED_PATH.write_text(json.dumps(expected, indent=2))
    print(f"\nDone. Seeded: {seeded_count}  |  Skipped (already seeded): {skipped_count}  |  Errors: {error_count}")
    print(f"Updated: {EXPECTED_PATH}")


if __name__ == "__main__":
    main()
