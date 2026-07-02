#!/usr/bin/env python3
"""Hybrid semantic + keyword search CLI for the AMFI document knowledge base.

Two search modes:
  semantic  — dense vector search via pgvector cosine similarity (finds conceptually
              similar content even without exact keyword matches)
  keyword   — BM25-style full-text search via PostgreSQL tsvector / GIN index
  hybrid    — both modes, results merged with Reciprocal Rank Fusion (RRF)

Examples:
    # Semantic: "What happened to equity fund AUMs in 2023?"
    python scripts/search.py "equity fund AUM growth" --mode semantic --limit 5

    # Keyword: exact terms like fund names or scheme codes
    python scripts/search.py "SBI Bluechip" --mode keyword --limit 5

    # Hybrid (default): best of both worlds
    python scripts/search.py "balanced advantage fund returns" --limit 8

    # Filter by time period
    python scripts/search.py "total AUM" --year 2024 --month 6

    # Filter by document category
    python scripts/search.py "scheme count" --category monthly
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_pipeline.config import settings  # noqa: E402
from financial_pipeline.storage.document_repo import DocumentRepository  # noqa: E402

MODEL_NAME = "all-MiniLM-L6-v2"

_model = None

def get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def rrf_merge(
    semantic_results: list[dict],
    keyword_results:  list[dict],
    k: int = 60,
) -> list[dict]:
    """Reciprocal Rank Fusion — combines two ranked lists without score normalisation.

    RRF score = sum(1 / (k + rank)) across both lists.
    k=60 is the standard constant (Cormack et al., 2009).
    """
    scores: dict[str, float] = {}
    meta:   dict[str, dict]  = {}

    for rank, result in enumerate(semantic_results, start=1):
        cid = str(result["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        meta[cid]   = result

    for rank, result in enumerate(keyword_results, start=1):
        cid = str(result["chunk_id"])
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
        if cid not in meta:
            meta[cid] = result

    merged = sorted(scores.keys(), key=lambda c: scores[c], reverse=True)
    return [meta[c] | {"rrf_score": round(scores[c], 5)} for c in merged]


def format_result(i: int, result: dict, mode: str) -> str:
    """Pretty-print a single search result."""
    lines = [
        f"\n{'─'*70}",
        f"  #{i+1}  {result.get('file_name', 'unknown')}",
    ]

    year  = result.get("period_year")
    month = result.get("period_month")
    cat   = result.get("category", "")
    if year:
        lines.append(f"       Period : {year}/{month:02d}" if month else f"       Period : {year}")
    if cat:
        lines.append(f"       Type   : {cat}")

    if mode == "semantic" and "similarity" in result:
        lines.append(f"       Score  : {result['similarity']:.4f} cosine similarity")
    elif mode == "keyword" and "rank" in result:
        lines.append(f"       Score  : {result['rank']:.4f} BM25 rank")
    elif "rrf_score" in result:
        lines.append(f"       Score  : {result['rrf_score']:.5f} RRF")

    text = result.get("text", "").strip().replace("\n", " ")
    preview = text[:300] + ("…" if len(text) > 300 else "")
    lines.append(f"\n  {preview}")
    return "\n".join(lines)


def search(
    query:    str,
    mode:     str = "hybrid",
    limit:    int = 10,
    year:     int | None = None,
    month:    int | None = None,
    category: str | None = None,
    min_sim:  float = 0.0,
) -> list[dict]:
    if not settings.postgres_url:
        print("POSTGRES_URL not set.", file=sys.stderr)
        sys.exit(1)

    repo = DocumentRepository(settings.postgres_url)

    semantic_results: list[dict] = []
    keyword_results:  list[dict] = []

    if mode in ("semantic", "hybrid"):
        model     = get_model()
        embedding = model.encode(query, normalize_embeddings=True).tolist()
        semantic_results = repo.search_similar(
            query_embedding=embedding,
            limit=limit * 2 if mode == "hybrid" else limit,
            period_year=year,
            period_month=month,
            category=category,
            min_similarity=min_sim,
        )

    if mode in ("keyword", "hybrid"):
        keyword_results = repo.search_fulltext(
            query=query,
            limit=limit * 2 if mode == "hybrid" else limit,
            period_year=year,
            category=category,
        )

    if mode == "hybrid":
        return rrf_merge(semantic_results, keyword_results)[:limit]
    elif mode == "semantic":
        return semantic_results[:limit]
    else:
        return keyword_results[:limit]


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("query",               help="Search query string")
    parser.add_argument("--mode",   default="hybrid",
                        choices=["semantic", "keyword", "hybrid"],
                        help="Search mode (default: hybrid)")
    parser.add_argument("--limit",  type=int, default=5,
                        help="Number of results to return (default: 5)")
    parser.add_argument("--year",   type=int, default=None,
                        help="Filter by period year (e.g. 2024)")
    parser.add_argument("--month",  type=int, default=None,
                        help="Filter by period month (e.g. 6)")
    parser.add_argument("--category", default=None,
                        choices=["monthly", "quarterly", "unknown"],
                        help="Filter by document category")
    parser.add_argument("--min-sim", type=float, default=0.0,
                        help="Minimum cosine similarity threshold (semantic mode)")
    args = parser.parse_args()

    print(f'\nSearching: "{args.query}"  [mode={args.mode}  limit={args.limit}]')
    if args.year or args.month or args.category:
        print(f"Filters: year={args.year}  month={args.month}  category={args.category}")

    results = search(
        query    = args.query,
        mode     = args.mode,
        limit    = args.limit,
        year     = args.year,
        month    = args.month,
        category = args.category,
        min_sim  = args.min_sim,
    )

    if not results:
        print("\nNo results found.")
        return

    print(f"\n{len(results)} result(s):")
    for i, result in enumerate(results):
        print(format_result(i, result, args.mode))

    print(f"\n{'─'*70}")


if __name__ == "__main__":
    main()
