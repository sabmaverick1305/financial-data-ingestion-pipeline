"""Phase 3: Retrieval Quality evaluation metrics."""
from __future__ import annotations
from typing import Any


def _citations_hit(citations: list[dict], label: dict) -> list[bool]:
    """Return per-citation boolean — True if citation matches any expected label."""
    expected_years = set(label.get("expected_doc_years") or [])
    expected_months = set(label.get("expected_doc_months") or [])
    keywords = [kw.lower() for kw in (label.get("expected_keywords") or [])]

    hits = []
    for c in citations:
        year_ok = not expected_years or c.get("period_year") in expected_years
        month_ok = not expected_months or c.get("period_month") in expected_months
        excerpt = (c.get("excerpt") or "").lower()
        kw_ok = not keywords or any(kw in excerpt for kw in keywords)
        hits.append(year_ok and month_ok and kw_ok)
    return hits


def hit_at_k(results: list[dict], labels: list[dict], k: int = 8) -> float:
    """Hit@K — fraction of queries where at least 1 of top-k citations is relevant."""
    hits = 0
    total = 0
    for r, l in zip(results, labels):
        if l is None:
            continue
        total += 1
        citations = r.get("citations", [])[:k]
        hit_flags = _citations_hit(citations, l)
        if any(hit_flags):
            hits += 1
    return hits / total if total else 0.0


def precision_at_k(results: list[dict], labels: list[dict], k: int = 8) -> float:
    """Precision@K — average fraction of top-k citations that are relevant."""
    scores = []
    for r, l in zip(results, labels):
        if l is None:
            continue
        citations = r.get("citations", [])[:k]
        if not citations:
            scores.append(0.0)
            continue
        hit_flags = _citations_hit(citations, l)
        scores.append(sum(hit_flags) / len(citations))
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(results: list[dict], labels: list[dict], k: int = 8) -> float:
    """Recall@K — fraction of required relevant citations found in top-k."""
    scores = []
    for r, l in zip(results, labels):
        if l is None:
            continue
        min_relevant = l.get("min_relevant_in_top_k", 1)
        citations = r.get("citations", [])[:k]
        hit_flags = _citations_hit(citations, l)
        found = sum(hit_flags)
        scores.append(min(found / min_relevant, 1.0))
    return sum(scores) / len(scores) if scores else 0.0


def mrr(results: list[dict], labels: list[dict]) -> float:
    """Mean Reciprocal Rank — 1/rank of first relevant citation."""
    scores = []
    for r, l in zip(results, labels):
        if l is None:
            continue
        citations = r.get("citations", [])
        hit_flags = _citations_hit(citations, l)
        rr = 0.0
        for rank, hit in enumerate(hit_flags, start=1):
            if hit:
                rr = 1.0 / rank
                break
        scores.append(rr)
    return sum(scores) / len(scores) if scores else 0.0


def keyword_coverage(results: list[dict], labels: list[dict], k: int = 8) -> float:
    """Average fraction of expected keywords found across top-k citation excerpts."""
    scores = []
    for r, l in zip(results, labels):
        if l is None:
            continue
        keywords = [kw.lower() for kw in (l.get("expected_keywords") or [])]
        if not keywords:
            continue
        citations = r.get("citations", [])[:k]
        combined_text = " ".join((c.get("excerpt") or "").lower() for c in citations)
        found = sum(1 for kw in keywords if kw in combined_text)
        scores.append(found / len(keywords))
    return sum(scores) / len(scores) if scores else 0.0


def summary(results: list[dict], labels: list[dict], k: int = 8) -> dict[str, Any]:
    return {
        f"hit@{k}": hit_at_k(results, labels, k),
        f"precision@{k}": precision_at_k(results, labels, k),
        f"recall@{k}": recall_at_k(results, labels, k),
        "mrr": mrr(results, labels),
        "keyword_coverage": keyword_coverage(results, labels, k),
    }
