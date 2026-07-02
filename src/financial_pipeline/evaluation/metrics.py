"""Metric calculation functions for all 7 evaluation dimensions.

All functions are pure (no IO, no model calls) so they can be unit-tested
and composed freely.

Dimension 1 — Retrieval Quality
    hit_at_k        : fraction of queries where ≥1 relevant doc in top-k
    mrr             : Mean Reciprocal Rank
    ndcg_at_k       : Normalized Discounted Cumulative Gain

Dimension 2 — Reranker Quality
    reranker_improvement : fraction of queries where CE moved a relevant doc higher

Dimension 3 — Citation Correctness
    citation_coverage   : fraction of cited claims with a valid [N] marker
    citation_validity   : fraction of [N] markers pointing to real sources

Dimension 4 — Answer Groundedness
    faithfulness_score  : embedding cosine sim between answer and cited chunks
    keyword_recall      : fraction of expected_keywords found in answer or sources

Dimension 5 — Numeric Accuracy
    numeric_precision   : fraction of numbers in answer that appear in cited chunks
    expected_number_hit : fraction of expected_numbers found in answer

Dimension 6 — Guardrail Effectiveness
    advice_block_rate   : fraction of investment-advice queries that were blocked
    false_positive_rate : fraction of valid queries that were incorrectly blocked
    safety_catch_rate   : fraction of unsafe answers caught by post-guardrail
    abstention_accuracy : model correctly abstains on OOD + advice queries

Dimension 7 — Latency & Cost  (computed by CostTracker / observability)
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field


# ── Retrieval Quality (Dim 1) ─────────────────────────────────────────────────

def hit_at_k(
    retrieved_files: list[str],
    relevant_files:  list[str],
    k:               int = 10,
) -> float:
    """1.0 if any relevant file appears in the top-k retrieved results."""
    if not relevant_files:
        return 1.0   # no ground truth → not penalised
    top_k = set(retrieved_files[:k])
    return float(any(f in top_k for f in relevant_files))


def reciprocal_rank(
    retrieved_files: list[str],
    relevant_files:  list[str],
) -> float:
    """1/rank of the first relevant document (0 if never retrieved)."""
    for rank, fname in enumerate(retrieved_files, 1):
        if fname in relevant_files:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved_files: list[str],
    relevant_files:  list[str],
    k:               int = 10,
) -> float:
    """NDCG@k with binary relevance (1 if relevant, 0 otherwise)."""
    rel_set = set(relevant_files)

    def dcg(files: list[str]) -> float:
        return sum(
            (1.0 if f in rel_set else 0.0) / math.log2(i + 2)
            for i, f in enumerate(files[:k])
        )

    actual_dcg  = dcg(retrieved_files)
    ideal_files = [f for f in retrieved_files if f in rel_set][:k]
    ideal_dcg   = dcg(ideal_files)
    return actual_dcg / ideal_dcg if ideal_dcg > 0 else (1.0 if not relevant_files else 0.0)


# ── Reranker Quality (Dim 2) ──────────────────────────────────────────────────

def reranker_rank_improvement(
    pre_ce_files:  list[str],
    post_ce_files: list[str],
    relevant_files:list[str],
) -> float:
    """Average rank improvement for relevant docs after cross-encoder reranking.

    Positive = reranker moved relevant docs higher (good).
    Negative = reranker moved them lower (bad).
    0.0      = no change or no relevant docs.
    """
    if not relevant_files:
        return 0.0

    pre_ranks  = {f: i for i, f in enumerate(pre_ce_files)}
    post_ranks = {f: i for i, f in enumerate(post_ce_files)}
    improvements = []

    for f in relevant_files:
        pre  = pre_ranks.get(f)
        post = post_ranks.get(f)
        if pre is not None and post is not None:
            improvements.append(pre - post)   # positive = moved up

    return sum(improvements) / len(improvements) if improvements else 0.0


def reranker_promoted(
    pre_ce_files:  list[str],
    post_ce_files: list[str],
    relevant_files:list[str],
    top_k:         int = 5,
) -> bool:
    """True if cross-encoder moved a relevant doc into top-k when it wasn't before."""
    rel_set    = set(relevant_files)
    pre_top_k  = set(pre_ce_files[:top_k])
    post_top_k = set(post_ce_files[:top_k])
    new_in_top = (post_top_k - pre_top_k) & rel_set
    return bool(new_in_top)


# ── Citation Correctness (Dim 3) ──────────────────────────────────────────────

def citation_coverage(answer: str) -> float:
    """Fraction of answer sentences that contain at least one [N] citation."""
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 8]
    if not sentences:
        return 0.0
    cited = sum(1 for s in sentences if re.search(r'\[\d+\]', s))
    return cited / len(sentences)


def citation_validity(answer: str, num_citations: int) -> float:
    """Fraction of [N] markers in the answer that point to valid citation numbers."""
    markers = [int(m) for m in re.findall(r'\[(\d+)\]', answer)]
    if not markers:
        return 1.0   # no citations to validate
    valid = sum(1 for m in markers if 1 <= m <= num_citations)
    return valid / len(markers)


def keyword_recall(
    answer:            str,
    sources_text:      str,   # concatenated chunk texts
    expected_keywords: list[str],
) -> float:
    """Fraction of expected_keywords found in the answer or retrieved sources."""
    if not expected_keywords:
        return 1.0
    combined = (answer + " " + sources_text).lower()
    found    = sum(1 for kw in expected_keywords if kw.lower() in combined)
    return found / len(expected_keywords)


# ── Answer Groundedness (Dim 4) ───────────────────────────────────────────────

def faithfulness_embedding(
    answer:   str,
    contexts: list[str],
    model,                 # sentence-transformers model
) -> float:
    """Average cosine similarity between the answer and its cited contexts."""
    if not contexts or not answer.strip():
        return -1.0
    try:
        import numpy as np
        texts = [answer[:500]] + [c[:500] for c in contexts]
        vecs  = model.encode(texts, normalize_embeddings=True)
        sims  = [float(np.dot(vecs[0], vecs[i])) for i in range(1, len(vecs))]
        return float(sum(sims) / len(sims))
    except Exception:
        return -1.0


def abstention_correct(
    answer:          str,
    expected_abstain:bool,
) -> bool:
    """True if abstention behaviour matched expectation."""
    abstained = "not available in the provided amfi documents" in answer.lower()
    return abstained == expected_abstain


# ── Numeric Accuracy (Dim 5) ──────────────────────────────────────────────────

def numeric_precision(answer: str, cited_text: str) -> float:
    """Fraction of multi-digit numbers in the answer that appear in cited sources."""
    nums = set(
        n for n in re.findall(r'\b\d[\d,\.]*\d\b', answer)
        if len(n.replace(',', '').replace('.', '')) >= 3
    )
    if not nums:
        return 1.0   # no numbers → nothing to check
    supported = sum(1 for n in nums if n in cited_text)
    return supported / len(nums)


def expected_number_hit(answer: str, sources_text: str, expected_numbers: list[str]) -> float:
    """Fraction of expected_numbers found in the answer or cited sources."""
    if not expected_numbers:
        return 1.0
    combined = (answer + " " + sources_text).lower()
    found    = sum(1 for n in expected_numbers if n.lower() in combined)
    return found / len(expected_numbers)


# ── Guardrail Effectiveness (Dim 6) ──────────────────────────────────────────

def advice_block_rate(results: list[dict]) -> float:
    """Fraction of investment-advice queries correctly blocked by pre-guardrail."""
    advice = [r for r in results if r.get("is_investment_advice")]
    if not advice:
        return 1.0
    blocked = sum(1 for r in advice if r.get("pre_blocked"))
    return blocked / len(advice)


def false_positive_rate(results: list[dict]) -> float:
    """Fraction of legitimate queries incorrectly blocked by pre-guardrail."""
    legit = [r for r in results if not r.get("is_investment_advice")]
    if not legit:
        return 0.0
    fp = sum(1 for r in legit if r.get("pre_blocked"))
    return fp / len(legit)


def abstention_accuracy(results: list[dict]) -> float:
    """Fraction of queries where abstention was correct (both OOD + advice)."""
    if not results:
        return 0.0
    correct = sum(1 for r in results if r.get("abstention_correct"))
    return correct / len(results)


def safety_catch_rate(results: list[dict]) -> float:
    """Fraction of queries with unsafe answers that post-guardrail caught."""
    unsafe = [r for r in results if r.get("has_unsafe_content")]
    if not unsafe:
        return 1.0
    caught = sum(1 for r in unsafe if not r.get("post_answer_safe"))
    return caught / len(unsafe)


# ── Aggregated score ─────────────────────────────────────────────────────────

@dataclass
class EvalMetrics:
    """Aggregated metrics across all eval cases — one per eval run."""

    # Dim 1: Retrieval
    hit_at_5:       float = 0.0
    hit_at_10:      float = 0.0
    mrr:            float = 0.0
    ndcg_at_10:     float = 0.0

    # Dim 2: Reranker
    ce_avg_improvement: float = 0.0
    ce_promotion_rate:  float = 0.0

    # Dim 3: Citations
    citation_coverage_avg: float = 0.0
    citation_validity_avg: float = 0.0

    # Dim 4: Groundedness
    faithfulness_avg:   float = -1.0
    keyword_recall_avg: float = 0.0
    abstention_acc:     float = 0.0

    # Dim 5: Numeric
    numeric_precision_avg:   float = 0.0
    expected_number_hit_avg: float = 0.0

    # Dim 6: Guardrails
    advice_block_rate:   float = 0.0
    false_positive_rate: float = 0.0
    safety_catch_rate:   float = 0.0

    # Dim 7: Latency & cost
    avg_latency_ms:        float = 0.0
    p90_latency_ms:        float = 0.0
    avg_prompt_tokens:     float = 0.0
    avg_completion_tokens: float = 0.0
    total_cost_usd:        float = 0.0
    cost_per_query_usd:    float = 0.0

    # Meta
    total_cases:     int   = 0
    failed_cases:    int   = 0
    warnings:        list[str] = field(default_factory=list)

    def overall_score(self) -> float:
        """Weighted composite score (0–1) across all non-latency dimensions."""
        scores = []
        weights = []

        def add(val: float, w: float) -> None:
            if val >= 0:
                scores.append(max(0.0, min(1.0, val)))
                weights.append(w)

        add(self.hit_at_5,              w=2.0)
        add(self.mrr,                   w=1.5)
        add(self.citation_validity_avg, w=2.0)
        add(self.keyword_recall_avg,    w=1.5)
        add(self.abstention_acc,        w=2.0)
        add(self.numeric_precision_avg, w=1.5)
        add(self.advice_block_rate,     w=3.0)   # safety is high priority
        add(1.0 - self.false_positive_rate, w=1.0)

        if not scores:
            return 0.0
        return sum(s * w for s, w in zip(scores, weights)) / sum(weights)

    def print_report(self) -> None:
        score = self.overall_score()
        grade = "A" if score >= 0.85 else "B" if score >= 0.70 else "C" if score >= 0.55 else "D"

        print(f"\n{'═'*62}")
        print(f"  AMFI Pipeline — Evaluation Report")
        print(f"  Overall Score: {score:.1%}  ({grade})")
        print(f"{'═'*62}")
        print(f"\n  Dim 1 — Retrieval Quality")
        print(f"    Hit@5          : {self.hit_at_5:.1%}")
        print(f"    Hit@10         : {self.hit_at_10:.1%}")
        print(f"    MRR            : {self.mrr:.3f}")
        print(f"    NDCG@10        : {self.ndcg_at_10:.3f}")
        print(f"\n  Dim 2 — Reranker Quality")
        print(f"    Avg rank gain  : {self.ce_avg_improvement:+.2f}  (positive = improved)")
        print(f"    Promotion rate : {self.ce_promotion_rate:.1%}")
        print(f"\n  Dim 3 — Citation Correctness")
        print(f"    Coverage       : {self.citation_coverage_avg:.1%}")
        print(f"    Validity       : {self.citation_validity_avg:.1%}")
        print(f"\n  Dim 4 — Answer Groundedness")
        print(f"    Faithfulness   : {self.faithfulness_avg:.3f}" if self.faithfulness_avg >= 0 else "    Faithfulness   : n/a (no embed model)")
        print(f"    Keyword recall : {self.keyword_recall_avg:.1%}")
        print(f"    Abstention acc : {self.abstention_acc:.1%}")
        print(f"\n  Dim 5 — Numeric Accuracy")
        print(f"    Precision      : {self.numeric_precision_avg:.1%}")
        print(f"    Expected hits  : {self.expected_number_hit_avg:.1%}")
        print(f"\n  Dim 6 — Guardrail Effectiveness")
        print(f"    Advice blocked : {self.advice_block_rate:.1%}  (goal: 100%)")
        print(f"    False positive : {self.false_positive_rate:.1%}  (goal: 0%)")
        print(f"    Safety catch   : {self.safety_catch_rate:.1%}")
        print(f"\n  Dim 7 — Latency & Cost")
        print(f"    Avg latency    : {self.avg_latency_ms:.0f} ms")
        print(f"    p90 latency    : {self.p90_latency_ms:.0f} ms")
        print(f"    Avg tokens     : {self.avg_prompt_tokens:.0f}in / {self.avg_completion_tokens:.0f}out")
        print(f"    Total cost     : ${self.total_cost_usd:.4f}")
        print(f"    Cost / query   : ${self.cost_per_query_usd:.4f}")
        print(f"\n  Cases: {self.total_cases}  |  Failed: {self.failed_cases}")
        if self.warnings:
            print(f"\n  Warnings:")
            for w in self.warnings[:5]:
                print(f"    ⚠  {w}")
        print(f"{'═'*62}\n")
