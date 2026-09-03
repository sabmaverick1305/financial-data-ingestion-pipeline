"""Automated triage — Stage 4 of the feedback loop (User feedback → Automated
triage → Human review → Regression corpus → Evaluation and release).

Rule-based, not LLM-based: every signal it needs (guardrail flags, user
rating) is already computed and persisted in query_log by the time a row
reaches here, so classifying it is a matter of ranking those signals by
severity, not another model call.

Rules are ordered most-actionable-first and the first match wins — a row
that would match multiple rules (e.g. a low rating *and* a guardrail
integrity flag) is filed under whichever signal is more informative for a
reviewer, not both.
"""
from __future__ import annotations

from dataclasses import dataclass

# Priority order, most urgent first — used both for display and for
# QueryLogRepository.fetch_for_review's ORDER BY.
PRIORITIES = ("high", "medium", "low")


@dataclass(frozen=True)
class TriageResult:
    category: str
    priority: str  # one of PRIORITIES
    reason: str


def triage_row(row: dict) -> TriageResult:
    """Classify a single query_log row (as returned by fetch_untriaged/fetch_row).

    Reads whatever subset of these keys is present, all optional except the
    row itself: blocked, block_reason, hallucination_risk, citation_valid,
    number_consistent, abstention_detected, feedback_rating, feedback_comment.
    """
    blocked = bool(row.get("blocked"))
    rating = row.get("feedback_rating")
    comment = row.get("feedback_comment")

    # 1. Post-generation guardrail flagged a real integrity problem but the
    #    answer was still shown to the user — this is the one case where the
    #    system itself knows something may be wrong, independent of whether
    #    anyone has rated it yet. Most dangerous class, checked first.
    if not blocked and (
        row.get("hallucination_risk") == "high"
        or row.get("citation_valid") is False
        or row.get("number_consistent") is False
    ):
        flags = [
            name for name, bad in (
                ("hallucination_risk=high", row.get("hallucination_risk") == "high"),
                ("citation_invalid", row.get("citation_valid") is False),
                ("numbers_inconsistent", row.get("number_consistent") is False),
            ) if bad
        ]
        return TriageResult(
            category="integrity_flag_not_blocked",
            priority="high",
            reason=f"post-guardrail flagged but answer was shown: {', '.join(flags)}",
        )

    # 2. User explicitly rated it badly, with a comment giving context —
    #    the most immediately reviewable class of real user-reported issue.
    #    A direct human explanation always trumps anything we can infer
    #    below, so this is checked before the more specific inferred rules.
    if rating is not None and rating <= 2 and comment:
        return TriageResult(
            category="user_reported_bad_answer",
            priority="high",
            reason=f"rating={rating} with comment",
        )

    # 3. The pipeline abstained ("not available in the provided documents")
    #    and the user wasn't satisfied — could be a correct abstention on a
    #    genuinely out-of-scope question, or a retrieval miss on an in-scope
    #    one. Needs a human to tell the two apart. Checked before the generic
    #    low-rating rule below so a rating<=2 abstention gets this more
    #    specific, more informative category instead of the generic one.
    if row.get("abstention_detected") and rating is not None and rating <= 3:
        return TriageResult(
            category="unwarranted_abstention",
            priority="medium",
            reason=f"abstained with rating={rating}",
        )

    # 4. Bad rating, no comment, not an abstention — still worth reviewing,
    #    just less context than case 2.
    if rating is not None and rating <= 2:
        return TriageResult(
            category="user_reported_bad_answer",
            priority="medium",
            reason=f"rating={rating}, no comment",
        )

    # 5. Pre-generation guardrail blocked the request outright. The system
    #    behaved as designed — only worth a closer look if it's a candidate
    #    false-positive block, which negative feedback would already have
    #    caught above. Otherwise this is routine, low-priority confirmation.
    if blocked:
        return TriageResult(
            category="guardrail_block",
            priority="low",
            reason=row.get("block_reason") or "blocked, no reason recorded",
        )

    # 6. Positive feedback — not urgent, but a candidate for promotion into
    #    the regression corpus as a known-good example later.
    if rating is not None and rating >= 4:
        return TriageResult(
            category="positive_feedback",
            priority="low",
            reason=f"rating={rating}",
        )

    # 7. Nothing actionable yet — no feedback, no guardrail flags.
    return TriageResult(
        category="unclassified",
        priority="low",
        reason="no feedback or guardrail signal yet",
    )


def triage_pending(query_log, *, limit: int = 200) -> dict[str, int]:
    """Triage every row in query_log that hasn't been triaged yet.

    query_log is a QueryLogRepository — passed as a plain object (not
    type-hinted against the class) to avoid a storage-layer import here;
    this module only depends on the fetch_untriaged/save_triage shape.

    Returns counts per category, for a caller (e.g. scripts/triage_feedback.py)
    to print a summary.
    """
    rows = query_log.fetch_untriaged(limit=limit)
    counts: dict[str, int] = {}
    for row in rows:
        result = triage_row(row)
        query_log.save_triage(
            query_id=str(row["query_id"]),
            category=result.category,
            priority=result.priority,
            reason=result.reason,
        )
        counts[result.category] = counts.get(result.category, 0) + 1
    return counts
