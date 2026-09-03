from financial_pipeline.feedback.triage import triage_pending, triage_row


def _row(**overrides) -> dict:
    base = {
        "blocked": False,
        "block_reason": None,
        "hallucination_risk": "low",
        "citation_valid": True,
        "number_consistent": True,
        "abstention_detected": False,
        "feedback_rating": None,
        "feedback_comment": None,
    }
    base.update(overrides)
    return base


def test_integrity_flag_wins_even_without_feedback() -> None:
    result = triage_row(_row(hallucination_risk="high"))
    assert result.category == "integrity_flag_not_blocked"
    assert result.priority == "high"


def test_integrity_flag_wins_over_a_positive_rating() -> None:
    # A guardrail-detected integrity issue is more actionable than a user's
    # rating not reflecting it yet — this is the "silently wrong answer"
    # case and should always surface regardless of feedback.
    result = triage_row(_row(citation_valid=False, feedback_rating=5))
    assert result.category == "integrity_flag_not_blocked"


def test_low_rating_with_comment_is_high_priority() -> None:
    result = triage_row(_row(feedback_rating=1, feedback_comment="wrong AMC totally"))
    assert result.category == "user_reported_bad_answer"
    assert result.priority == "high"


def test_low_rating_without_comment_is_medium_priority() -> None:
    result = triage_row(_row(feedback_rating=2))
    assert result.category == "user_reported_bad_answer"
    assert result.priority == "medium"


def test_blocked_request_is_low_priority_by_default() -> None:
    result = triage_row(_row(blocked=True, block_reason="Investment advice request"))
    assert result.category == "guardrail_block"
    assert result.priority == "low"
    assert result.reason == "Investment advice request"


def test_blocked_request_with_bad_rating_is_reported_as_user_feedback_not_guardrail() -> None:
    # A user who complains about a blocked answer is more useful to review
    # as a possible false-positive block than as a routine guardrail event.
    result = triage_row(_row(blocked=True, feedback_rating=1, feedback_comment="I just wanted a number"))
    assert result.category == "user_reported_bad_answer"


def test_unwarranted_abstention() -> None:
    result = triage_row(_row(abstention_detected=True, feedback_rating=2))
    assert result.category == "unwarranted_abstention"
    assert result.priority == "medium"


def test_positive_feedback() -> None:
    result = triage_row(_row(feedback_rating=5))
    assert result.category == "positive_feedback"
    assert result.priority == "low"


def test_unclassified_when_nothing_to_go_on() -> None:
    result = triage_row(_row())
    assert result.category == "unclassified"
    assert result.priority == "low"


class _FakeQueryLog:
    """Minimal stand-in for QueryLogRepository — just enough surface for
    triage_pending, so this test doesn't need a real Postgres connection."""

    def __init__(self, rows: list[dict]) -> None:
        self._rows = rows
        self.saved: list[dict] = []

    def fetch_untriaged(self, limit: int) -> list[dict]:
        return self._rows[:limit]

    def save_triage(self, *, query_id: str, category: str, priority: str, reason: str) -> None:
        self.saved.append({
            "query_id": query_id, "category": category,
            "priority": priority, "reason": reason,
        })


def test_triage_pending_processes_every_row_and_returns_category_counts() -> None:
    rows = [
        {"query_id": "q1", **_row(hallucination_risk="high")},
        {"query_id": "q2", **_row(feedback_rating=1, feedback_comment="bad")},
        {"query_id": "q3", **_row()},
    ]
    fake_log = _FakeQueryLog(rows)

    counts = triage_pending(fake_log, limit=10)

    assert counts == {
        "integrity_flag_not_blocked": 1,
        "user_reported_bad_answer": 1,
        "unclassified": 1,
    }
    assert len(fake_log.saved) == 3
    assert {s["query_id"] for s in fake_log.saved} == {"q1", "q2", "q3"}
