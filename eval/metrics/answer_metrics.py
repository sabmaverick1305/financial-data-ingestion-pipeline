"""Phase 4: Answer Quality evaluation metrics."""
from __future__ import annotations
import re
from typing import Any


def fact_coverage(answers: list[str], labels: list[dict]) -> float:
    """Fraction of required facts found (case-insensitive substring) in answers."""
    scores = []
    for answer, label in zip(answers, labels):
        if label is None:
            continue
        required = label.get("required_facts", [])
        if not required:
            continue
        answer_lower = (answer or "").lower()
        found = sum(1 for fact in required if fact.lower() in answer_lower)
        scores.append(found / len(required))
    return sum(scores) / len(scores) if scores else 0.0


def citation_presence_rate(answers: list[str], labels: list[dict]) -> float:
    """Fraction of answers that include [1] when required_citation is True."""
    required_total = 0
    correct = 0
    for answer, label in zip(answers, labels):
        if label is None:
            continue
        if not label.get("required_citation", False):
            continue
        required_total += 1
        tag = label.get("citation_tag", "[1]")
        if tag in (answer or ""):
            correct += 1
    return correct / required_total if required_total else 1.0


def safety_compliance_rate(answers: list[str], labels: list[dict]) -> float:
    """Fraction of answers that do NOT contain any forbidden phrases."""
    scores = []
    for answer, label in zip(answers, labels):
        if label is None:
            continue
        forbidden = label.get("should_not_contain", [])
        if not forbidden:
            scores.append(1.0)
            continue
        answer_lower = (answer or "").lower()
        violations = sum(1 for phrase in forbidden if phrase.lower() in answer_lower)
        scores.append(1.0 if violations == 0 else 0.0)
    return sum(scores) / len(scores) if scores else 1.0


def required_content_rate(answers: list[str], labels: list[dict]) -> float:
    """Fraction of answers that contain all phrases listed in should_contain (failure queries)."""
    scores = []
    for answer, label in zip(answers, labels):
        if label is None:
            continue
        must_have = label.get("should_contain", [])
        if not must_have:
            continue
        answer_lower = (answer or "").lower()
        found = sum(1 for phrase in must_have if phrase.lower() in answer_lower)
        scores.append(found / len(must_have))
    return sum(scores) / len(scores) if scores else 1.0


def hallucination_rate(answers: list[str], labels: list[dict]) -> float:
    """Fraction of answers containing hallucination markers (inverse of safety compliance for those markers)."""
    markers = ["fabricated", "invented", "as an estimate", "approximately", "i believe", "likely around"]
    flagged = 0
    total = 0
    for answer, label in zip(answers, labels):
        if label is None:
            continue
        answer_type = label.get("answer_type", "")
        if answer_type not in ("numeric", "numeric_or_table", "table_or_trend", "trend_narrative"):
            continue
        total += 1
        answer_lower = (answer or "").lower()
        if any(m in answer_lower for m in markers):
            flagged += 1
    return flagged / total if total else 0.0


def answer_length_stats(answers: list[str]) -> dict[str, float]:
    lengths = [len((a or "").split()) for a in answers]
    if not lengths:
        return {"mean": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": sum(lengths) / len(lengths),
        "min": float(min(lengths)),
        "max": float(max(lengths)),
    }


def summary(answers: list[str], labels: list[dict]) -> dict[str, Any]:
    return {
        "fact_coverage": fact_coverage(answers, labels),
        "citation_presence_rate": citation_presence_rate(answers, labels),
        "safety_compliance_rate": safety_compliance_rate(answers, labels),
        "required_content_rate": required_content_rate(answers, labels),
        "hallucination_rate": hallucination_rate(answers, labels),
        "answer_length": answer_length_stats(answers),
    }
