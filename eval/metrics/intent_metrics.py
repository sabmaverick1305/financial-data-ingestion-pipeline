"""Phase 1: Intent & Routing evaluation metrics."""
from __future__ import annotations
from collections import defaultdict
from typing import Any


def accuracy(predictions: list[dict], labels: list[dict], field: str) -> float:
    """Fraction of predictions where prediction[field] == label[field]."""
    if not predictions:
        return 0.0
    correct = sum(1 for p, l in zip(predictions, labels) if p.get(field) == l.get(field))
    return correct / len(predictions)


def routing_accuracy(predictions: list[dict], labels: list[dict]) -> float:
    return accuracy(predictions, labels, "routing")


def intent_type_accuracy(predictions: list[dict], labels: list[dict]) -> float:
    return accuracy(predictions, labels, "intent_type")


def metric_accuracy(predictions: list[dict], labels: list[dict]) -> float:
    return accuracy(predictions, labels, "metric")


def scheme_type_accuracy(predictions: list[dict], labels: list[dict]) -> float:
    if not predictions:
        return 0.0
    correct = 0
    for p, l in zip(predictions, labels):
        predicted = sorted(str(v).lower() for v in (p.get("scheme_types") or []))
        expected = sorted(str(v).lower() for v in (l.get("scheme_types") or []))
        if predicted == expected:
            correct += 1
    return correct / len(predictions)


def year_extraction_accuracy(predictions: list[dict], labels: list[dict]) -> float:
    correct = 0
    total = 0
    for p, l in zip(predictions, labels):
        for field in ("year", "year_from", "year_to"):
            lv = l.get(field)
            if lv is not None:
                total += 1
                if p.get(field) == lv:
                    correct += 1
    return correct / total if total else 0.0


def per_class_precision_recall(
    predictions: list[dict], labels: list[dict], field: str
) -> dict[str, dict[str, float]]:
    """Per-class precision and recall for a categorical field."""
    classes: set[str] = set()
    for l in labels:
        v = l.get(field)
        if v:
            classes.add(v)

    results: dict[str, dict[str, float]] = {}
    for cls in sorted(classes):
        tp = sum(1 for p, l in zip(predictions, labels) if p.get(field) == cls and l.get(field) == cls)
        fp = sum(1 for p, l in zip(predictions, labels) if p.get(field) == cls and l.get(field) != cls)
        fn = sum(1 for p, l in zip(predictions, labels) if p.get(field) != cls and l.get(field) == cls)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        results[cls] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return results


def confusion_matrix(
    predictions: list[dict], labels: list[dict], field: str
) -> dict[str, dict[str, int]]:
    """Returns {actual: {predicted: count}}."""
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p, l in zip(predictions, labels):
        actual = str(l.get(field, "unknown"))
        predicted = str(p.get(field, "unknown"))
        matrix[actual][predicted] += 1
    return {k: dict(v) for k, v in matrix.items()}


def summary(predictions: list[dict], labels: list[dict]) -> dict[str, Any]:
    return {
        "routing_accuracy": routing_accuracy(predictions, labels),
        "intent_type_accuracy": intent_type_accuracy(predictions, labels),
        "metric_accuracy": metric_accuracy(predictions, labels),
        "scheme_type_accuracy": scheme_type_accuracy(predictions, labels),
        "year_extraction_accuracy": year_extraction_accuracy(predictions, labels),
        "per_class_routing": per_class_precision_recall(predictions, labels, "routing"),
        "per_class_intent": per_class_precision_recall(predictions, labels, "intent_type"),
        "per_class_metric": per_class_precision_recall(predictions, labels, "metric"),
    }
