"""Capability schemas that constrain planner requests before execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from financial_pipeline.intelligence.evidence_ontology import EvidenceOntology
from financial_pipeline.intelligence.metric_ontology import MetricOntology
from financial_pipeline.intelligence.research_plan import ResearchAction


@dataclass(frozen=True)
class CapabilityContract:
    """Machine-enforced schema for one registered capability."""

    supported_metrics: tuple[str, ...] | None = None
    metric_aliases: dict[str, str] = field(default_factory=dict)
    supported_evidence_types: tuple[str, ...] | None = None
    evidence_aliases: dict[str, str] = field(default_factory=dict)
    metric_ontology: MetricOntology = field(default_factory=MetricOntology)
    evidence_ontology: EvidenceOntology = field(default_factory=EvidenceOntology)

    def normalize(
        self,
        action: ResearchAction,
    ) -> tuple[ResearchAction, tuple[str, ...], tuple[str, ...]]:
        """Normalize metric and evidence vocabularies independently.

        For backward compatibility, recognized evidence terms accidentally placed
        in ``metrics`` are migrated to ``evidence_types`` before validation.
        """
        metric_values = list(action.metrics)
        evidence_values = list(action.evidence_types)

        if self.supported_evidence_types is not None and metric_values:
            retained_metrics: list[str] = []
            for value in metric_values:
                evidence_canonical = self.evidence_ontology.canonicalize(value)
                if (
                    self.evidence_ontology.is_known(value)
                    or evidence_canonical in set(self.supported_evidence_types)
                ):
                    evidence_values.append(value)
                else:
                    retained_metrics.append(value)
            metric_values = retained_metrics

        metrics, unsupported_metrics = self._normalize_values(
            tuple(metric_values),
            self.supported_metrics,
            self.metric_aliases,
            self.metric_ontology.canonicalize,
        )
        evidence_types, unsupported_evidence = self._normalize_values(
            tuple(evidence_values),
            self.supported_evidence_types,
            self.evidence_aliases,
            self.evidence_ontology.canonicalize,
        )

        return (
            replace(action, metrics=metrics, evidence_types=evidence_types),
            unsupported_metrics,
            unsupported_evidence,
        )

    @staticmethod
    def _normalize_values(values, supported_values, aliases, canonicalize):
        if supported_values is None or not values:
            return values, ()

        supported = set(supported_values)
        normalized: list[str] = []
        unsupported: list[str] = []
        for value in values:
            ontology_value = canonicalize(value)
            canonical = aliases.get(value, aliases.get(ontology_value, ontology_value))
            if canonical in supported:
                if canonical not in normalized:
                    normalized.append(canonical)
            else:
                unsupported.append(value)
        return tuple(normalized), tuple(unsupported)
