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

    def normalize(self, action: ResearchAction) -> tuple[ResearchAction, tuple[str, ...]]:
        metrics, unsupported_metrics = self._normalize_values(
            action.metrics,
            self.supported_metrics,
            self.metric_aliases,
            self.metric_ontology.canonicalize,
        )
        evidence_types, unsupported_evidence = self._normalize_values(
            action.evidence_types,
            self.supported_evidence_types,
            self.evidence_aliases,
            self.evidence_ontology.canonicalize,
        )

        unsupported = tuple(
            [*unsupported_metrics, *unsupported_evidence]
        )
        return replace(
            action,
            metrics=metrics,
            evidence_types=evidence_types,
        ), unsupported

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
