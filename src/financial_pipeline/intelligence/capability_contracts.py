"""Capability schemas that constrain planner-requested metrics before execution."""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from financial_pipeline.intelligence.research_plan import ResearchAction


@dataclass(frozen=True)
class CapabilityContract:
    """Machine-enforced schema for one registered capability.

    ``supported_metrics=None`` means the capability does not constrain metrics.
    An empty tuple means the capability accepts no metric-specific request.
    """

    supported_metrics: tuple[str, ...] | None = None
    metric_aliases: dict[str, str] = field(default_factory=dict)

    def normalize(self, action: ResearchAction) -> tuple[ResearchAction, tuple[str, ...]]:
        if self.supported_metrics is None or not action.metrics:
            return action, ()

        supported = set(self.supported_metrics)
        normalized: list[str] = []
        unsupported: list[str] = []

        for metric in action.metrics:
            canonical = self.metric_aliases.get(metric, metric)
            if canonical in supported:
                if canonical not in normalized:
                    normalized.append(canonical)
            else:
                unsupported.append(metric)

        return replace(action, metrics=tuple(normalized)), tuple(unsupported)
