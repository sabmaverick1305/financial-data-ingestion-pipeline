"""Normalize planner actions and route canonical metrics to owning capabilities."""
from __future__ import annotations
from dataclasses import replace
from financial_pipeline.intelligence.metric_ontology import MetricOntology
from financial_pipeline.intelligence.metric_semantics import MetricSemanticsRegistry
from financial_pipeline.intelligence.research_plan import ResearchAction, ResearchPlan

class MetricOwnershipRouter:
    def __init__(self) -> None:
        self._ontology = MetricOntology()
        self._semantics = MetricSemanticsRegistry()

    def route(self, plan: ResearchPlan) -> ResearchPlan:
        buckets: dict = {}
        order: list = []
        for action in plan.actions:
            local: list[str] = []
            for raw in action.metrics:
                metric = self._ontology.canonicalize(raw)
                owner = self._semantics.owner(metric)
                target = owner or action.action_type
                if target == action.action_type:
                    local.append(metric)
                else:
                    self._append(buckets, order, target, metric, action)
            self._merge_action(buckets, order, replace(action, metrics=tuple(dict.fromkeys(local))))
        return ResearchPlan(
            objective=plan.objective,
            actions=tuple(buckets[action_type] for action_type in order),
            assumptions=plan.assumptions,
            plan_id=plan.plan_id,
        )

    @staticmethod
    def _append(buckets, order, target, metric, source):
        if target not in buckets:
            buckets[target] = ResearchAction(
                action_type=target, metrics=(metric,),
                rationale=f"metric ownership rerouted from {source.action_type.value}",
            )
            order.append(target)
        else:
            existing = buckets[target]
            buckets[target] = replace(existing, metrics=tuple(dict.fromkeys((*existing.metrics, metric))))

    @staticmethod
    def _merge_action(buckets, order, action):
        existing = buckets.get(action.action_type)
        if existing is None:
            buckets[action.action_type] = action
            order.append(action.action_type)
            return
        buckets[action.action_type] = replace(
            existing,
            metrics=tuple(dict.fromkeys((*existing.metrics, *action.metrics))),
            evidence_types=tuple(dict.fromkeys((*existing.evidence_types, *action.evidence_types))),
            parameters={**existing.parameters, **action.parameters},
        )
