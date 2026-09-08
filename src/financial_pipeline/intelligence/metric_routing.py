"""Normalize planner actions, operations, checks, and metric ownership."""
from __future__ import annotations
import re
from dataclasses import replace
from financial_pipeline.intelligence.metric_ontology import MetricOntology
from financial_pipeline.intelligence.metric_semantics import MetricSemanticsRegistry
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan

_CHECK_TERMS = {"performance_consistency", "style_drift", "flow_return_divergence", "strategy_holdings_mismatch"}

class MetricOwnershipRouter:
    def __init__(self) -> None:
        self._ontology = MetricOntology()
        self._semantics = MetricSemanticsRegistry()

    def route(self, plan: ResearchPlan) -> ResearchPlan:
        buckets: dict = {}
        order: list = []
        for original in plan.actions:
            action = self._normalize_action_semantics(original)
            local: list[str] = []
            for raw in action.metrics:
                metric = self._ontology.canonicalize(raw)
                owner = self._semantics.owner(metric)
                target = owner or action.action_type
                if target == action.action_type:
                    local.append(metric)
                else:
                    self._append(buckets, order, target, metric, action)
            action = replace(action, metrics=tuple(dict.fromkeys(local)))
            if action.action_type is ActionType.COMPARE_PEERS and not action.metrics:
                action = replace(action, metrics=("percentile_rank", "peer_outperformance"))
            self._merge_action(buckets, order, action)
        return ResearchPlan(objective=plan.objective, actions=tuple(buckets[t] for t in order), assumptions=plan.assumptions, plan_id=plan.plan_id)

    def _normalize_action_semantics(self, action: ResearchAction) -> ResearchAction:
        metrics = list(action.metrics)
        checks = list(action.checks)
        parameters = dict(action.parameters)

        if action.action_type is ActionType.DISCOVER_FUNDS:
            retained = []
            for raw in metrics:
                key = raw.strip().lower().replace("-", "_").replace(" ", "_")
                match = re.fullmatch(r"top_(\d+)_by_(.+)", key)
                if match:
                    parameters.setdefault("rank_by", self._ontology.canonicalize(match.group(2)))
                    parameters.setdefault("sort_order", "desc")
                    parameters.setdefault("limit", int(match.group(1)))
                    continue
                retained.append(raw)
            metrics = retained

        if action.action_type is ActionType.CHECK_CONTRADICTIONS:
            retained = []
            for raw in metrics:
                key = raw.strip().lower().replace("-", "_").replace(" ", "_")
                if key in _CHECK_TERMS:
                    checks.append(key)
                else:
                    retained.append(raw)
            metrics = retained
            if not checks:
                checks.extend(("performance_consistency", "style_drift", "flow_return_divergence"))

        return replace(action, metrics=tuple(metrics), checks=tuple(dict.fromkeys(checks)), parameters=parameters)

    @staticmethod
    def _append(buckets, order, target, metric, source):
        if target not in buckets:
            buckets[target] = ResearchAction(action_type=target, metrics=(metric,), rationale=f"metric ownership rerouted from {source.action_type.value}")
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
        buckets[action.action_type] = replace(existing, metrics=tuple(dict.fromkeys((*existing.metrics, *action.metrics))), evidence_types=tuple(dict.fromkeys((*existing.evidence_types, *action.evidence_types))), checks=tuple(dict.fromkeys((*existing.checks, *action.checks))), parameters={**existing.parameters, **action.parameters})
