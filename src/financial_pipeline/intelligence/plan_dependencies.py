"""Deterministically order research actions according to graph dependencies."""
from __future__ import annotations
from financial_pipeline.intelligence.action_dependencies import ActionDependencyGraph
from financial_pipeline.intelligence.research_plan import ResearchPlan

class PlanDependencyResolver:
    def __init__(self) -> None:
        self._graph = ActionDependencyGraph()

    def order(self, plan: ResearchPlan) -> ResearchPlan:
        by_type = {action.action_type: action for action in plan.actions}
        layers = self._graph.layers(tuple(by_type))
        ordered = tuple(
            by_type[action_type]
            for layer in layers
            for action_type in layer
            if action_type in by_type
        )
        return ResearchPlan(
            objective=plan.objective,
            actions=ordered,
            assumptions=plan.assumptions,
            plan_id=plan.plan_id,
        )
