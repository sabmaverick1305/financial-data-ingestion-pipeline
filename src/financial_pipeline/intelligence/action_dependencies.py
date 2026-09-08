"""Action-level dependency graph for reasoning execution."""
from __future__ import annotations
from financial_pipeline.intelligence.research_plan import ActionType

_DEPENDENCIES: dict[ActionType, tuple[ActionType, ...]] = {
    ActionType.DISCOVER_FUNDS: (ActionType.DISCOVER_CATEGORIES,),
    ActionType.FETCH_PERFORMANCE: (ActionType.DISCOVER_FUNDS,),
    ActionType.FETCH_FLOWS: (ActionType.DISCOVER_FUNDS,),
    ActionType.FETCH_AUM: (ActionType.DISCOVER_FUNDS,),
    ActionType.COMPUTE_RETURNS: (ActionType.FETCH_PERFORMANCE,),
    ActionType.COMPUTE_RISK: (ActionType.FETCH_PERFORMANCE,),
    ActionType.COMPARE_PEERS: (ActionType.DISCOVER_FUNDS, ActionType.FETCH_PERFORMANCE),
    ActionType.RETRIEVE_EVIDENCE: (ActionType.DISCOVER_FUNDS,),
    ActionType.CHECK_CONTRADICTIONS: (ActionType.COMPARE_PEERS, ActionType.RETRIEVE_EVIDENCE),
}

class ActionDependencyGraph:
    def dependencies(self, action_type: ActionType) -> tuple[ActionType, ...]:
        return _DEPENDENCIES.get(action_type, ())
    def ready(self, action_type: ActionType, completed: set[ActionType]) -> bool:
        return all(dep in completed for dep in self.dependencies(action_type))
    def layers(self, actions: tuple[ActionType, ...]) -> tuple[tuple[ActionType, ...], ...]:
        remaining = list(dict.fromkeys(actions))
        completed: set[ActionType] = set()
        layers: list[tuple[ActionType, ...]] = []
        while remaining:
            ready = tuple(a for a in remaining if self.ready(a, completed))
            if not ready:
                ready = (remaining[0],)
            layers.append(ready)
            for action in ready:
                completed.add(action)
                remaining.remove(action)
        return tuple(layers)
