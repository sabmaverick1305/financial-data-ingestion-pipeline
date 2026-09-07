"""Execute research plans through registered, harness-governed capabilities."""

from __future__ import annotations

from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.harness import ReasoningHarness
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ResearchPlan


class ResearchExecutor:
    def __init__(self, registry: CapabilityRegistry, harness: ReasoningHarness) -> None:
        self._registry = registry
        self._harness = harness

    def execute(self, state: ReasoningState, plan: ResearchPlan) -> ReasoningState:
        state.set_plan(plan)

        for action in plan.actions:
            self._harness.validate_action(state, action)
            state.tool_calls += 1
            observation = self._registry.execute(action)
            state.add_observation(observation)

            if observation.status is ActionStatus.FAILED:
                break

        return state
