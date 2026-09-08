"""Execute research plans through registered, harness-governed capabilities."""

from __future__ import annotations

from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.harness import ReasoningHarness
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ResearchPlan
from financial_pipeline.intelligence.reasoning_trace import ReasoningTraceEventType
from financial_pipeline.intelligence.state_binding import ActionStateBinder


class ResearchExecutor:
    def __init__(self, registry: CapabilityRegistry, harness: ReasoningHarness) -> None:
        self._registry = registry
        self._harness = harness
        self._binder = ActionStateBinder()

    @property
    def registry(self) -> CapabilityRegistry:
        return self._registry

    def execute(self, state: ReasoningState, plan: ResearchPlan) -> ReasoningState:
        state.set_plan(plan)

        for action in plan.actions:
            action = self._binder.bind(state, action)
            assert state.trace is not None
            state.trace.record(
                ReasoningTraceEventType.ACTION_STARTED,
                round=state.investigation_round,
                plan_id=plan.plan_id,
                action_id=action.action_id,
                action_type=action.action_type.value,
                payload={
                    "rationale": action.rationale,
                    "metrics": list(action.metrics),
                    "evidence_types": list(action.evidence_types),
                    "checks": list(action.checks),
                    "parameters": dict(action.parameters),
                },
            )
            self._harness.validate_action(state, action)
            state.tool_calls += 1
            observation = self._registry.execute(action)
            state.add_observation(observation)
            state.trace.record(
                ReasoningTraceEventType.ACTION_FINISHED,
                round=state.investigation_round,
                plan_id=plan.plan_id,
                action_id=action.action_id,
                action_type=action.action_type.value,
                payload={
                    "status": observation.status.value,
                    "evidence_refs": list(observation.evidence_refs),
                    "tradeoff_reason": observation.tradeoff_reason,
                    "error": observation.error,
                    "failure_domain": observation.failure_domain,
                    "retryable": observation.retryable,
                    "replannable": observation.replannable,
                },
            )

            # Do not fail-fast. Independent evidence actions should continue so
            # the evaluator can make a portfolio-level tradeoff after the pass.

        return state
