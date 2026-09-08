"""Bounded plan-act-observe-evaluate-replan loop."""

from __future__ import annotations

from financial_pipeline.intelligence.evidence import EvidenceEvaluator, EvidenceRequirement
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import ReasoningHarness
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.replan import EvidenceReplanner
from financial_pipeline.intelligence.research_plan import ResearchPlan


class ReasoningLoop:
    def __init__(
        self,
        executor: ResearchExecutor,
        evaluator: EvidenceEvaluator,
        replanner: EvidenceReplanner,
        harness: ReasoningHarness,
    ) -> None:
        self._executor = executor
        self._evaluator = evaluator
        self._replanner = replanner
        self._harness = harness

    def run(
        self,
        *,
        state: ReasoningState,
        initial_plan: ResearchPlan,
        requirements: tuple[EvidenceRequirement, ...],
    ) -> ReasoningState:
        plan = initial_plan

        while True:
            state.investigation_round += 1
            self._executor.execute(state, plan)

            evaluation = self._evaluator.evaluate(state, requirements)
            if evaluation.is_sufficient:
                return state

            if not self._harness.can_investigate(state):
                state.abstention_reason = "investigation-round budget exhausted before evidence became sufficient"
                return state

            decision = self._replanner.decide(evaluation)
            if not decision.should_replan:
                return state

            if not self._harness.can_replan(state):
                state.abstention_reason = "replan budget exhausted before evidence became sufficient"
                return state

            state.replan_count += 1
            plan = self._replanner.build_plan(
                objective=initial_plan.objective,
                decision=decision,
            )
