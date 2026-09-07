"""Shared case-file state for graph, loop, and harness execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from financial_pipeline.intelligence.research_plan import (
    ActionObservation,
    ResearchPlan,
)


@dataclass
class ReasoningState:
    query: str
    plan: ResearchPlan | None = None
    observations: list[ActionObservation] = field(default_factory=list)
    pending_action_ids: list[str] = field(default_factory=list)
    investigation_round: int = 0
    tool_calls: int = 0
    llm_calls: int = 0
    replan_count: int = 0
    final_answer: str | None = None
    abstention_reason: str | None = None

    def set_plan(self, plan: ResearchPlan) -> None:
        self.plan = plan
        self.pending_action_ids = [action.action_id for action in plan.actions]

    def add_observation(self, observation: ActionObservation) -> None:
        self.observations.append(observation)
        if observation.action_id in self.pending_action_ids:
            self.pending_action_ids.remove(observation.action_id)

    @property
    def is_complete(self) -> bool:
        return self.plan is not None and not self.pending_action_ids
