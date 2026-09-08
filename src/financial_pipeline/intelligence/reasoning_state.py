"""Shared case-file state for graph, loop, and harness execution."""

from __future__ import annotations

from dataclasses import dataclass, field

from financial_pipeline.intelligence.research_plan import (
    ActionObservation,
    ResearchPlan,
)
from financial_pipeline.intelligence.reasoning_trace import ReasoningTrace


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
    confidence_score: float | None = None
    ranked_funds: list[dict] = field(default_factory=list)
    evidence_tradeoffs: list[str] = field(default_factory=list)
    soft_evidence_gaps: list[str] = field(default_factory=list)
    abstention_reason: str | None = None
    trace: ReasoningTrace | None = None

    def __post_init__(self) -> None:
        if self.trace is None:
            self.trace = ReasoningTrace(query=self.query)

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
