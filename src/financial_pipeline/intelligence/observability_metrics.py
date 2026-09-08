"""Reasoning-level observability metrics."""
from __future__ import annotations
from dataclasses import dataclass
from financial_pipeline.intelligence.reasoning_state import ReasoningState

@dataclass(frozen=True)
class ReasoningMetrics:
    planner_repair_rate: float
    unsupported_capability_rate: float
    evidence_sufficiency: float
    replans: int
    tool_calls: int
    llm_calls: int

class ReasoningMetricsCollector:
    def collect(self, state: ReasoningState) -> ReasoningMetrics:
        total = max(1, len(state.observations))
        unsupported = sum(1 for o in state.observations if o.error and "unsupported capability" in o.error)
        return ReasoningMetrics(
            planner_repair_rate=1.0 if state.llm_calls > 1 else 0.0,
            unsupported_capability_rate=unsupported / total,
            evidence_sufficiency=0.0 if state.abstention_reason else 1.0,
            replans=state.replan_count,
            tool_calls=state.tool_calls,
            llm_calls=state.llm_calls,
        )
