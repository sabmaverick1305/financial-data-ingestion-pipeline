"""Graph-style orchestration for the FIES reasoning engine.

This module keeps graph transitions explicit while delegating repeated
plan-act-observe-evaluate cycles to ReasoningLoop and policy enforcement
to ReasoningHarness.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from financial_pipeline.intelligence.evidence import EvidenceEvaluator
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import ReasoningHarness
from financial_pipeline.intelligence.planner_protocol import ResearchPlanner
from financial_pipeline.intelligence.reasoning_loop import ReasoningLoop
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.replan import EvidenceReplanner


class GraphNode(StrEnum):
    PLAN = "plan"
    EXECUTE_AND_EVALUATE = "execute_and_evaluate"
    FINALIZE = "finalize"
    ABSTAIN = "abstain"


@dataclass(frozen=True)
class GraphResult:
    state: ReasoningState
    visited_nodes: tuple[GraphNode, ...]


class ReasoningGraph:
    """Small deterministic graph used before introducing LangGraph planner nodes."""

    def __init__(
        self,
        *,
        planner: ResearchPlanner,
        executor: ResearchExecutor,
        evaluator: EvidenceEvaluator,
        replanner: EvidenceReplanner,
        harness: ReasoningHarness,
    ) -> None:
        self._planner = planner
        self._loop = ReasoningLoop(executor, evaluator, replanner, harness)

    def run(self, query: str) -> GraphResult:
        visited: list[GraphNode] = [GraphNode.PLAN]
        plan, requirements = self._planner.plan(query)

        visited.append(GraphNode.EXECUTE_AND_EVALUATE)
        state = ReasoningState(query=query)
        state.llm_calls += self._planner.llm_calls_used
        state = self._loop.run(
            state=state,
            initial_plan=plan,
            requirements=requirements,
        )

        if state.abstention_reason:
            visited.append(GraphNode.ABSTAIN)
        else:
            visited.append(GraphNode.FINALIZE)

        return GraphResult(state=state, visited_nodes=tuple(visited))
