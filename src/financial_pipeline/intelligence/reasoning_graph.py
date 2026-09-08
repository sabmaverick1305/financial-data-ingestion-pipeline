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
from financial_pipeline.intelligence.metric_routing import MetricOwnershipRouter
from financial_pipeline.intelligence.investment_eligibility import InvestmentEligibilityPolicy
from financial_pipeline.intelligence.plan_dependencies import PlanDependencyResolver
from financial_pipeline.intelligence.confidence import ConfidenceScorer
from financial_pipeline.intelligence.answer_synthesis import AnswerSynthesizer
from financial_pipeline.intelligence.fund_ranking import FundRanker
from financial_pipeline.intelligence.decision_pipeline import CandidateDecisionPipeline
from financial_pipeline.intelligence.planner_protocol import ResearchPlanner
from financial_pipeline.intelligence.reasoning_loop import ReasoningLoop
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.reasoning_trace import ReasoningTraceEventType
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
        self._metric_router = MetricOwnershipRouter()
        self._eligibility = InvestmentEligibilityPolicy()
        self._dependency_resolver = PlanDependencyResolver()
        self._confidence = ConfidenceScorer()
        self._synthesizer = AnswerSynthesizer()
        self._ranker = FundRanker()
        self._decision_pipeline = CandidateDecisionPipeline()
        self._loop = ReasoningLoop(executor, evaluator, replanner, harness)

    def run(self, query: str) -> GraphResult:
        visited: list[GraphNode] = [GraphNode.PLAN]
        plan, requirements = self._planner.plan(query)
        plan = self._metric_router.route(plan)
        plan, mandate = self._eligibility.apply(query, plan)
        plan = self._dependency_resolver.order(plan)

        visited.append(GraphNode.EXECUTE_AND_EVALUATE)
        state = ReasoningState(query=query)
        state.investment_mandate = mandate.mandate.value
        state.eligible_categories = list(mandate.eligible_categories)
        state.llm_calls += self._planner.llm_calls_used
        state = self._loop.run(
            state=state,
            initial_plan=plan,
            requirements=requirements,
        )

        if state.abstention_reason:
            visited.append(GraphNode.ABSTAIN)
        else:
            decisions = self._decision_pipeline.evaluate(state)
            state.candidate_decisions = {
                code: {
                    "scheme_code": decision.scheme_code,
                    "scheme_name": decision.scheme_name,
                    "category": decision.category,
                    "eligible_for_ranking": decision.eligible_for_ranking,
                    "gates": {
                        name: {
                            "status": gate.status.value,
                            "reasons": list(gate.reasons),
                            "evidence_refs": list(gate.evidence_refs),
                        }
                        for name, gate in decision.gates.items()
                    },
                }
                for code, decision in decisions.items()
            }
            assert state.trace is not None
            state.trace.record(
                ReasoningTraceEventType.DECISION_GATES_EVALUATED,
                round=state.investigation_round,
                payload={
                    "total_candidates": len(decisions),
                    "eligible_for_ranking": sum(
                        1 for decision in decisions.values()
                        if decision.eligible_for_ranking
                    ),
                    "rejected": [
                        {
                            "scheme_code": code,
                            "failed_gates": [
                                name
                                for name, gate in decision.gates.items()
                                if gate.status.value == "fail"
                            ],
                        }
                        for code, decision in decisions.items()
                        if not decision.eligible_for_ranking
                    ],
                },
            )

            allowed_codes = {
                code
                for code, decision in decisions.items()
                if decision.eligible_for_ranking
            }
            ranked = self._ranker.rank(
                state,
                limit=10,
                allowed_codes=allowed_codes,
            )
            state.ranked_funds = [fund.__dict__ for fund in ranked]
            state.trace.record(
                ReasoningTraceEventType.RANKING_COMPLETED,
                round=state.investigation_round,
                payload={
                    "ranked_count": len(ranked),
                    "top": [
                        {
                            "scheme_code": fund.scheme_code,
                            "score": fund.score,
                        }
                        for fund in ranked[:10]
                    ],
                },
            )

            confidence = self._confidence.score(state)
            state.confidence_score = confidence.score
            state.trace.record(
                ReasoningTraceEventType.CONFIDENCE_EVALUATED,
                round=state.investigation_round,
                payload={
                    "score": confidence.score,
                    "evidence_coverage": confidence.evidence_coverage,
                    "source_quality": confidence.source_quality,
                    "candidate_quality": confidence.candidate_quality,
                    "soft_evidence_penalty": confidence.soft_evidence_penalty,
                    "replan_penalty": confidence.replan_penalty,
                },
            )
            state.final_answer = self._synthesizer.synthesize(state, confidence)
            visited.append(GraphNode.FINALIZE)

        return GraphResult(state=state, visited_nodes=tuple(visited))
