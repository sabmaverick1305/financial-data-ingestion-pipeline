"""Planner/reasoning evaluation harness for multi-query E2E runs."""
from __future__ import annotations
from dataclasses import dataclass
from time import perf_counter
from financial_pipeline.intelligence.observability_metrics import ReasoningMetricsCollector

@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_actions: tuple[str, ...] = ()

@dataclass(frozen=True)
class EvaluationResult:
    query: str
    passed: bool
    first_pass_success: bool
    missing_action_rate: float
    unsupported_metric_rate: float
    replan_rate: float
    tool_calls: int
    llm_calls: int
    latency_ms: int
    confidence: float | None

class PlannerQualityEvaluator:
    def evaluate(self, graph, cases: tuple[EvaluationCase, ...]) -> tuple[EvaluationResult, ...]:
        results = []
        collector = ReasoningMetricsCollector()
        for case in cases:
            started = perf_counter()
            graph_result = graph.run(case.query)
            state = graph_result.state
            metrics = collector.collect(state)
            present = {o.action_type.value for o in state.observations}
            missing = set(case.expected_actions) - present
            missing_rate = len(missing) / max(1, len(case.expected_actions))
            results.append(EvaluationResult(
                query=case.query,
                passed=state.abstention_reason is None and missing_rate == 0,
                first_pass_success=state.investigation_round <= 1,
                missing_action_rate=missing_rate,
                unsupported_metric_rate=metrics.unsupported_capability_rate,
                replan_rate=1.0 if state.replan_count else 0.0,
                tool_calls=state.tool_calls,
                llm_calls=state.llm_calls,
                latency_ms=int((perf_counter() - started) * 1000),
                confidence=state.confidence_score,
            ))
        return tuple(results)
