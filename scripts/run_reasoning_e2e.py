#!/usr/bin/env python3
"""Run one pre-LLM FIES reasoning query and print the full reasoning trace."""

from __future__ import annotations

import json
import sys

sys.path.insert(0, "src")

from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.evidence import EvidenceEvaluator
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import HarnessLimits, ReasoningHarness
from financial_pipeline.intelligence.mock_capabilities import MockVerifiedCapabilityPack
from financial_pipeline.intelligence.mock_planner import MockPlanner
from financial_pipeline.intelligence.reasoning_graph import ReasoningGraph
from financial_pipeline.intelligence.replan import EvidenceReplanner


QUERY = "Give me some of the best mutual funds to invest in 2026"


def main() -> None:
    registry = CapabilityRegistry()
    capability_pack = MockVerifiedCapabilityPack()
    capability_pack.register_all(registry)

    harness = ReasoningHarness(
        registry,
        HarnessLimits(
            max_investigation_rounds=3,
            max_tool_calls=30,
            max_llm_calls=0,
            max_replans=2,
        ),
    )

    graph = ReasoningGraph(
        planner=MockPlanner(),
        executor=ResearchExecutor(registry, harness),
        evaluator=EvidenceEvaluator(),
        replanner=EvidenceReplanner(),
        harness=harness,
    )

    result = graph.run(QUERY)
    state = result.state

    print("\n=== QUERY ===")
    print(QUERY)

    print("\n=== GRAPH NODES ===")
    for node in result.visited_nodes:
        print(f"- {node.value}")

    print("\n=== SUMMARY ===")
    print(f"investigation_rounds={state.investigation_round}")
    print(f"replans={state.replan_count}")
    print(f"tool_calls={state.tool_calls}")
    print(f"llm_calls={state.llm_calls}")
    print(f"abstention_reason={state.abstention_reason}")

    print("\n=== OBSERVATIONS ===")
    for observation in state.observations:
        print(
            f"- {observation.action_type.value}: {observation.status.value}"
            f" | evidence={list(observation.evidence_refs)}"
            f" | tradeoff={observation.tradeoff_reason}"
            f" | error={observation.error}"
        )

    print("\n=== REASONING TRACE ===")
    assert state.trace is not None
    print(json.dumps(state.trace.to_log_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
