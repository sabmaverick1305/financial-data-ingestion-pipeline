#!/usr/bin/env python3
"""Run the FIES reasoning graph against production Postgres + RAG capabilities."""
from __future__ import annotations

import json
import sys
from sqlalchemy import create_engine

sys.path.insert(0, "src")

from financial_pipeline.augmentation.generator import AnswerGenerator
from financial_pipeline.config import settings
from financial_pipeline.intelligence.evidence import EvidenceEvaluator
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import HarnessLimits, ReasoningHarness
from financial_pipeline.intelligence.llm_planner import LLMPlanner
from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.production_registry import build_production_registry
from financial_pipeline.intelligence.production_repository import ReasoningProductionRepository
from financial_pipeline.intelligence.reasoning_graph import ReasoningGraph
from financial_pipeline.intelligence.replan import EvidenceReplanner
from financial_pipeline.retrieval.rag import RAGPipeline
from financial_pipeline.retrieval.retriever import Retriever
from financial_pipeline.storage.document_repo import DocumentRepository

QUERY = "Give me some of the best mutual funds to invest in 2026"


def main() -> None:
    if not settings.postgres_url:
        raise RuntimeError("POSTGRES_URL is required for the production reasoning E2E")

    generator = AnswerGenerator()
    if not generator.is_configured():
        raise RuntimeError("Planner LLM is not configured")

    engine = create_engine(settings.postgres_url, pool_pre_ping=True)
    production_repo = ReasoningProductionRepository(engine)

    document_repo = DocumentRepository(settings.postgres_url)
    retriever = Retriever(document_repo)
    rag_pipeline = RAGPipeline(retriever)

    pack = ProductionCapabilityPack(
        repository=production_repo,
        rag_pipeline=rag_pipeline,
    )
    registry = build_production_registry(pack)

    planner = LLMPlanner(generator)
    harness = ReasoningHarness(
        registry,
        HarnessLimits(
            max_investigation_rounds=3,
            max_tool_calls=30,
            max_llm_calls=2,
            max_replans=2,
        ),
    )
    graph = ReasoningGraph(
        planner=planner,
        executor=ResearchExecutor(registry, harness),
        evaluator=EvidenceEvaluator(),
        replanner=EvidenceReplanner(registry),
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
    print(f"confidence_score={state.confidence_score}")
    print(f"abstention_reason={state.abstention_reason}")

    print("\n=== FINAL ANSWER ===")
    print(state.final_answer)

    print("\n=== OBSERVATIONS ===")
    for observation in state.observations:
        print(
            f"- {observation.action_type.value}: {observation.status.value}"
            f" | evidence={list(observation.evidence_refs)}"
            f" | tradeoff={observation.tradeoff_reason}"
            f" | error={observation.error}"
        )
        if observation.result is not None:
            print("  result=" + json.dumps(observation.result, default=str)[:3000])

    print("\n=== REASONING TRACE ===")
    assert state.trace is not None
    print(json.dumps(state.trace.to_log_dict(), indent=2, default=str))


if __name__ == "__main__":
    main()
