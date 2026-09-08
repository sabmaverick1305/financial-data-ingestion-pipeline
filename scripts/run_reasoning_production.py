#!/usr/bin/env python3
"""Run the FIES reasoning graph against production Postgres + RAG capabilities."""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from sqlalchemy import create_engine

sys.path.insert(0, "src")

from financial_pipeline.augmentation.generator import AnswerGenerator
from financial_pipeline.config import settings
from financial_pipeline.intelligence.evidence import EvidenceEvaluator
from financial_pipeline.intelligence.executor import ResearchExecutor
from financial_pipeline.intelligence.harness import HarnessLimits, ReasoningHarness
from financial_pipeline.intelligence.llm_planner import LLMPlanner
from financial_pipeline.intelligence.observability import ReasoningObservability
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
    production_repo.ensure_latency_indexes()

    document_repo = DocumentRepository(settings.postgres_url)
    document_repo.create_tables()
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

    t0 = time.perf_counter()
    result = graph.run(QUERY)
    total_latency_ms = int((time.perf_counter() - t0) * 1000)
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


    print("\n=== DECISION FUNNEL ===")
    decisions = list(state.candidate_decisions.values())
    discovered = len(decisions)
    eligibility_pass = sum(
        1 for decision in decisions
        if decision.get("gates", {}).get("eligibility", {}).get("status") == "pass"
    )
    data_quality_pass = sum(
        1 for decision in decisions
        if decision.get("gates", {}).get("data_quality", {}).get("status") == "pass"
    )
    evidence_pass = sum(
        1 for decision in decisions
        if decision.get("gates", {}).get("evidence", {}).get("status") == "pass"
    )
    peer_pass = sum(
        1 for decision in decisions
        if decision.get("gates", {}).get("peer_comparison", {}).get("status") == "pass"
    )
    eligible_for_ranking = sum(
        1 for decision in decisions
        if decision.get("eligible_for_ranking")
    )
    print(f"mandate={state.investment_mandate}")
    print(f"eligible_categories={state.eligible_categories}")
    print(f"discovered={discovered}")
    print(f"eligibility_pass={eligibility_pass}")
    print(f"data_quality_pass={data_quality_pass}")
    print(f"hard_evidence_pass={evidence_pass}")
    print(f"peer_comparable={peer_pass}")
    print(f"eligible_for_ranking={eligible_for_ranking}")
    print(f"ranked={len(state.ranked_funds)}")
    print(f"confidence={state.confidence_score}")
    print(f"soft_evidence_gaps={state.soft_evidence_gaps}")
    print(f"tradeoffs={state.evidence_tradeoffs}")

    if decisions:
        print("\n=== REJECTED CANDIDATES ===")
        for decision in decisions:
            if decision.get("eligible_for_ranking"):
                continue
            failed_gates = [
                gate_name
                for gate_name, gate in decision.get("gates", {}).items()
                if gate.get("status") == "fail"
            ]
            print(
                f"- {decision.get('scheme_code')} | {decision.get('scheme_name')}"
                f" | category={decision.get('category')}"
                f" | failed_gates={failed_gates}"
            )

    print("\n=== DOCUMENTARY PREFLIGHT ===")
    discovered_observation = next(
        (
            observation
            for observation in state.observations
            if observation.action_type.value == "discover_funds"
        ),
        None,
    )
    discovered_result = (
        discovered_observation.result
        if discovered_observation is not None
        and isinstance(discovered_observation.result, dict)
        else {}
    )
    family_keys = [
        str(fund.get("scheme_family_key"))
        for fund in discovered_result.get("funds", [])
        if fund.get("scheme_family_key")
    ]
    documentary_types = (
        "fund_prospectus",
        "fund_fact_sheet",
        "fund_strategy_document",
    )
    documentary_coverage = document_repo.documentary_coverage(
        fund_names=family_keys,
        required_document_types=documentary_types,
    )
    documentary_ratios = [
        float(item.get("coverage_ratio", 0.0))
        for item in documentary_coverage.values()
    ]
    documentary_preflight_ratio = (
        sum(documentary_ratios) / len(documentary_ratios)
        if documentary_ratios else 0.0
    )
    print(f"funds_checked={len(family_keys)}")
    print(f"coverage_ratio={documentary_preflight_ratio}")
    print(
        "fully_covered_funds="
        + str(sum(1 for item in documentary_coverage.values() if item.get("covered")))
    )

    print("\n=== BENCHMARK ===")
    print(f"total_latency_ms={total_latency_ms}")
    assert state.trace is not None
    starts = {}
    action_latencies = {}
    for event in state.trace.events:
        if event.event_type.value == "action_started" and event.action_id:
            starts[event.action_id] = datetime.fromisoformat(event.timestamp)
        elif event.event_type.value == "action_finished" and event.action_id in starts:
            elapsed = (
                datetime.fromisoformat(event.timestamp) - starts[event.action_id]
            ).total_seconds() * 1000
            action_latencies[event.action_type or event.action_id] = int(elapsed)
    for action_type, latency_ms in action_latencies.items():
        print(f"{action_type}_latency_ms={latency_ms}")

    print("\n=== CLOSED BETA GO/NO-GO ===")
    gates = {
        "one_round_or_less": state.investigation_round <= 1,
        "zero_replans": state.replan_count == 0,
        "ranked_results_present": len(state.ranked_funds) > 0,
        "confidence_present": state.confidence_score is not None,
        "hard_evidence_candidates_present": any(
            decision.get("eligible_for_ranking")
            for decision in state.candidate_decisions.values()
        ),
        "no_abstention": state.abstention_reason is None,
        "latency_under_30s_target": total_latency_ms <= 30000,
        "documentary_identity_resolved": (
            documentary_preflight_ratio > 0.0
            or "documentary" in state.soft_evidence_gaps
        ),
    }
    for name, passed in gates.items():
        print(f"{name}={'PASS' if passed else 'FAIL'}")
    print(f"overall={'GO' if all(gates.values()) else 'NO-GO'}")

    print("\n=== OBSERVABILITY SNAPSHOT ===")
    snapshot = ReasoningObservability().snapshot(state)
    print(json.dumps(snapshot.to_dict(), indent=2, default=str))

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
