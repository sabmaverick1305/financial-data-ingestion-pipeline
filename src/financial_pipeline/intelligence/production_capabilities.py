"""Production capability adapters over existing FIES Postgres and RAG surfaces.

Adapters are dependency-injected: they do not create DB/LLM clients themselves.
This keeps tests deterministic and deployment configuration outside reasoning.
"""
from __future__ import annotations
from typing import Any
from financial_pipeline.intelligence.capability_registry import CapabilityResult
from financial_pipeline.intelligence.research_plan import ResearchAction

class ProductionCapabilityPack:
    def __init__(self, *, performance_repo=None, ingestion_repo=None, rag_pipeline=None) -> None:
        self.performance_repo = performance_repo
        self.ingestion_repo = ingestion_repo
        self.rag_pipeline = rag_pipeline

    def performance(self, action: ResearchAction) -> CapabilityResult:
        if self.performance_repo is None:
            raise RuntimeError("performance repository is not configured")
        scheme_code = action.parameters.get("scheme_code")
        if not scheme_code:
            raise ValueError("scheme_code is required for real performance capability")
        history = self.performance_repo.fetch_history(str(scheme_code))
        from financial_pipeline.mf_performance.calculator import compute_performance
        metrics = compute_performance(str(scheme_code), history)
        if metrics is None:
            raise ValueError(f"no NAV history for scheme {scheme_code}")
        result = {metric: getattr(metrics, metric, None) for metric in action.metrics}
        return CapabilityResult(
            result=result,
            evidence_refs=(f"verified:postgres:mf_nav_history:{scheme_code}",),
        )

    def documentary(self, action: ResearchAction) -> CapabilityResult:
        if self.rag_pipeline is None:
            raise RuntimeError("RAG pipeline is not configured")
        query = action.parameters.get("query") or " ".join((*action.metrics, *action.evidence_types))
        response = self.rag_pipeline.ask(str(query))
        refs = tuple(
            f"verified:rag:{source.get('document_id', source.get('source', index))}"
            for index, source in enumerate(response.sources)
        )
        return CapabilityResult(
            result={"answer": response.answer, "sources": response.sources},
            evidence_refs=refs,
        )

    def not_configured(self, action: ResearchAction) -> CapabilityResult:
        raise RuntimeError(f"real capability not yet configured: {action.action_type.value}")
