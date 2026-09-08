import pytest

from financial_pipeline.documentary.authoritative_ingestion import AuthoritativeFundDocument
from financial_pipeline.documentary.evidence_policy import (
    DEFAULT_BETA_REQUIREMENTS,
    semantic_requirements_for_evidence_types,
)
from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType, ResearchAction
from financial_pipeline.retrieval.rag import RAGResponse


def test_planner_document_labels_map_to_semantic_requirements():
    requirements = semantic_requirements_for_evidence_types(
        ("prospectus", "fact_sheet", "strategy", "disclosures")
    )
    assert requirements == (
        "scheme_mandate",
        "portfolio_composition",
        "investment_strategy",
    )
    assert set(requirements) == set(DEFAULT_BETA_REQUIREMENTS)


def test_authoritative_document_contract_requires_domain_field():
    document = AuthoritativeFundDocument(
        scheme_family_key="alpha midcap fund",
        scheme_code="123",
        provider="Alpha Mutual Fund",
        source="amc_official",
        source_url="https://example.com/alpha.pdf",
        authoritative_domain="example.com",
        document_type="scheme_information_document",
        file_name="alpha-sid.pdf",
    )
    assert document.authoritative_domain == "example.com"


def test_semantic_documentary_result_uses_verified_resolved_ids():
    class Rag:
        def semantic_documentary_coverage(self, **kwargs):
            return {
                "alpha midcap fund": {
                    "covered": True,
                    "coverage_ratio": 1.0,
                    "missing_requirements": [],
                    "by_requirement": {
                        "scheme_mandate": {
                            "document_ids": ["sid-1"],
                        },
                        "investment_strategy": {
                            "document_ids": ["sid-1"],
                        },
                        "portfolio_composition": {
                            "document_ids": ["factsheet-1"],
                        },
                    },
                }
            }

        def ask_documentary_ids(self, query, *, document_ids):
            assert set(document_ids) == {"sid-1", "factsheet-1"}
            return RAGResponse(
                query=query,
                answer="verified evidence",
                sources=[
                    {
                        "document_id": "sid-1",
                        "document_type": "scheme_information_document",
                    }
                ],
                retrieval_count=1,
            )

    action = ResearchAction(
        ActionType.RETRIEVE_EVIDENCE,
        evidence_types=("prospectus", "fact_sheet", "strategy", "disclosures"),
        parameters={
            "candidate_names": ["Alpha Midcap Fund Direct Growth"],
            "document_search_names": ["alpha midcap fund"],
            "query": "scheme mandate strategy portfolio evidence",
        },
    )
    result = ProductionCapabilityPack(
        repository=object(),
        rag_pipeline=Rag(),
    ).documentary(action)

    assert result.result["documentary_coverage_ratio"] == 1.0
    assert result.result["semantic_requirements"] == [
        "scheme_mandate",
        "portfolio_composition",
        "investment_strategy",
    ]
    assert result.result["ingestion_backlog"] == []
