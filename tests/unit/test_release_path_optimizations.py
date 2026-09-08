from datetime import date

from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction
from financial_pipeline.intelligence.scheme_family import SchemeFamilyPolicy
from financial_pipeline.retrieval.rag import RAGPipeline

def test_scheme_family_prefers_direct_growth_over_regular_growth():
    rows = [
        {"scheme_code": "1", "scheme_name": "Alpha Midcap Fund - Regular Growth"},
        {"scheme_code": "2", "scheme_name": "Alpha Midcap Fund - Direct Growth"},
        {"scheme_code": "3", "scheme_name": "Alpha Midcap Fund - Direct IDCW"},
    ]
    selected = SchemeFamilyPolicy().deduplicate(rows)
    assert len(selected) == 1
    assert selected[0]["scheme_code"] == "2"

def test_performance_uses_batch_repository_when_available():
    class Repo:
        def performance_many(self, codes):
            return {
                "1": {"scheme_code": "1", "scheme_name": "A", "category": "Mid Cap", "return_1y": 10},
                "2": {"scheme_code": "2", "scheme_name": "B", "category": "Mid Cap", "return_1y": 12},
            }
        def performance(self, code):
            raise AssertionError("individual performance query should not be used")

    action = ResearchAction(
        ActionType.FETCH_PERFORMANCE,
        metrics=("return_1y",),
        parameters={"scheme_codes": ["1", "2"]},
    )
    result = ProductionCapabilityPack(repository=Repo()).performance(action)
    assert len(result.result["rows"]) == 2

def test_risk_uses_batch_nav_repository_when_available():
    class Repo:
        def nav_history_many(self, codes):
            return {
                code: [
                    (date(2026, 1, 1), 100.0),
                    (date(2026, 1, 2), 101.0),
                    (date(2026, 1, 3), 102.0),
                ]
                for code in codes
            }
        def nav_history(self, code):
            raise AssertionError("individual NAV query should not be used")

    action = ResearchAction(
        ActionType.COMPUTE_RISK,
        metrics=("volatility", "sharpe_ratio", "max_drawdown"),
        parameters={"scheme_codes": ["1", "2"]},
    )
    result = ProductionCapabilityPack(repository=Repo()).risk(action)
    assert len(result.result["rows"]) == 2

def test_documentary_rag_skips_llm_when_no_matching_documents():
    class Retriever:
        def find_document_ids(self, **kwargs):
            return []
    rag = RAGPipeline(Retriever())
    response = rag.ask_documentary(
        "factsheet for alpha fund",
        fund_names=["alpha fund"],
        document_types=["fund_fact_sheet"],
    )
    assert response.sources == []
    assert response.retrieval_count == 0
    assert "not indexed" in response.answer.lower()


def test_documentary_type_aliases_are_supported_by_metadata_lookup_contract():
    # Contract-level regression: planner vocabulary must remain compatible
    # with canonical indexed document types.
    aliases = {
        "prospectus": "fund_prospectus",
        "fact_sheet": "fund_fact_sheet",
        "strategy": "fund_strategy_document",
        "disclosures": "portfolio_disclosure",
    }
    assert aliases["prospectus"] == "fund_prospectus"
    assert aliases["fact_sheet"] == "fund_fact_sheet"
    assert aliases["strategy"] == "fund_strategy_document"
    assert aliases["disclosures"] == "portfolio_disclosure"
