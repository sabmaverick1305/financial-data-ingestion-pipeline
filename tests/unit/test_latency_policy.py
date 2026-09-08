from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction

def test_production_candidate_universe_is_capped_at_twenty():
    class Repo:
        def discover_funds(self, *, category=None, limit=20):
            return [
                {
                    "scheme_code": str(i),
                    "scheme_name": f"Fund {i} Direct Growth",
                    "category": category or "Flexi Cap",
                    "return_3y_cagr": 30 - i,
                    "return_1y": 20 - i,
                }
                for i in range(limit)
            ]

    action = ResearchAction(
        ActionType.DISCOVER_FUNDS,
        parameters={
            "eligible_categories": ["Flexi Cap"],
            "limit": 50,
            "mandate": "core_diversified",
        },
    )
    result = ProductionCapabilityPack(repository=Repo()).discover_funds(action)
    assert len(result.result["funds"]) <= 20
