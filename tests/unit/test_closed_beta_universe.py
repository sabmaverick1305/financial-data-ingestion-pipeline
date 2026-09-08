from financial_pipeline.intelligence.closed_beta_policy import (
    CLOSED_BETA_SCHEME_FAMILIES,
    ClosedBetaUniversePolicy,
)
from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction


def test_closed_beta_policy_contains_exactly_six_fund_families():
    assert len(CLOSED_BETA_SCHEME_FAMILIES) == 6
    assert set(CLOSED_BETA_SCHEME_FAMILIES) == {
        "bandhan small cap fund",
        "icici prudential large and mid cap fund",
        "invesco india large and mid cap fund",
        "invesco india midcap fund",
        "motilal oswal large and midcap fund",
        "motilal oswal midcap fund",
    }


def test_discovery_filters_non_beta_fund_families():
    class Repo:
        def discover_funds_many(self, *, categories, limit):
            return [
                {
                    "scheme_code": "1",
                    "scheme_name": "Bandhan Small Cap Fund - Direct Growth",
                    "category": "Small Cap",
                    "return_3y_cagr": 20.0,
                    "return_1y": 15.0,
                },
                {
                    "scheme_code": "2",
                    "scheme_name": "HSBC Midcap Fund - Direct Growth",
                    "category": "Mid Cap",
                    "return_3y_cagr": 21.0,
                    "return_1y": 16.0,
                },
                {
                    "scheme_code": "3",
                    "scheme_name": "Invesco India Midcap Fund - Direct Growth",
                    "category": "Mid Cap",
                    "return_3y_cagr": 19.0,
                    "return_1y": 14.0,
                },
            ]

    action = ResearchAction(
        ActionType.DISCOVER_FUNDS,
        parameters={
            "eligible_categories": ["Small Cap", "Mid Cap"],
            "limit": 20,
            "mandate": "core_diversified",
        },
    )

    result = ProductionCapabilityPack(
        repository=Repo(),
        beta_universe_policy=ClosedBetaUniversePolicy.default(),
    ).discover_funds(action)

    families = {
        row["scheme_family_key"]
        for row in result.result["funds"]
    }
    assert families == {
        "bandhan small cap fund",
        "invesco india midcap fund",
    }
    assert result.result["beta_universe_enabled"] is True
