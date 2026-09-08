from datetime import date

from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction

def test_risk_and_returns_reuse_one_nav_snapshot():
    class Repo:
        def __init__(self):
            self.calls = 0
        def nav_history_many(self, codes):
            self.calls += 1
            return {
                code: [
                    (date(2023, 9, 8), 100.0),
                    (date(2024, 9, 8), 110.0),
                    (date(2025, 9, 8), 120.0),
                    (date(2026, 9, 7), 130.0),
                ]
                for code in codes
            }

    repo = Repo()
    pack = ProductionCapabilityPack(repository=repo)

    risk_action = ResearchAction(
        ActionType.COMPUTE_RISK,
        metrics=("volatility", "sharpe_ratio", "max_drawdown"),
        parameters={"scheme_codes": ["1", "2"]},
    )
    returns_action = ResearchAction(
        ActionType.COMPUTE_RETURNS,
        metrics=("annualized_return",),
        parameters={"scheme_codes": ["1", "2"]},
    )

    pack.risk(risk_action)
    pack.returns(returns_action)

    assert repo.calls == 1

def test_peer_comparison_deduplicates_share_classes_before_percentile():
    class Repo:
        def peer_performance_many(self, *, categories, limit_per_category):
            return {
                "Small Cap": [
                    {
                        "scheme_code": "1",
                        "scheme_name": "Alpha Small Cap Fund - Direct Growth",
                        "return_3y_cagr": 20.0,
                    },
                    {
                        "scheme_code": "2",
                        "scheme_name": "Alpha Small Cap Fund - Regular Growth",
                        "return_3y_cagr": 19.0,
                    },
                    {
                        "scheme_code": "3",
                        "scheme_name": "Beta Small Cap Fund - Direct Growth",
                        "return_3y_cagr": 18.0,
                    },
                ]
            }

    action = ResearchAction(
        ActionType.COMPARE_PEERS,
        metrics=("percentile_rank", "peer_outperformance"),
        parameters={
            "categories": ["Small Cap"],
            "rank_by": "return_3y_cagr",
        },
    )
    result = ProductionCapabilityPack(repository=Repo()).peer_compare(action)
    peers = result.result["categories"][0]["peers"]

    assert len(peers) == 2
    assert [peer["scheme_code"] for peer in peers] == ["1", "3"]
    assert peers[0]["percentile_rank"] == 100.0
    assert peers[1]["percentile_rank"] == 0.0
