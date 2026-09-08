from datetime import date
import pytest
from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction

class Repo:
    def discover_funds(self, **kwargs): return [{"scheme_code":"1","scheme_name":"Alpha","category":"Mid Cap"}]
    def performance(self, code): return {"return_1y":12.0,"return_3y_cagr":15.0}
    def nav_history(self, code): return [(date(2025,1,1),100.0),(date(2025,1,2),101.0),(date(2025,1,3),99.0),(date(2025,1,4),103.0)]
    def peer_performance(self, **kwargs): return [{"scheme_code":"1","return_3y_cagr":15.0},{"scheme_code":"2","return_3y_cagr":10.0}]
    def latest_category_facts(self, *, metric, category=None): return [{"fund_category":category or "Mid Cap","value":100,"source_document_id":"doc1"}]

def test_real_fund_discovery():
    result=ProductionCapabilityPack(repository=Repo()).discover_funds(ResearchAction(ActionType.DISCOVER_FUNDS, parameters={"category":"Mid Cap"}))
    assert result.result["scope"]=="scheme"
    assert result.result["funds"][0]["scheme_code"]=="1"

def test_real_performance():
    action=ResearchAction(ActionType.FETCH_PERFORMANCE, metrics=("return_1y","return_3y_cagr"), parameters={"scheme_code":"1"})
    result=ProductionCapabilityPack(repository=Repo()).performance(action)
    assert result.result["metrics"]["return_3y_cagr"]==15.0

def test_real_risk_is_deterministic():
    action=ResearchAction(ActionType.COMPUTE_RISK, metrics=("volatility","sharpe_ratio","max_drawdown"), parameters={"scheme_code":"1"})
    metrics=ProductionCapabilityPack(repository=Repo()).risk(action).result["metrics"]
    assert metrics["volatility"]>0
    assert metrics["max_drawdown"]>0

def test_peer_rank_is_computed_from_real_peer_values():
    action=ResearchAction(ActionType.COMPARE_PEERS, metrics=("percentile_rank",), parameters={"category":"Mid Cap"})
    peers=ProductionCapabilityPack(repository=Repo()).peer_compare(action).result["peers"]
    assert peers[0]["rank"]==1
    assert peers[0]["percentile_rank"]==100.0

def test_aum_and_flows_are_explicitly_category_scoped():
    pack=ProductionCapabilityPack(repository=Repo())
    assert pack.aum(ResearchAction(ActionType.FETCH_AUM, parameters={"category":"Mid Cap"})).result["scope"]=="fund_category"
    assert pack.flows(ResearchAction(ActionType.FETCH_FLOWS, parameters={"category":"Mid Cap"})).result["scope"]=="fund_category"

def test_scheme_level_aum_fails_closed():
    with pytest.raises(ValueError, match="scheme-level AUM"):
        ProductionCapabilityPack(repository=Repo()).aum(ResearchAction(ActionType.FETCH_AUM, parameters={"scheme_code":"1"}))
