from financial_pipeline.intelligence.investment_eligibility import InvestmentEligibilityPolicy
from financial_pipeline.intelligence.investment_mandate import InvestmentMandate, InvestmentMandatePolicy
from financial_pipeline.intelligence.production_capabilities import ProductionCapabilityPack
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan

def test_unqualified_best_funds_defaults_to_core_diversified():
    decision = InvestmentMandatePolicy().decide(
        "Give me some of the best mutual funds to invest in 2026"
    )
    assert decision.mandate is InvestmentMandate.CORE_DIVERSIFIED
    assert "Flexi Cap" in decision.eligible_categories
    assert "Mid Cap" in decision.eligible_categories
    assert "Small Cap" in decision.eligible_categories
    assert "Silver ETF" not in decision.eligible_categories
    assert "Sectoral/Thematic" not in decision.eligible_categories

def test_explicit_small_cap_query_overrides_default_universe():
    decision = InvestmentMandatePolicy().decide(
        "What are the best small cap mutual funds?"
    )
    assert decision.eligible_categories == ("Small Cap",)

def test_explicit_commodity_query_allows_requested_category():
    decision = InvestmentMandatePolicy().decide(
        "Compare the best silver mutual funds"
    )
    assert decision.mandate is InvestmentMandate.COMMODITY
    assert decision.eligible_categories == ("Silver ETF",)

def test_policy_removes_planner_specialized_category_for_generic_query():
    plan = ResearchPlan(
        objective="rank",
        actions=(
            ResearchAction(
                ActionType.DISCOVER_FUNDS,
                category="Sectoral/Thematic",
                parameters={"category": "Sectoral/Thematic", "limit": 10},
            ),
        ),
    )
    updated, decision = InvestmentEligibilityPolicy().apply(
        "Give me some of the best mutual funds to invest in 2026",
        plan,
    )
    action = updated.actions[0]
    assert action.category is None
    assert "category" not in action.parameters
    assert "Sectoral/Thematic" not in action.parameters["eligible_categories"]
    assert action.parameters["mandate"] == "core_diversified"

def test_production_discovery_balances_across_eligible_categories():
    class Repo:
        def discover_funds(self, *, category=None, limit=20):
            return [{
                "scheme_code": f"{category}-1",
                "scheme_name": f"{category} Fund",
                "category": category,
                "return_3y_cagr": 10.0,
                "return_1y": 5.0,
            }]
    action = ResearchAction(
        ActionType.DISCOVER_FUNDS,
        parameters={
            "eligible_categories": ["Large Cap", "Flexi Cap", "Mid Cap"],
            "mandate": "core_diversified",
            "limit": 10,
        },
    )
    result = ProductionCapabilityPack(repository=Repo()).discover_funds(action)
    categories = {row["category"] for row in result.result["funds"]}
    assert categories == {"Large Cap", "Flexi Cap", "Mid Cap"}
    assert result.result["mandate"] == "core_diversified"
