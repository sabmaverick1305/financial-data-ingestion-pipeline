"""Apply deterministic investment-mandate eligibility to a research plan."""
from __future__ import annotations
from dataclasses import replace

from financial_pipeline.intelligence.investment_mandate import InvestmentMandatePolicy, MandateDecision
from financial_pipeline.intelligence.research_plan import ActionType, ResearchPlan

class InvestmentEligibilityPolicy:
    def __init__(self) -> None:
        self._mandates = InvestmentMandatePolicy()

    def apply(self, query: str, plan: ResearchPlan) -> tuple[ResearchPlan, MandateDecision]:
        decision = self._mandates.decide(query)
        actions = []
        for action in plan.actions:
            params = dict(action.parameters)
            if action.action_type is ActionType.DISCOVER_FUNDS:
                if not action.category and "category" not in params:
                    params["eligible_categories"] = list(decision.eligible_categories)
                params["mandate"] = decision.mandate.value
                params["mandate_rationale"] = decision.rationale
            if action.action_type is ActionType.DISCOVER_CATEGORIES:
                params["eligible_categories"] = list(decision.eligible_categories)
                params["mandate"] = decision.mandate.value
            actions.append(replace(action, parameters=params))

        return (
            ResearchPlan(
                objective=plan.objective,
                actions=tuple(actions),
                assumptions=tuple(dict.fromkeys((*plan.assumptions, decision.rationale))),
                plan_id=plan.plan_id,
            ),
            decision,
        )
