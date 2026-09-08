"""Bind outputs from earlier actions into downstream action parameters."""
from __future__ import annotations
from dataclasses import replace
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType, ResearchAction

class ActionStateBinder:
    def bind(self, state: ReasoningState, action: ResearchAction) -> ResearchAction:
        params = dict(action.parameters)
        categories: list[str] = []
        scheme_codes: list[str] = []

        for observation in state.observations:
            if observation.status not in (ActionStatus.SUCCEEDED, ActionStatus.PARTIAL):
                continue
            result = observation.result if isinstance(observation.result, dict) else {}
            if observation.action_type is ActionType.DISCOVER_CATEGORIES:
                categories.extend(str(v) for v in result.get("categories", []) if v)
            if observation.action_type is ActionType.DISCOVER_FUNDS:
                for fund in result.get("funds", []):
                    if not isinstance(fund, dict):
                        continue
                    code = fund.get("scheme_code")
                    category = fund.get("category")
                    if code:
                        scheme_codes.append(str(code))
                    if category:
                        categories.append(str(category))

        categories = list(dict.fromkeys(categories))
        scheme_codes = list(dict.fromkeys(scheme_codes))

        if action.action_type in (ActionType.FETCH_PERFORMANCE, ActionType.COMPUTE_RISK):
            if "scheme_code" not in params and "scheme_codes" not in params and scheme_codes:
                params["scheme_codes"] = scheme_codes

        if action.action_type is ActionType.COMPARE_PEERS:
            if "category" not in params and "categories" not in params and categories:
                params["categories"] = categories

        if action.action_type in (ActionType.FETCH_AUM, ActionType.FETCH_FLOWS):
            if "category" not in params and categories:
                params["category"] = categories[0]

        return replace(action, parameters=params)
