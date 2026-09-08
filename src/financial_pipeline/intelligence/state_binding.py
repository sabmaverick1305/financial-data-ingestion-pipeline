"""Bind outputs from earlier actions into downstream action parameters."""
from __future__ import annotations
from dataclasses import replace
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType, ResearchAction

class ActionStateBinder:
    def bind(self, state: ReasoningState, action: ResearchAction) -> ResearchAction:
        params = dict(action.parameters)
        global_categories: list[str] = []
        candidate_categories: list[str] = []
        scheme_codes: list[str] = []
        candidate_names: list[str] = []
        document_search_names: list[str] = []

        for observation in state.observations:
            if observation.status not in (ActionStatus.SUCCEEDED, ActionStatus.PARTIAL):
                continue
            result = observation.result if isinstance(observation.result, dict) else {}
            if observation.action_type is ActionType.DISCOVER_CATEGORIES:
                global_categories.extend(str(v) for v in result.get("categories", []) if v)
            if observation.action_type is ActionType.DISCOVER_FUNDS:
                for fund in result.get("funds", []):
                    if not isinstance(fund, dict):
                        continue
                    code = fund.get("scheme_code")
                    category = fund.get("category")
                    if code:
                        scheme_codes.append(str(code))
                    name = fund.get("scheme_name")
                    if name:
                        candidate_names.append(str(name))
                    family_key = fund.get("scheme_family_key")
                    if family_key:
                        document_search_names.append(str(family_key))
                    elif name:
                        document_search_names.append(str(name))
                    if category:
                        candidate_categories.append(str(category))

        global_categories = list(dict.fromkeys(global_categories))
        candidate_categories = list(dict.fromkeys(candidate_categories))
        scheme_codes = list(dict.fromkeys(scheme_codes))
        candidate_names = list(dict.fromkeys(candidate_names))
        document_search_names = list(dict.fromkeys(document_search_names))

        if action.action_type in (
            ActionType.FETCH_PERFORMANCE,
            ActionType.COMPUTE_RETURNS,
            ActionType.COMPUTE_RISK,
        ):
            if "scheme_code" not in params and "scheme_codes" not in params and scheme_codes:
                params["scheme_codes"] = scheme_codes

        if action.action_type is ActionType.COMPARE_PEERS:
            if "category" not in params and "categories" not in params:
                categories = candidate_categories or global_categories
                if categories:
                    params["categories"] = categories

        if action.action_type in (ActionType.FETCH_AUM, ActionType.FETCH_FLOWS):
            if "category" not in params:
                categories = candidate_categories or global_categories
                if categories:
                    params["category"] = categories[0]

        if action.action_type is ActionType.RETRIEVE_EVIDENCE:
            if candidate_names:
                params["candidate_names"] = candidate_names[:10]
            if document_search_names:
                params["document_search_names"] = document_search_names[:10]
            if "query" not in params and candidate_names:
                requested = " ".join(action.evidence_types)
                names = "; ".join(candidate_names[:5])
                params["query"] = f"{requested} for mutual funds: {names}"

        return replace(action, parameters=params)
