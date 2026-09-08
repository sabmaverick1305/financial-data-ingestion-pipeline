"""LLM planner that produces only validated FIES ResearchPlan objects."""

from __future__ import annotations

import json
import re
from typing import Any

from financial_pipeline.intelligence.evidence import EvidenceDimension, EvidenceRequirement
from financial_pipeline.intelligence.planner_policy import PlannerPolicy
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan


_SYSTEM_PROMPT = """
You are the planning component of FIES, a financial intelligence reasoning engine.
Your job is ONLY to produce an executable research plan. Do not answer the user.

You may use only these action types:
- discover_categories
- discover_funds
- fetch_performance
- fetch_flows
- fetch_aum
- compute_returns
- compute_risk
- compare_peers
- retrieve_evidence
- check_contradictions

For mutual-fund research:
- compare funds within valid peer categories, not across incompatible categories;
- use performance, risk, peer, flow/AUM, documentary, and contradiction evidence;
- do not invent metrics or tools;
- do not ask for human approval for trusted analytical actions;
- keep each rationale under 12 words;
- include at most 4 metrics per action;
- return compact JSON only;
- do not use markdown or add explanation outside the JSON.

JSON schema:
{
  "objective": "string",
  "assumptions": ["string"],
  "actions": [
    {
      "action_type": "one allowed action type",
      "entity": "string or null",
      "category": "string or null",
      "metrics": ["string"],
      "rationale": "string or null",
      "parameters": {}
    }
  ],
  "evidence_requirements": [
    {
      "dimension": "category|fund_discovery|performance|risk|peer_comparison|flows|aum|documentary|contradiction",
      "allow_partial": false
    }
  ]
}
"""


class LLMPlanner:
    def __init__(self, generator: Any, policy: PlannerPolicy | None = None) -> None:
        self._generator = generator
        self._policy = policy or PlannerPolicy()
        self._llm_calls_used = 0

    @property
    def llm_calls_used(self) -> int:
        return self._llm_calls_used

    def plan(self, query: str) -> tuple[ResearchPlan, tuple[EvidenceRequirement, ...]]:
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": f"User query: {query}"},
        ]
        payload = self._generate_payload(messages)
        plan = self._build_plan(query, payload)

        missing = self._policy.missing_required_actions(query, plan)
        if missing and self._llm_calls_used < 2:
            payload = self._repair_policy_plan(query, messages, missing)
            plan = self._build_plan(query, payload)

        self._policy.validate_plan(query, plan)

        requested_requirements = tuple(
            EvidenceRequirement(
                dimension=EvidenceDimension(item["dimension"]),
                allow_partial=bool(item.get("allow_partial", False)),
            )
            for item in payload.get("evidence_requirements", [])
        )
        requirements = self._policy.enforce_requirements(query, requested_requirements)
        return plan, requirements

    def _generate_payload(self, messages: list[dict]) -> dict[str, Any]:
        """Generate planner JSON with one bounded repair retry on malformed output."""
        attempts: tuple[tuple[list[dict], int], ...] = (
            (messages, 1800),
            (
                [
                    *messages,
                    {
                        "role": "user",
                        "content": (
                            "Return the complete plan again as compact valid JSON only. "
                            "Do not use markdown. Keep rationales under 12 words."
                        ),
                    },
                ],
                2800,
            ),
        )

        last_error: Exception | None = None

        for attempt_messages, max_tokens in attempts:
            result = self._generator.generate(
                attempt_messages,
                intent_type="factual",
                max_tokens=max_tokens,
            )
            self._llm_calls_used += 1

            raw = re.sub(
                r"```(?:json)?\s*|\s*```",
                "",
                result.answer.strip(),
            ).strip()

            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                last_error = exc
                continue

            if not isinstance(payload, dict):
                last_error = ValueError("planner output must be a JSON object")
                continue

            return payload

        raise ValueError(
            "planner returned invalid or truncated JSON after retry"
        ) from last_error

    def _repair_policy_plan(
        self,
        query: str,
        messages: list[dict],
        missing: tuple[ActionType, ...],
    ) -> dict[str, Any]:
        missing_names = ", ".join(action.value for action in missing)
        repair_messages = [
            *messages,
            {
                "role": "user",
                "content": (
                    "The previous plan violated FIES planner policy. "
                    f"It omitted these required actions: {missing_names}. "
                    "Regenerate the complete plan, including every omitted action. "
                    "Return compact valid JSON only; no markdown or explanation."
                ),
            },
        ]
        result = self._generator.generate(
            repair_messages,
            intent_type="factual",
            max_tokens=2800,
        )
        self._llm_calls_used += 1
        raw = re.sub(r"```(?:json)?\s*|\s*```", "", result.answer.strip()).strip()
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("planner policy repair returned invalid JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError("planner policy repair output must be a JSON object")
        return payload

    @staticmethod
    def _build_plan(query: str, payload: dict[str, Any]) -> ResearchPlan:
        parsed_actions = tuple(
            LLMPlanner._parse_action(item)
            for item in payload.get("actions", [])
        )
        actions = LLMPlanner._deduplicate_actions(parsed_actions)
        return ResearchPlan(
            objective=str(payload.get("objective") or query),
            actions=actions,
            assumptions=tuple(str(item) for item in payload.get("assumptions", [])),
        )

    @staticmethod
    def _deduplicate_actions(
        actions: tuple[ResearchAction, ...],
    ) -> tuple[ResearchAction, ...]:
        """Merge duplicate action types while preserving first-seen ordering."""
        merged: dict[ActionType, ResearchAction] = {}
        order: list[ActionType] = []

        for action in actions:
            existing = merged.get(action.action_type)
            if existing is None:
                merged[action.action_type] = action
                order.append(action.action_type)
                continue

            metrics = tuple(dict.fromkeys((*existing.metrics, *action.metrics)))
            parameters = {**existing.parameters, **action.parameters}
            rationale = existing.rationale or action.rationale
            entity = existing.entity or action.entity
            category = existing.category or action.category

            merged[action.action_type] = ResearchAction(
                action_type=action.action_type,
                entity=entity,
                category=category,
                metrics=metrics,
                rationale=rationale,
                parameters=parameters,
                action_id=existing.action_id,
            )

        return tuple(merged[action_type] for action_type in order)

    @staticmethod
    def _parse_action(item: dict[str, Any]) -> ResearchAction:
        return ResearchAction(
            action_type=ActionType(item["action_type"]),
            entity=item.get("entity"),
            category=item.get("category"),
            metrics=tuple(str(metric) for metric in item.get("metrics", [])),
            rationale=item.get("rationale"),
            parameters=dict(item.get("parameters") or {}),
        )
