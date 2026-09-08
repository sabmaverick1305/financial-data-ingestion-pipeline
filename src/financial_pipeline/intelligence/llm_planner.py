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

        actions = tuple(self._parse_action(item) for item in payload.get("actions", []))
        plan = ResearchPlan(
            objective=str(payload.get("objective") or query),
            actions=actions,
            assumptions=tuple(str(item) for item in payload.get("assumptions", [])),
        )
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
