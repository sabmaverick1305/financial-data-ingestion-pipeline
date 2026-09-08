import json

import pytest

from financial_pipeline.augmentation.generator import GenerationResult
from financial_pipeline.intelligence.evidence import EvidenceDimension
from financial_pipeline.intelligence.llm_planner import LLMPlanner
from financial_pipeline.intelligence.research_plan import ActionType


class FakeGenerator:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0

    def generate(self, messages, intent_type="default", max_tokens=1024):
        self.calls += 1
        return GenerationResult(
            answer=json.dumps(self.payload),
            model="fake-planner",
            provider="fake",
            prompt_tokens=100,
            completion_tokens=100,
            latency_ms=1,
        )


def _flagship_payload() -> dict:
    actions = [
        "discover_categories",
        "discover_funds",
        "fetch_performance",
        "compute_risk",
        "compare_peers",
        "fetch_flows",
        "fetch_aum",
        "retrieve_evidence",
        "check_contradictions",
    ]
    return {
        "objective": "research strong mutual fund candidates for 2026",
        "assumptions": ["research shortlist, not personalized advice"],
        "actions": [
            {
                "action_type": action,
                "metrics": [],
                "rationale": f"need {action}",
                "parameters": {},
            }
            for action in actions
        ],
        "evidence_requirements": [],
    }


def test_llm_planner_parses_only_allowed_actions_and_enforces_evidence_policy() -> None:
    planner = LLMPlanner(FakeGenerator(_flagship_payload()))
    plan, requirements = planner.plan("Give me some of the best mutual funds to invest in 2026")

    assert planner.llm_calls_used == 1
    assert [action.action_type for action in plan.actions] == [
        ActionType.DISCOVER_CATEGORIES,
        ActionType.DISCOVER_FUNDS,
        ActionType.FETCH_PERFORMANCE,
        ActionType.COMPUTE_RISK,
        ActionType.COMPARE_PEERS,
        ActionType.FETCH_FLOWS,
        ActionType.FETCH_AUM,
        ActionType.RETRIEVE_EVIDENCE,
        ActionType.CHECK_CONTRADICTIONS,
    ]
    assert {requirement.dimension for requirement in requirements} == set(EvidenceDimension)


def test_llm_planner_rejects_invented_action() -> None:
    payload = _flagship_payload()
    payload["actions"][0]["action_type"] = "browse_random_website"

    with pytest.raises(ValueError):
        LLMPlanner(FakeGenerator(payload)).plan("Give me some of the best mutual funds to invest in 2026")


def test_llm_planner_rejects_flagship_plan_with_missing_required_action() -> None:
    payload = _flagship_payload()
    payload["actions"] = [
        action for action in payload["actions"]
        if action["action_type"] != "check_contradictions"
    ]

    with pytest.raises(ValueError, match="omitted required flagship actions"):
        LLMPlanner(FakeGenerator(payload)).plan("Give me some of the best mutual funds to invest in 2026")


class TruncatedThenValidGenerator:
    def __init__(self, payload: dict) -> None:
        self.payload = payload
        self.calls = 0
        self.max_tokens_seen: list[int] = []

    def generate(self, messages, intent_type="default", max_tokens=1024):
        self.calls += 1
        self.max_tokens_seen.append(max_tokens)
        answer = '{"objective": "truncated' if self.calls == 1 else json.dumps(self.payload)
        return GenerationResult(
            answer=answer,
            model="fake-planner",
            provider="fake",
            prompt_tokens=100,
            completion_tokens=max_tokens if self.calls == 1 else 100,
            latency_ms=1,
        )


class AlwaysInvalidGenerator:
    def __init__(self) -> None:
        self.calls = 0

    def generate(self, messages, intent_type="default", max_tokens=1024):
        self.calls += 1
        return GenerationResult(
            answer='{"objective": "still truncated',
            model="fake-planner",
            provider="fake",
            prompt_tokens=100,
            completion_tokens=max_tokens,
            latency_ms=1,
        )


def test_llm_planner_retries_once_after_truncated_json() -> None:
    generator = TruncatedThenValidGenerator(_flagship_payload())
    planner = LLMPlanner(generator)

    plan, _ = planner.plan("Give me some of the best mutual funds to invest in 2026")

    assert plan.actions
    assert planner.llm_calls_used == 2
    assert generator.calls == 2
    assert generator.max_tokens_seen == [1800, 2800]


def test_llm_planner_fails_safely_after_two_invalid_json_responses() -> None:
    generator = AlwaysInvalidGenerator()
    planner = LLMPlanner(generator)

    with pytest.raises(ValueError, match="invalid or truncated JSON after retry"):
        planner.plan("Give me some of the best mutual funds to invest in 2026")

    assert planner.llm_calls_used == 2
    assert generator.calls == 2
