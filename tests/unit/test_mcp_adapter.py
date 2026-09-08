from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.mcp_adapter import (
    MCPActionBinding,
    MCPCapabilityAdapter,
    MCPToolResult,
)
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType, ResearchAction


class FakeInvoker:
    def __init__(self, result: MCPToolResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict]] = []

    def call_tool(self, tool_name: str, arguments: dict) -> MCPToolResult:
        self.calls.append((tool_name, arguments))
        return self.result


def test_mcp_adapter_maps_action_to_tool_call_and_preserves_evidence_refs() -> None:
    invoker = FakeInvoker(
        MCPToolResult(
            structured_content={"funds": ["Fund A", "Fund B"]},
            evidence_refs=("amfi:doc-1",),
        )
    )
    registry = CapabilityRegistry()
    adapter = MCPCapabilityAdapter(invoker)
    adapter.register(
        registry,
        MCPActionBinding(
            action_type=ActionType.DISCOVER_FUNDS,
            tool_name="get_funds_by_category",
        ),
    )

    action = ResearchAction(
        action_type=ActionType.DISCOVER_FUNDS,
        category="Mid Cap Fund",
        metrics=("return_1y", "return_3y_cagr"),
        parameters={"as_of_year": 2026},
    )
    observation = registry.execute(action)

    assert observation.status is ActionStatus.SUCCEEDED
    assert observation.result == {"funds": ["Fund A", "Fund B"]}
    assert observation.evidence_refs == ("amfi:doc-1",)
    assert invoker.calls == [(
        "get_funds_by_category",
        {
            "as_of_year": 2026,
            "category": "Mid Cap Fund",
            "metrics": ["return_1y", "return_3y_cagr"],
        },
    )]


def test_mcp_adapter_turns_tool_error_into_failed_observation() -> None:
    invoker = FakeInvoker(MCPToolResult(is_error=True, error="upstream unavailable"))
    registry = CapabilityRegistry()
    adapter = MCPCapabilityAdapter(invoker)
    adapter.register(
        registry,
        MCPActionBinding(
            action_type=ActionType.RETRIEVE_EVIDENCE,
            tool_name="search_verified_research",
        ),
    )

    observation = registry.execute(ResearchAction(action_type=ActionType.RETRIEVE_EVIDENCE))

    assert observation.status is ActionStatus.FAILED
    assert observation.error == "upstream unavailable"


def test_untrusted_mcp_binding_remains_visible_to_harness() -> None:
    invoker = FakeInvoker(MCPToolResult(structured_content={}))
    registry = CapabilityRegistry()
    MCPCapabilityAdapter(invoker).register(
        registry,
        MCPActionBinding(
            action_type=ActionType.RETRIEVE_EVIDENCE,
            tool_name="external_search",
            trusted=False,
        ),
    )

    assert not registry.get(ActionType.RETRIEVE_EVIDENCE).trusted
