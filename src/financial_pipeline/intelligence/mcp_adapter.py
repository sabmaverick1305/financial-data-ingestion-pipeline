"""MCP-backed capability adapter for the FIES reasoning engine.

The reasoning domain remains transport-agnostic. A concrete MCP SDK client
only needs to implement MCPToolInvoker.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from financial_pipeline.intelligence.capability_registry import (
    CapabilityRegistry,
    CapabilityResult,
)
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction


@dataclass(frozen=True)
class MCPToolDescriptor:
    name: str
    description: str | None = None
    input_schema: dict[str, Any] | None = None


@dataclass(frozen=True)
class MCPToolResult:
    structured_content: Any = None
    evidence_refs: tuple[str, ...] = ()
    is_error: bool = False
    error: str | None = None


class MCPToolInvoker(Protocol):
    """Minimal port required by FIES to call an MCP tool."""

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> MCPToolResult:
        ...


@dataclass(frozen=True)
class MCPActionBinding:
    action_type: ActionType
    tool_name: str
    trusted: bool = True


class MCPArgumentBuilder:
    """Translate FIES action fields into a generic MCP tool argument object."""

    @staticmethod
    def build(action: ResearchAction) -> dict[str, Any]:
        arguments = dict(action.parameters)
        if action.entity is not None:
            arguments.setdefault("entity", action.entity)
        if action.category is not None:
            arguments.setdefault("category", action.category)
        if action.metrics:
            arguments.setdefault("metrics", list(action.metrics))
        return arguments


class MCPCapabilityAdapter:
    def __init__(self, invoker: MCPToolInvoker) -> None:
        self._invoker = invoker

    def handler_for(self, binding: MCPActionBinding):
        def handler(action: ResearchAction) -> CapabilityResult:
            arguments = MCPArgumentBuilder.build(action)
            result = self._invoker.call_tool(binding.tool_name, arguments)
            if result.is_error:
                raise RuntimeError(result.error or f"MCP tool failed: {binding.tool_name}")
            return CapabilityResult(
                result=result.structured_content,
                evidence_refs=result.evidence_refs,
            )

        return handler

    def register(
        self,
        registry: CapabilityRegistry,
        binding: MCPActionBinding,
    ) -> None:
        registry.register(
            binding.action_type,
            self.handler_for(binding),
            trusted=binding.trusted,
        )
