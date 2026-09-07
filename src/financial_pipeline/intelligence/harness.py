"""Bounded execution policy for autonomous FIES reasoning."""

from __future__ import annotations

from dataclasses import dataclass

from financial_pipeline.intelligence.capability_registry import CapabilityRegistry
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ResearchAction


@dataclass(frozen=True)
class HarnessLimits:
    max_investigation_rounds: int = 3
    max_tool_calls: int = 20
    max_llm_calls: int = 4
    max_replans: int = 2


class ReasoningHarness:
    def __init__(self, registry: CapabilityRegistry, limits: HarnessLimits | None = None) -> None:
        self._registry = registry
        self._limits = limits or HarnessLimits()

    def validate_action(self, state: ReasoningState, action: ResearchAction) -> None:
        capability = self._registry.get(action.action_type)
        if not capability.trusted:
            raise PermissionError(f"untrusted capability requires approval: {action.action_type}")
        if state.tool_calls >= self._limits.max_tool_calls:
            raise RuntimeError("tool-call budget exhausted")

    def can_replan(self, state: ReasoningState) -> bool:
        return state.replan_count < self._limits.max_replans

    def can_investigate(self, state: ReasoningState) -> bool:
        return state.investigation_round < self._limits.max_investigation_rounds

    def can_call_llm(self, state: ReasoningState) -> bool:
        return state.llm_calls < self._limits.max_llm_calls
