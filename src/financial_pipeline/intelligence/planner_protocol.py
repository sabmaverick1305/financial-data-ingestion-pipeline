"""Planner contract shared by deterministic and LLM planners."""

from __future__ import annotations

from typing import Protocol

from financial_pipeline.intelligence.evidence import EvidenceRequirement
from financial_pipeline.intelligence.research_plan import ResearchPlan


class ResearchPlanner(Protocol):
    @property
    def llm_calls_used(self) -> int:
        ...

    def plan(self, query: str) -> tuple[ResearchPlan, tuple[EvidenceRequirement, ...]]:
        ...
