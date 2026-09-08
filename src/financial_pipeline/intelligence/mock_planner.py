"""Deterministic mock planner used to validate FIES reasoning before an LLM planner."""

from __future__ import annotations

from financial_pipeline.intelligence.evidence import EvidenceDimension, EvidenceRequirement
from financial_pipeline.intelligence.research_plan import ActionType, ResearchAction, ResearchPlan


class MockPlanner:
    """Rule-based planner for representative mutual-fund research queries."""

    @property
    def llm_calls_used(self) -> int:
        return 0

    def plan(self, query: str) -> tuple[ResearchPlan, tuple[EvidenceRequirement, ...]]:
        q = query.lower()

        if "best mutual fund" in q or "best mutual funds" in q:
            return self._best_funds_plan(query)

        if "mid cap" in q and ("compare" in q or "best" in q):
            return self._category_comparison_plan(query, "Mid Cap Fund")

        if "large cap" in q and ("compare" in q or "best" in q):
            return self._category_comparison_plan(query, "Large Cap Fund")

        if "why" in q and ("inflow" in q or "outflow" in q):
            return self._flow_investigation_plan(query)

        raise ValueError(f"mock planner does not support query: {query}")

    def _best_funds_plan(self, query: str):
        plan = ResearchPlan(
            objective=query,
            actions=(
                ResearchAction(ActionType.DISCOVER_CATEGORIES, rationale="establish comparable fund categories"),
                ResearchAction(ActionType.DISCOVER_FUNDS, rationale="identify eligible funds within categories"),
                ResearchAction(
                    ActionType.FETCH_PERFORMANCE,
                    metrics=("return_1y", "return_3y_cagr", "return_5y_cagr"),
                    rationale="compare medium- and long-horizon performance",
                ),
                ResearchAction(
                    ActionType.COMPUTE_RISK,
                    metrics=("rolling_volatility", "rolling_stddev"),
                    rationale="avoid ranking on returns alone",
                ),
                ResearchAction(ActionType.COMPARE_PEERS, rationale="compare funds only within valid peer groups"),
                ResearchAction(ActionType.FETCH_FLOWS, metrics=("net_inflow",)),
                ResearchAction(ActionType.FETCH_AUM, metrics=("aum",)),
                ResearchAction(ActionType.RETRIEVE_EVIDENCE, rationale="add verified documentary evidence"),
                ResearchAction(ActionType.CHECK_CONTRADICTIONS, rationale="challenge top-ranked candidates"),
            ),
            assumptions=("No personalized risk profile supplied; produce category-separated research shortlist.",),
        )
        requirements = tuple(EvidenceRequirement(d) for d in (
            EvidenceDimension.CATEGORY,
            EvidenceDimension.FUND_DISCOVERY,
            EvidenceDimension.PERFORMANCE,
            EvidenceDimension.RISK,
            EvidenceDimension.PEER_COMPARISON,
            EvidenceDimension.FLOWS,
            EvidenceDimension.AUM,
            EvidenceDimension.DOCUMENTARY,
            EvidenceDimension.CONTRADICTION,
        ))
        return plan, requirements

    def _category_comparison_plan(self, query: str, category: str):
        plan = ResearchPlan(
            objective=query,
            actions=(
                ResearchAction(ActionType.DISCOVER_FUNDS, category=category),
                ResearchAction(
                    ActionType.FETCH_PERFORMANCE,
                    category=category,
                    metrics=("return_1y", "return_3y_cagr", "return_5y_cagr"),
                ),
                ResearchAction(ActionType.COMPUTE_RISK, category=category),
                ResearchAction(ActionType.COMPARE_PEERS, category=category),
                ResearchAction(ActionType.CHECK_CONTRADICTIONS, category=category),
            ),
        )
        requirements = tuple(EvidenceRequirement(d) for d in (
            EvidenceDimension.FUND_DISCOVERY,
            EvidenceDimension.PERFORMANCE,
            EvidenceDimension.RISK,
            EvidenceDimension.PEER_COMPARISON,
            EvidenceDimension.CONTRADICTION,
        ))
        return plan, requirements

    def _flow_investigation_plan(self, query: str):
        plan = ResearchPlan(
            objective=query,
            actions=(
                ResearchAction(ActionType.FETCH_FLOWS),
                ResearchAction(ActionType.FETCH_AUM),
                ResearchAction(ActionType.RETRIEVE_EVIDENCE),
                ResearchAction(ActionType.CHECK_CONTRADICTIONS),
            ),
        )
        requirements = tuple(EvidenceRequirement(d) for d in (
            EvidenceDimension.FLOWS,
            EvidenceDimension.AUM,
            EvidenceDimension.DOCUMENTARY,
            EvidenceDimension.CONTRADICTION,
        ))
        return plan, requirements
