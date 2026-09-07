"""Executable research-plan contracts for the FIES reasoning engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class ActionType(StrEnum):
    DISCOVER_CATEGORIES = "discover_categories"
    DISCOVER_FUNDS = "discover_funds"
    FETCH_PERFORMANCE = "fetch_performance"
    FETCH_FLOWS = "fetch_flows"
    FETCH_AUM = "fetch_aum"
    COMPUTE_RETURNS = "compute_returns"
    COMPUTE_RISK = "compute_risk"
    COMPARE_PEERS = "compare_peers"
    RETRIEVE_EVIDENCE = "retrieve_evidence"
    CHECK_CONTRADICTIONS = "check_contradictions"


class ActionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class ResearchAction:
    action_type: ActionType
    entity: str | None = None
    category: str | None = None
    metrics: tuple[str, ...] = ()
    rationale: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    action_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class ResearchPlan:
    objective: str
    actions: tuple[ResearchAction, ...]
    assumptions: tuple[str, ...] = ()
    plan_id: str = field(default_factory=lambda: uuid4().hex)


@dataclass(frozen=True)
class ActionObservation:
    action_id: str
    action_type: ActionType
    status: ActionStatus
    result: Any = None
    evidence_refs: tuple[str, ...] = ()
    error: str | None = None
