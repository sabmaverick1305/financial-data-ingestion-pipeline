"""Evidence requirements and deterministic evaluation for FIES reasoning."""

from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType

class EvidenceDimension(StrEnum):
    CATEGORY = "category"
    FUND_DISCOVERY = "fund_discovery"
    PERFORMANCE = "performance"
    RISK = "risk"
    PEER_COMPARISON = "peer_comparison"
    FLOWS = "flows"
    AUM = "aum"
    DOCUMENTARY = "documentary"
    CONTRADICTION = "contradiction"

_DIMENSION_ACTIONS: dict[EvidenceDimension, ActionType] = {
    EvidenceDimension.CATEGORY: ActionType.DISCOVER_CATEGORIES,
    EvidenceDimension.FUND_DISCOVERY: ActionType.DISCOVER_FUNDS,
    EvidenceDimension.PERFORMANCE: ActionType.FETCH_PERFORMANCE,
    EvidenceDimension.RISK: ActionType.COMPUTE_RISK,
    EvidenceDimension.PEER_COMPARISON: ActionType.COMPARE_PEERS,
    EvidenceDimension.FLOWS: ActionType.FETCH_FLOWS,
    EvidenceDimension.AUM: ActionType.FETCH_AUM,
    EvidenceDimension.DOCUMENTARY: ActionType.RETRIEVE_EVIDENCE,
    EvidenceDimension.CONTRADICTION: ActionType.CHECK_CONTRADICTIONS,
}

@dataclass(frozen=True)
class EvidenceRequirement:
    dimension: EvidenceDimension
    required: bool = True
    allow_partial: bool = False

@dataclass(frozen=True)
class EvidenceEvaluation:
    is_sufficient: bool
    satisfied: tuple[EvidenceDimension, ...]
    missing: tuple[EvidenceDimension, ...]
    failed: tuple[EvidenceDimension, ...]
    partial: tuple[EvidenceDimension, ...]
    tradeoffs: tuple[str, ...]

class EvidenceEvaluator:
    def evaluate(self, state: ReasoningState, requirements: tuple[EvidenceRequirement, ...]) -> EvidenceEvaluation:
        latest = {}
        for observation in state.observations:
            latest[observation.action_type] = observation

        satisfied = []
        missing = []
        failed = []
        partial = []
        tradeoffs = []

        for requirement in requirements:
            if not requirement.required:
                continue
            action_type = _DIMENSION_ACTIONS[requirement.dimension]
            observation = latest.get(action_type)
            if observation is None:
                missing.append(requirement.dimension)
            elif observation.status is ActionStatus.SUCCEEDED:
                satisfied.append(requirement.dimension)
            elif observation.status is ActionStatus.PARTIAL:
                partial.append(requirement.dimension)
                if observation.tradeoff_reason:
                    tradeoffs.append(observation.tradeoff_reason)
                if requirement.allow_partial:
                    satisfied.append(requirement.dimension)
                else:
                    failed.append(requirement.dimension)
            elif observation.status is ActionStatus.FAILED:
                failed.append(requirement.dimension)
            else:
                missing.append(requirement.dimension)

        return EvidenceEvaluation(
            is_sufficient=not missing and not failed,
            satisfied=tuple(satisfied),
            missing=tuple(missing),
            failed=tuple(failed),
            partial=tuple(partial),
            tradeoffs=tuple(tradeoffs),
        )

    @staticmethod
    def action_for_dimension(dimension: EvidenceDimension) -> ActionType:
        return _DIMENSION_ACTIONS[dimension]
