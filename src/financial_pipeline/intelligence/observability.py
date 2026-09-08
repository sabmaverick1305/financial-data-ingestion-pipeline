"""Production observability snapshot for FIES reasoning requests."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime

from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType

@dataclass(frozen=True)
class ReasoningObservabilitySnapshot:
    investigation_rounds: int
    replans: int
    tool_calls: int
    llm_calls: int
    first_pass_success: bool
    finalized: bool
    abstained: bool
    confidence_score: float | None
    candidate_count: int
    eligible_for_ranking: int
    candidate_quality_ratio: float
    soft_evidence_gaps: tuple[str, ...]
    documentary_coverage_ratio: float
    documentary_backlog_count: int
    total_action_latency_ms: int
    action_latency_ms: dict[str, int]
    failed_actions: tuple[str, ...]
    partial_actions: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)

class ReasoningObservability:
    def snapshot(self, state: ReasoningState) -> ReasoningObservabilitySnapshot:
        action_latency_ms = self._action_latencies(state)
        decisions = list(state.candidate_decisions.values())
        eligible = sum(1 for item in decisions if item.get("eligible_for_ranking"))

        latest_documentary = next(
            (
                observation
                for observation in reversed(state.observations)
                if observation.action_type is ActionType.RETRIEVE_EVIDENCE
            ),
            None,
        )
        documentary_result = (
            latest_documentary.result
            if latest_documentary is not None
            and isinstance(latest_documentary.result, dict)
            else {}
        )

        failed = tuple(
            observation.action_type.value
            for observation in state.observations
            if observation.status is ActionStatus.FAILED
        )
        partial = tuple(
            observation.action_type.value
            for observation in state.observations
            if observation.status is ActionStatus.PARTIAL
        )

        return ReasoningObservabilitySnapshot(
            investigation_rounds=state.investigation_round,
            replans=state.replan_count,
            tool_calls=state.tool_calls,
            llm_calls=state.llm_calls,
            first_pass_success=state.investigation_round <= 1 and state.replan_count == 0,
            finalized=state.final_answer is not None and state.abstention_reason is None,
            abstained=state.abstention_reason is not None,
            confidence_score=state.confidence_score,
            candidate_count=len(decisions),
            eligible_for_ranking=eligible,
            candidate_quality_ratio=(eligible / len(decisions) if decisions else 0.0),
            soft_evidence_gaps=tuple(state.soft_evidence_gaps),
            documentary_coverage_ratio=float(
                documentary_result.get("documentary_coverage_ratio") or 0.0
            ),
            documentary_backlog_count=len(
                documentary_result.get("ingestion_backlog") or []
            ),
            total_action_latency_ms=sum(action_latency_ms.values()),
            action_latency_ms=action_latency_ms,
            failed_actions=failed,
            partial_actions=partial,
        )

    @staticmethod
    def _action_latencies(state: ReasoningState) -> dict[str, int]:
        if state.trace is None:
            return {}
        starts = {}
        latencies: dict[str, int] = {}
        for event in state.trace.events:
            if event.event_type.value == "action_started" and event.action_id:
                starts[event.action_id] = datetime.fromisoformat(event.timestamp)
            elif (
                event.event_type.value == "action_finished"
                and event.action_id
                and event.action_id in starts
            ):
                elapsed = (
                    datetime.fromisoformat(event.timestamp)
                    - starts[event.action_id]
                ).total_seconds() * 1000
                key = event.action_type or event.action_id
                latencies[key] = int(elapsed)
        return latencies
