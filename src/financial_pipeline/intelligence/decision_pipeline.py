"""Formal candidate decision pipeline for FIES recommendations."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.research_plan import ActionStatus, ActionType

class GateStatus(StrEnum):
    PASS = "pass"
    PARTIAL = "partial"
    FAIL = "fail"

@dataclass(frozen=True)
class GateResult:
    stage: str
    status: GateStatus
    reasons: tuple[str, ...] = ()
    evidence_refs: tuple[str, ...] = ()

@dataclass
class CandidateDecision:
    scheme_code: str
    scheme_name: str
    category: str
    gates: dict[str, GateResult] = field(default_factory=dict)

    @property
    def eligible_for_ranking(self) -> bool:
        required = ("eligibility", "data_quality", "evidence", "peer_comparison")
        return all(
            self.gates.get(stage) is not None
            and self.gates[stage].status is not GateStatus.FAIL
            for stage in required
        )

class CandidateDecisionPipeline:
    def evaluate(self, state: ReasoningState) -> dict[str, CandidateDecision]:
        latest = {}
        for observation in state.observations:
            latest[observation.action_type] = observation

        discovered = latest.get(ActionType.DISCOVER_FUNDS)
        decisions: dict[str, CandidateDecision] = {}
        if discovered is None or discovered.status not in (ActionStatus.SUCCEEDED, ActionStatus.PARTIAL):
            return decisions

        result = discovered.result if isinstance(discovered.result, dict) else {}
        for fund in result.get("funds", []):
            if not isinstance(fund, dict) or not fund.get("scheme_code"):
                continue
            code = str(fund["scheme_code"])
            decision = CandidateDecision(
                scheme_code=code,
                scheme_name=str(fund.get("scheme_name") or code),
                category=str(fund.get("category") or ""),
            )
            decisions[code] = decision

            eligible = decision.category in set(state.eligible_categories)
            decision.gates["eligibility"] = GateResult(
                stage="eligibility",
                status=GateStatus.PASS if eligible else GateStatus.FAIL,
                reasons=() if eligible else (f"category {decision.category} is outside mandate",),
                evidence_refs=("verified:deterministic:investment_mandate",),
            )

            quality = fund.get("data_quality") or {}
            quality_valid = bool(quality.get("valid", False))
            issues = tuple(
                str(item.get("reason") or item.get("code") or item)
                if isinstance(item, dict) else str(item)
                for item in quality.get("issues", [])
            )
            decision.gates["data_quality"] = GateResult(
                stage="data_quality",
                status=GateStatus.PASS if quality_valid else GateStatus.FAIL,
                reasons=issues or (() if quality_valid else ("candidate did not pass data-quality gate",)),
                evidence_refs=("verified:deterministic:data_quality_gate",),
            )

        perf = self._rows_by_code(latest.get(ActionType.FETCH_PERFORMANCE))
        risk = self._rows_by_code(latest.get(ActionType.COMPUTE_RISK))
        peer = self._peer_by_code(latest.get(ActionType.COMPARE_PEERS))

        for code, decision in decisions.items():
            missing = []
            if code not in perf:
                missing.append("performance")
            if code not in risk:
                missing.append("risk")
            decision.gates["evidence"] = GateResult(
                stage="evidence",
                status=GateStatus.PASS if not missing else GateStatus.FAIL,
                reasons=tuple(f"missing hard evidence: {item}" for item in missing),
                evidence_refs=tuple(
                    ref
                    for action_type in (ActionType.FETCH_PERFORMANCE, ActionType.COMPUTE_RISK)
                    for ref in self._refs(latest.get(action_type))
                ),
            )

            peer_value = peer.get(code)
            decision.gates["peer_comparison"] = GateResult(
                stage="peer_comparison",
                status=GateStatus.PASS if peer_value is not None else GateStatus.FAIL,
                reasons=() if peer_value is not None else ("candidate missing peer percentile",),
                evidence_refs=self._refs(latest.get(ActionType.COMPARE_PEERS)),
            )

        return decisions

    @staticmethod
    def _rows_by_code(observation):
        if observation is None or observation.status not in (ActionStatus.SUCCEEDED, ActionStatus.PARTIAL):
            return {}
        result = observation.result if isinstance(observation.result, dict) else {}
        rows = result.get("rows", [])
        if result.get("scope") == "scheme":
            rows = [result]
        return {
            str(row.get("scheme_code")): row
            for row in rows
            if isinstance(row, dict) and row.get("scheme_code")
        }

    @staticmethod
    def _peer_by_code(observation):
        if observation is None or observation.status not in (ActionStatus.SUCCEEDED, ActionStatus.PARTIAL):
            return {}
        result = observation.result if isinstance(observation.result, dict) else {}
        groups = result.get("categories", [])
        if result.get("scope") == "scheme_peer_set":
            groups = [result]
        peer = {}
        for group in groups:
            for row in group.get("peers", []):
                if isinstance(row, dict) and row.get("scheme_code"):
                    peer[str(row["scheme_code"])] = row.get("percentile_rank")
        return peer

    @staticmethod
    def _refs(observation):
        return tuple(observation.evidence_refs) if observation is not None else ()
