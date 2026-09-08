"""Deterministic confidence scoring for reasoning results."""
from __future__ import annotations
from dataclasses import dataclass
from financial_pipeline.intelligence.reasoning_state import ReasoningState

@dataclass(frozen=True)
class ConfidenceScore:
    score: float
    evidence_coverage: float
    source_quality: float
    freshness: float
    contradiction_penalty: float
    replan_penalty: float
    soft_evidence_penalty: float

class ConfidenceScorer:
    def score(self, state: ReasoningState, *, required_dimensions: int = 9) -> ConfidenceScore:
        unique_success = len({
            o.action_type
            for o in state.observations
            if o.status.value == "succeeded"
        })
        coverage = min(1.0, unique_success / max(1, required_dimensions))
        successful = [o for o in state.observations if o.status.value == "succeeded"]
        source_quality = (
            1.0
            if all(
                (not observation.evidence_refs)
                or all(ref.startswith("verified:") for ref in observation.evidence_refs)
                for observation in successful
            )
            else 0.8
        )
        freshness = 1.0
        contradiction_penalty = 0.0
        replan_penalty = min(0.2, state.replan_count * 0.05)
        soft_evidence_penalty = min(0.24, len(set(state.soft_evidence_gaps)) * 0.08)
        score = max(
            0.0,
            min(
                1.0,
                coverage * 0.55
                + source_quality * 0.25
                + freshness * 0.20
                - contradiction_penalty
                - replan_penalty
                - soft_evidence_penalty,
            ),
        )
        return ConfidenceScore(
            score,
            coverage,
            source_quality,
            freshness,
            contradiction_penalty,
            replan_penalty,
            soft_evidence_penalty,
        )
