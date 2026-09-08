"""Final deterministic synthesis from verified observations."""
from __future__ import annotations
from financial_pipeline.intelligence.confidence import ConfidenceScore
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.safety import FinancialSafetyBoundary

class AnswerSynthesizer:
    def __init__(self) -> None:
        self._safety = FinancialSafetyBoundary()
    def synthesize(self, state: ReasoningState, confidence: ConfidenceScore) -> str:
        verified = [o for o in state.observations if o.status.value == "succeeded"]
        actions = ", ".join(dict.fromkeys(o.action_type.value for o in verified))
        return (
            f"FIES research completed with confidence={confidence.score:.2f}. "
            f"Verified evidence dimensions were produced by: {actions}. "
            + self._safety.framing(personalized=False)
        )
