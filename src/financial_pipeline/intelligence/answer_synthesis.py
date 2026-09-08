"""Final deterministic synthesis from verified observations."""
from __future__ import annotations
from financial_pipeline.intelligence.confidence import ConfidenceScore
from financial_pipeline.intelligence.reasoning_state import ReasoningState
from financial_pipeline.intelligence.safety import FinancialSafetyBoundary

class AnswerSynthesizer:
    def __init__(self) -> None:
        self._safety = FinancialSafetyBoundary()

    def synthesize(self, state: ReasoningState, confidence: ConfidenceScore) -> str:
        if not state.ranked_funds:
            return (
                f"FIES completed the evidence workflow with confidence={confidence.score:.2f}, "
                "but there was not enough aligned scheme-level evidence to produce a ranked shortlist. "
                + self._safety.framing(personalized=False)
            )

        lines = [
            f"FIES research ranking (confidence={confidence.score:.2f}):"
        ]
        for index, fund in enumerate(state.ranked_funds[:5], 1):
            lines.append(
                f"{index}. {fund['scheme_name']} [{fund['category']}] "
                f"score={fund['score']:.2f}; "
                f"1Y={self._fmt(fund.get('return_1y'))}; "
                f"3Y CAGR={self._fmt(fund.get('return_3y_cagr'))}; "
                f"Sharpe={self._fmt(fund.get('sharpe_ratio'))}; "
                f"Max DD={self._fmt(fund.get('max_drawdown'))}; "
                f"Peer percentile={self._fmt(fund.get('peer_percentile'))}"
            )
        lines.append(self._safety.framing(personalized=False))
        return "\n".join(lines)

    @staticmethod
    def _fmt(value) -> str:
        return "n/a" if value is None else f"{float(value):.2f}"
