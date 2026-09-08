"""Financial-safety framing for FIES outputs."""
from __future__ import annotations

class FinancialSafetyBoundary:
    def framing(self, personalized: bool = False) -> str:
        if personalized:
            return "This analysis requires suitability inputs before personalized investment guidance."
        return "Research ranking based on available evidence; not personalized investment advice."
