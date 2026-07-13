"""Reasoning Engine — evaluates domain/semantic/reasoning_rules.yaml against real
computed metric directions to answer causal "why" queries.

    "Why did AUM increase while net inflow decreased for Large Cap funds
     between 2020 and 2024?"
      -> reasoning_data.py fetches aum/net_inflow (+ funds_mobilized/redemption
         where relevant) at the period's start and end from amfi_fund_stats
      -> this module computes % change per metric and classifies it into a
         direction (increase/decrease/stable/spike) per reasoning_rules.yaml's
         direction_thresholds
      -> matches the directions against reasoning_rules.yaml's IF/THEN rules
      -> renders the matched rule's explanation_template with the real
         computed numbers, grounded by its supporting_relationship

This is the capability fies/ontology/templates.yaml's T_REASON_001 is marked
`status: unimplemented` for — "AMFI reports are raw data tables, not
explanations" (no prose ever states market appreciation caused an AUM move).
This engine derives the explanation from the data itself, instead of
retrieving prose that doesn't exist in the corpus.

Pure logic, no I/O — reasoning_data.py owns fetching MetricObservations from
the database; this module only matches and renders.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from financial_pipeline.semantic.semantic_engine import get_engine


@dataclass
class MetricObservation:
    metric_id: str
    start_value: float
    end_value: float
    start_label: str  # e.g. "2020" or "Jan 2020" — for the rendered explanation
    end_label: str

    @property
    def pct_change(self) -> float:
        if self.start_value == 0:
            return 0.0 if self.end_value == 0 else 100.0
        return ((self.end_value - self.start_value) / abs(self.start_value)) * 100.0

    def direction(self) -> str:
        """increase/decrease/stable, by sign — independent of magnitude.
        A 30% jump is direction="increase" AND is_spike()=True at once;
        "spike" is a magnitude flag layered on top, not a 4th exclusive
        direction (see is_spike)."""
        th = get_engine().direction_thresholds()
        stable_band = th.get("stable_band_pct", 2.0)
        pct = self.pct_change
        if pct > stable_band:
            return "increase"
        if pct < -stable_band:
            return "decrease"
        return "stable"

    def is_spike(self) -> bool:
        spike_band = get_engine().direction_thresholds().get("spike_band_pct", 25.0)
        return abs(self.pct_change) >= spike_band


@dataclass
class ReasoningResult:
    matched: bool
    rule_id: str | None
    explanation: str
    confidence: str
    supporting_relationship: str | None = None
    supporting_pattern: str | None = None
    observations: dict[str, MetricObservation] = field(default_factory=dict)

    def describe(self) -> str:
        parts = [f"rule={self.rule_id or 'none'}", f"confidence={self.confidence}"]
        if self.supporting_relationship:
            parts.append(f"relationship={self.supporting_relationship}")
        return " ".join(parts)


def _template_context(observations: dict[str, MetricObservation]) -> dict[str, str]:
    """Build the {metric_pct_change}-style substitution values an
    explanation_template in reasoning_rules.yaml can reference."""
    ctx: dict[str, str] = {}
    for metric_id, obs in observations.items():
        ctx[f"{metric_id}_pct_change"] = f"{obs.pct_change:+.1f}"
        ctx[f"{metric_id}_pct_change_abs"] = f"{abs(obs.pct_change):.1f}"

    # aum_movement's non-flow ("market appreciation") contribution: the
    # residual once net_inflow's crore-value change is subtracted from
    # aum's crore-value change. Only meaningful when both are present —
    # this is an approximation (ignores dividend payouts/valuation nuances
    # financial_relationships.yaml's aum_movement already documents as
    # separate, smaller contributors).
    if "aum" in observations and "net_inflow" in observations:
        aum_delta = observations["aum"].end_value - observations["aum"].start_value
        inflow_delta = observations["net_inflow"].end_value - observations["net_inflow"].start_value
        ctx["implied_market_contribution"] = f"{aum_delta - inflow_delta:,.0f}"

    return ctx


def _condition_matches(cond: dict, directions: dict[str, str], spikes: dict[str, bool]) -> bool:
    metric = cond.get("metric")
    if cond.get("spike"):
        return spikes.get(metric, False)
    return directions.get(metric) == cond.get("direction")


def evaluate(observations: dict[str, MetricObservation]) -> ReasoningResult:
    """Match computed metric directions against domain/semantic/reasoning_rules.yaml.
    `observations` keys are metric_ids (e.g. "aum", "net_inflow")."""
    eng = get_engine()
    directions = {mid: obs.direction() for mid, obs in observations.items()}
    spikes = {mid: obs.is_spike() for mid, obs in observations.items()}
    ctx = _template_context(observations)

    # Collect every rule whose conditions are satisfied, then prefer the most
    # specific (most conditions) — a 3-metric "everything moved together"
    # story is more informative than a 1-metric spike rule matching the same
    # data as a coincidental subset. Ties keep reasoning_rules.yaml's order.
    matches = [
        rule for rule in eng.rules()
        if rule.get("conditions") and all(_condition_matches(c, directions, spikes) for c in rule["conditions"])
    ]
    if matches:
        rule = max(matches, key=lambda r: len(r["conditions"]))
        template = rule["conclusion"]["explanation_template"]
        try:
            explanation = template.format(**ctx)
        except (KeyError, IndexError):
            # A referenced metric wasn't fetched (e.g. funds_mobilized wasn't
            # part of this observation set) — fall back to the unrendered
            # template rather than crashing the answer path.
            explanation = template
        return ReasoningResult(
            matched=True,
            rule_id=rule["rule_id"],
            explanation=" ".join(explanation.split()),
            confidence=rule["conclusion"].get("confidence", "low"),
            supporting_relationship=rule.get("supporting_relationship"),
            supporting_pattern=rule.get("supporting_pattern"),
            observations=observations,
        )

    no_match = eng.no_match_response()
    metric_ids = list(observations.keys())
    fallback_ctx = {
        "metric_a": metric_ids[0] if metric_ids else "metric",
        "direction_a": directions.get(metric_ids[0], "unknown") if metric_ids else "unknown",
        "metric_b": metric_ids[1] if len(metric_ids) > 1 else "the other metric",
        "direction_b": directions.get(metric_ids[1], "unknown") if len(metric_ids) > 1 else "unknown",
    }
    try:
        explanation = no_match.get("explanation_template", "").format(**fallback_ctx)
    except (KeyError, IndexError):
        explanation = "No matching causal pattern found for the observed metric changes."
    return ReasoningResult(
        matched=False,
        rule_id=None,
        explanation=" ".join(explanation.split()),
        confidence=no_match.get("confidence", "none"),
        observations=observations,
    )
