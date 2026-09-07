"""Deterministic financial computation primitives used by FIES reasoning."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from math import sqrt


@dataclass(frozen=True)
class PercentageChange:
    previous: Decimal
    current: Decimal
    absolute_change: Decimal
    percentage_change: Decimal | None


@dataclass(frozen=True)
class CAGRResult:
    start_value: Decimal
    end_value: Decimal
    years: Decimal
    cagr_percent: Decimal | None


@dataclass(frozen=True)
class AnomalyScore:
    current_value: Decimal
    mean: Decimal
    stddev: Decimal
    z_score: Decimal | None


@dataclass(frozen=True)
class RankedValue:
    entity: str
    value: Decimal
    rank: int


@dataclass(frozen=True)
class RankingResult:
    metric: str
    values: tuple[RankedValue, ...]


class ComputationEngine:
    def percentage_change(self, previous: Decimal, current: Decimal) -> PercentageChange:
        absolute = current - previous
        pct = None if previous == 0 else (absolute / abs(previous)) * Decimal("100")
        return PercentageChange(previous, current, absolute, pct)

    def cagr(self, *, start_value: Decimal, end_value: Decimal, years: Decimal) -> CAGRResult:
        if start_value <= 0 or end_value <= 0 or years <= 0:
            return CAGRResult(start_value, end_value, years, None)
        result = (float(end_value / start_value) ** (1 / float(years)) - 1) * 100
        return CAGRResult(start_value, end_value, years, Decimal(str(result)))

    def mean(self, values: list[Decimal]) -> Decimal | None:
        if not values:
            return None
        return sum(values, Decimal("0")) / Decimal(len(values))

    def sample_stddev(self, values: list[Decimal]) -> Decimal | None:
        if len(values) < 2:
            return None
        mean_value = self.mean(values)
        assert mean_value is not None
        variance = sum(((v - mean_value) ** 2 for v in values), Decimal("0")) / Decimal(len(values) - 1)
        return Decimal(str(sqrt(float(variance))))

    def z_score(self, *, historical_values: list[Decimal], current_value: Decimal) -> AnomalyScore | None:
        if len(historical_values) < 2:
            return None
        mean_value = self.mean(historical_values)
        stddev = self.sample_stddev(historical_values)
        if mean_value is None or stddev is None:
            return None
        z = None if stddev == 0 else (current_value - mean_value) / stddev
        return AnomalyScore(current_value, mean_value, stddev, z)

    def rank(self, *, metric: str, values: dict[str, Decimal], descending: bool = True) -> RankingResult:
        ordered = sorted(values.items(), key=lambda item: item[1], reverse=descending)
        ranked = tuple(RankedValue(entity=e, value=v, rank=i) for i, (e, v) in enumerate(ordered, start=1))
        return RankingResult(metric=metric, values=ranked)
