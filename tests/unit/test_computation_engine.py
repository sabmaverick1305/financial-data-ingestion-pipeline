from decimal import Decimal

import pytest

from financial_pipeline.intelligence.computation import ComputationEngine


def test_percentage_change() -> None:
    result = ComputationEngine().percentage_change(Decimal("8400"), Decimal("5100"))
    assert result.absolute_change == Decimal("-3300")
    assert float(result.percentage_change) == pytest.approx(-39.2857142857)


def test_cagr() -> None:
    result = ComputationEngine().cagr(
        start_value=Decimal("100"),
        end_value=Decimal("161.051"),
        years=Decimal("5"),
    )
    assert result.cagr_percent is not None
    assert float(result.cagr_percent) == pytest.approx(10.0, rel=1e-3)


def test_z_score_detects_outlier() -> None:
    result = ComputationEngine().z_score(
        historical_values=[
            Decimal("8000"), Decimal("8400"), Decimal("7900"),
            Decimal("8600"), Decimal("8100"), Decimal("8300"),
        ],
        current_value=Decimal("5100"),
    )
    assert result is not None
    assert result.z_score is not None
    assert result.z_score < Decimal("-2")


def test_rank_descending() -> None:
    result = ComputationEngine().rank(
        metric="net_inflow",
        values={
            "Mid Cap Fund": Decimal("5100"),
            "Large Cap Fund": Decimal("7500"),
            "Small Cap Fund": Decimal("3200"),
        },
    )
    assert result.values[0].entity == "Large Cap Fund"
    assert result.values[0].rank == 1
