from decimal import Decimal

from financial_pipeline.intelligence.change_detection import ChangeDetector, ChangeDirection, MetricChange
from financial_pipeline.intelligence.contracts import ScenarioDataContract
from financial_pipeline.intelligence.materiality import MaterialityLevel, MaterialityPolicy
from financial_pipeline.intelligence.models import DataCoverage, FinancialFact, SourceProvenance
from financial_pipeline.intelligence.scenario import GoldenScenario


def _fact(month: int, value: str) -> FinancialFact:
    return FinancialFact(
        entity_type="fund_category",
        entity="Mid Cap Fund",
        metric="net_inflow",
        value=Decimal(value),
        unit="INR_CRORE",
        period_year=2026,
        period_month=month,
        provenance=SourceProvenance(source_name="AMFI"),
    )


def test_change_detector_and_materiality() -> None:
    scenario = GoldenScenario("fund_category", "Mid Cap Fund", "net_inflow", 2026, 8)
    coverage = DataCoverage.monthly(year=2026, observed_months=range(1, 9), expected_through_month=8)
    contract = ScenarioDataContract(scenario, _fact(8, "5100"), _fact(7, "8400"), coverage)

    change = ChangeDetector().detect(contract)
    assert change.direction is ChangeDirection.DOWN
    assert change.absolute_change == Decimal("-3300")

    materiality = MaterialityPolicy().evaluate(change)
    assert materiality.is_material
    assert materiality.level is MaterialityLevel.HIGH


def test_flat_change_has_no_materiality() -> None:
    change = MetricChange(
        Decimal("5000"), Decimal("5000"), Decimal("0"), Decimal("0"), ChangeDirection.FLAT
    )
    result = MaterialityPolicy().evaluate(change)
    assert not result.is_material
    assert result.level is MaterialityLevel.NONE
