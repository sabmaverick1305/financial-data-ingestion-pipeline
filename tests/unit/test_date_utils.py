from datetime import date

from financial_pipeline.utils.date import date_range, parse_date


def test_parse_date() -> None:
    assert parse_date("2024-01-15") == date(2024, 1, 15)


def test_date_range_inclusive() -> None:
    result = date_range("2024-01-01", "2024-01-03")
    assert result == [date(2024, 1, 1), date(2024, 1, 2), date(2024, 1, 3)]


def test_date_range_single_day() -> None:
    result = date_range("2024-06-01", "2024-06-01")
    assert result == [date(2024, 6, 1)]
