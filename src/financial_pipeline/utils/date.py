from datetime import date, timedelta


def parse_date(value: str) -> date:
    """Parse an ISO-format date string (YYYY-MM-DD)."""
    return date.fromisoformat(value)


def date_range(start: date | str, end: date | str) -> list[date]:
    """Return an inclusive list of dates from start to end."""
    s = parse_date(start) if isinstance(start, str) else start
    e = parse_date(end) if isinstance(end, str) else end
    delta = (e - s).days
    return [s + timedelta(days=i) for i in range(delta + 1)]
