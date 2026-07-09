"""Pure computation of per-scheme performance metrics from a NAV history.

No I/O here — takes a sorted (date, nav) series in, returns a
PerformanceMetrics dataclass out. Keeps the DB/threading concerns in
repository.py/run.py fully separate from the math.

Period lookups use calendar-day offsets (30/91/182/365/1095/1825/3650 days
for 1m/3m/6m/1y/3y/5y/10y — the standard approximation used by most fund
data providers) resolved "as of" the closest available NAV on or before the
target date, since NAV isn't published on non-trading days. return_1d
instead uses the single previous available NAV entry (unambiguous, and
"1 calendar day ago" would frequently land on a weekend).
"""

from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date, timedelta

_TRADING_DAYS_PER_YEAR = 252

# (field name, calendar-day offset) — resolved via as-of lookup, not return_1d.
_SIMPLE_RETURN_PERIODS = (
    ("return_1w", 7),
    ("return_1m", 30),
    ("return_3m", 91),
    ("return_6m", 182),
    ("return_1y", 365),
)
_CAGR_PERIODS = (
    ("return_3y_cagr", 1095, 3),
    ("return_5y_cagr", 1825, 5),
    ("return_10y_cagr", 3650, 10),
)


@dataclass
class PerformanceMetrics:
    scheme_code: str
    latest_nav: float
    latest_nav_date: date
    return_1d: float | None = None
    return_1w: float | None = None
    return_1m: float | None = None
    return_3m: float | None = None
    return_6m: float | None = None
    return_1y: float | None = None
    return_3y_cagr: float | None = None
    return_5y_cagr: float | None = None
    return_10y_cagr: float | None = None
    all_time_return: float | None = None
    rolling_volatility: float | None = None
    rolling_stddev: float | None = None
    nav_high_52w: float | None = None
    nav_low_52w: float | None = None


def _nav_as_of(dates: list[date], navs: list[float], target: date) -> float | None:
    """Closest NAV on or before `target`. None if target predates all history."""
    idx = bisect_right(dates, target) - 1
    return navs[idx] if idx >= 0 else None


def _simple_return(latest: float, old: float | None) -> float | None:
    if not old:
        return None
    return (latest / old - 1) * 100


def _cagr(latest: float, old: float | None, years: float) -> float | None:
    if not old or old <= 0:
        return None
    return ((latest / old) ** (1 / years) - 1) * 100


def _sample_stddev(values: list[float]) -> float | None:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return math.sqrt(variance)


def compute_performance(scheme_code: str, history: list[tuple[date, float]]) -> PerformanceMetrics | None:
    """`history` must be sorted ascending by date with no duplicate dates."""
    if not history:
        return None

    dates = [d for d, _ in history]
    navs = [n for _, n in history]
    latest_date, latest_nav = history[-1]

    metrics = PerformanceMetrics(scheme_code=scheme_code, latest_nav=latest_nav, latest_nav_date=latest_date)

    metrics.return_1d = _simple_return(latest_nav, navs[-2]) if len(navs) >= 2 else None

    for field, offset_days in _SIMPLE_RETURN_PERIODS:
        old_nav = _nav_as_of(dates, navs, latest_date - timedelta(days=offset_days))
        setattr(metrics, field, _simple_return(latest_nav, old_nav))

    for field, offset_days, years in _CAGR_PERIODS:
        old_nav = _nav_as_of(dates, navs, latest_date - timedelta(days=offset_days))
        setattr(metrics, field, _cagr(latest_nav, old_nav, years))

    metrics.all_time_return = _simple_return(latest_nav, navs[0])

    window_start = latest_date - timedelta(days=365)
    idx_52w = bisect_right(dates, window_start)
    window_navs = navs[idx_52w:] or [latest_nav]
    metrics.nav_high_52w = max(window_navs)
    metrics.nav_low_52w = min(window_navs)

    # Annualized volatility from the stddev of daily returns over the full
    # available history (not a trailing window — see mf_performance/README
    # decision: whole-history stddev is more stable for schemes with gaps).
    daily_returns = [navs[i] / navs[i - 1] - 1 for i in range(1, len(navs)) if navs[i - 1] > 0]
    stddev = _sample_stddev(daily_returns)
    if stddev is not None:
        metrics.rolling_stddev = stddev * 100
        metrics.rolling_volatility = stddev * math.sqrt(_TRADING_DAYS_PER_YEAR) * 100

    return metrics
