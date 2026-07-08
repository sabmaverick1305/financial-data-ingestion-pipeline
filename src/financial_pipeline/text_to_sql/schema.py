"""DDL and helpers for the amfi_fund_stats structured table.

amfi_fund_stats holds one row per (year, month, fund_category) and stores
the seven AMFI monthly report columns in typed numeric form so that Vanna
can generate and execute precise SQL against them.
"""
from __future__ import annotations

from sqlalchemy import text


DDL = """
CREATE TABLE IF NOT EXISTS amfi_fund_stats (
    id                 SERIAL PRIMARY KEY,
    period_year        INTEGER NOT NULL,
    period_month       INTEGER NOT NULL,
    fund_category      TEXT    NOT NULL,
    no_of_schemes      INTEGER,
    no_of_folios       BIGINT,
    funds_mobilized    NUMERIC(18, 2),
    redemption         NUMERIC(18, 2),
    net_inflow         NUMERIC(18, 2),
    aum                NUMERIC(18, 2),
    avg_aum            NUMERIC(18, 2),
    source_document_id UUID REFERENCES document_metadata(document_id) ON DELETE SET NULL,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (period_year, period_month, fund_category)
);

CREATE INDEX IF NOT EXISTS idx_amfi_stats_year_month
    ON amfi_fund_stats (period_year, period_month);

CREATE INDEX IF NOT EXISTS idx_amfi_stats_category
    ON amfi_fund_stats (fund_category);

CREATE INDEX IF NOT EXISTS idx_amfi_stats_category_year_month
    ON amfi_fund_stats (fund_category, period_year, period_month);
"""

# All open-ended equity fund categories as they appear in AMFI reports
EQUITY_FUND_CATEGORIES: list[str] = [
    "Multi Cap Fund",
    "Large Cap Fund",
    "Large & Mid Cap Fund",
    "Mid Cap Fund",
    "Small Cap Fund",
    "Dividend Yield Fund",
    "Value Fund/Contra Fund",
    "Focused Fund",
    "Sectoral/Thematic Funds",
    "ELSS",
    "Flexi Cap Fund",
]

ALL_FUND_CATEGORIES: list[str] = EQUITY_FUND_CATEGORIES + [
    "Overnight Fund",
    "Liquid Fund",
    "Ultra Short Duration Fund",
    "Low Duration Fund",
    "Money Market Fund",
    "Short Duration Fund",
    "Medium Duration Fund",
    "Medium to Long Duration Fund",
    "Long Duration Fund",
    "Dynamic Bond Fund",
    "Corporate Bond Fund",
    "Credit Risk Fund",
    "Banking and PSU Fund",
    "Gilt Fund",
    "Gilt Fund with 10 year constant duration",
    "Floater Fund",
    "Conservative Hybrid Fund",
    "Balanced Hybrid Fund/Aggressive Hybrid Fund",
    "Dynamic Asset Allocation/Balanced Advantage Fund",
    "Multi Asset Allocation Fund",
    "Arbitrage Fund",
    "Equity Savings Fund",
    "Retirement Fund",
    "Childrens Fund",
    "Index Funds",
    "GOLD ETF",
    "Other ETFs",
    "Fund of funds investing overseas",
]

_ALL_CATEGORY_SET = set(ALL_FUND_CATEGORIES)


def create_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(text(DDL))


def parse_number(raw: str) -> float | None:
    """Parse Indian-formatted number string to float. Returns None for '-' or blank."""
    s = raw.strip()
    if not s or s == "-":
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def parse_int(raw: str) -> int | None:
    v = parse_number(raw)
    return int(v) if v is not None else None


def extract_rows_from_chunk(text: str) -> list[dict]:
    """Extract fund-category rows from a pipe-separated markdown chunk.

    Returns list of dicts with keys:
        fund_category, no_of_schemes, no_of_folios,
        funds_mobilized, redemption, net_inflow, aum, avg_aum
    """
    rows: list[dict] = []
    for line in text.splitlines():
        if "|" not in line:
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) < 4:
            continue
        # First cell may be "iv Mid Cap Fund" or just "Mid Cap Fund"
        first = cells[0]
        # Strip leading roman numeral / number prefix
        fund_name = _strip_row_prefix(first)
        if fund_name not in _ALL_CATEGORY_SET:
            continue

        # Map remaining cells to columns (7 after name: schemes, folios,
        # mobilized, redemption, net_inflow, aum, avg_aum)
        vals = cells[1:]
        rows.append({
            "fund_category":   fund_name,
            "no_of_schemes":   parse_int(vals[0])   if len(vals) > 0 else None,
            "no_of_folios":    parse_int(vals[1])   if len(vals) > 1 else None,
            "funds_mobilized": parse_number(vals[2]) if len(vals) > 2 else None,
            "redemption":      parse_number(vals[3]) if len(vals) > 3 else None,
            "net_inflow":      parse_number(vals[4]) if len(vals) > 4 else None,
            "aum":             parse_number(vals[5]) if len(vals) > 5 else None,
            "avg_aum":         parse_number(vals[6]) if len(vals) > 6 else None,
        })
    return rows


def _strip_row_prefix(cell: str) -> str:
    """Remove leading roman numeral or number prefix from a cell like 'iv Mid Cap Fund'."""
    import re
    return re.sub(r"^[ivxlcdmIVXLCDM]+\s+", "", cell).strip()
