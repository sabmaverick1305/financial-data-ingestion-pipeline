"""DDL and parsing helpers for amfi_amc_stats (pre-2020 legacy format).

Pre-2020 AMFI monthly reports categorise data by AMC-sponsor type (Bank
Sponsored / Institutions / Private Sector) in Table 1, and by broad scheme
type (Income, Equity, Balanced, etc.) in Table 4.

amfi_amc_stats stores:
  • One row per scheme_type from Table 4 (AUM breakdown).
  • One row with scheme_type='Total' that additionally carries total_mobilized,
    redemption and net_inflow from the GRAND TOTAL row of Table 1.

Available for 2009–2019. From 2020 onwards use amfi_fund_stats instead.
"""
from __future__ import annotations

import re
from sqlalchemy import text as sa_text

# ── DDL ───────────────────────────────────────────────────────────────────────

DDL = """
CREATE TABLE IF NOT EXISTS amfi_amc_stats (
    id                 SERIAL PRIMARY KEY,
    period_year        INTEGER NOT NULL,
    period_month       INTEGER NOT NULL,
    scheme_type        TEXT    NOT NULL,
    total_mobilized    NUMERIC(18, 2),
    redemption         NUMERIC(18, 2),
    net_inflow         NUMERIC(18, 2),
    aum                NUMERIC(18, 2),
    aum_pct            NUMERIC(5, 2),
    source_document_id UUID REFERENCES document_metadata(document_id) ON DELETE SET NULL,
    created_at         TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE (period_year, period_month, scheme_type)
);

CREATE INDEX IF NOT EXISTS idx_amfi_amc_stats_year_month
    ON amfi_amc_stats (period_year, period_month);

CREATE INDEX IF NOT EXISTS idx_amfi_amc_stats_scheme_type
    ON amfi_amc_stats (scheme_type);
"""

# Scheme types as they appear in Table 4 of pre-2020 AMFI monthly reports
SCHEME_TYPES = [
    "INCOME",
    "INFRASTRUCTURE DEBT FUND",
    "EQUITY",
    "BALANCED",
    "LIQUID/MONEY MARKET",
    "GILT",
    "ELSS - EQUITY",
    "GOLD ETF",
    "OTHER ETFs",
    "FUND OF FUNDS INVESTING OVERSEAS",
]

_SCHEME_TYPE_SET = set(SCHEME_TYPES)

# Canonical display names (stored in DB)
_CANONICAL: dict[str, str] = {
    "INCOME":                           "Income",
    "INFRASTRUCTURE DEBT FUND":         "Infrastructure Debt Fund",
    "EQUITY":                           "Equity",
    "BALANCED":                         "Balanced",
    "LIQUID/MONEY MARKET":              "Liquid/Money Market",
    "GILT":                             "Gilt",
    "ELSS - EQUITY":                    "ELSS - Equity",
    "GOLD ETF":                         "Gold ETF",
    "OTHER ETFs":                       "Other ETFs",
    "FUND OF FUNDS INVESTING OVERSEAS": "Fund of Funds Investing Overseas",
}

_STOP_MARKERS = {"Note", "Notes", "TABLE", "Table", "@", "Fund of Funds"}


def create_table(engine) -> None:
    with engine.begin() as conn:
        conn.execute(sa_text(DDL))


# ── Number parsing ────────────────────────────────────────────────────────────

def _parse_num(s: str) -> float | None:
    s = s.strip()
    if not s or s in ("-", "@", "^"):
        return None
    try:
        return float(s.replace(",", ""))
    except ValueError:
        return None


def _is_percentage(s: str) -> bool:
    """True for values that look like a whole-number percentage (1-100)."""
    v = _parse_num(s)
    return v is not None and 0 < v <= 100 and v == int(v)


# ── Table 4 parser — AUM by scheme type ──────────────────────────────────────

def _parse_table4(lines: list[str]) -> dict[str, dict]:
    """Extract AUM and % of total per scheme type from Table 4 lines.

    Returns dict mapping canonical scheme_type → {aum, aum_pct}.
    """
    results: dict[str, dict] = {}
    i = 0
    n = len(lines)

    while i < n:
        raw = lines[i].strip()

        # Match exact scheme type names (upper-case)
        if raw.upper() in _SCHEME_TYPE_SET or raw.upper() == "TOTAL":
            canonical = _CANONICAL.get(raw.upper(), raw.title())
            if raw.upper() == "TOTAL":
                canonical = "Total"

            # Collect numeric tokens until next scheme/marker
            nums: list[str] = []
            j = i + 1
            while j < n:
                tok = lines[j].strip()
                if tok.upper() in _SCHEME_TYPE_SET or tok.upper() == "TOTAL":
                    break
                if any(tok.startswith(m) for m in _STOP_MARKERS):
                    break
                if tok == "@":
                    nums.append("0.5")   # "Less than 1%"
                elif tok and tok not in ("Open End", "Close End", "Interval Fund",
                                          "TOTAL", "% to Total"):
                    nums.append(tok)
                j += 1

            # AUM is the largest non-pct value; pct is the trailing small integer
            # Typically: Open, Close, Interval, Total, Pct (5 values)
            # Or: Total, Pct (2 values for older formats)
            aum = aum_pct = None
            if nums:
                # Last value: percentage if it's a whole number ≤ 100
                if _is_percentage(nums[-1]):
                    aum_pct = _parse_num(nums[-1])
                    aum = _parse_num(nums[-2]) if len(nums) >= 2 else None
                else:
                    aum = _parse_num(nums[-1])

            if aum is not None:
                results[canonical] = {"aum": aum, "aum_pct": aum_pct}

            i = j
        else:
            i += 1

    return results


# ── Table 1 parser — GRAND TOTAL mobilization + redemption ───────────────────

def _parse_grand_total(lines: list[str]) -> dict | None:
    """Extract total_mobilized and redemption from the GRAND TOTAL row of Table 1.

    The GRAND TOTAL row in pre-2020 reports has 7 numeric values:
      [0] NFO schemes count
      [1] NFO amount
      [2] Existing schemes amount
      [3] Total for month  ← total_mobilized
      [4] Cumulative YTD sales
      [5] Redemption for month  ← redemption
      [6] Cumulative YTD redemption
    """
    for i, line in enumerate(lines):
        if "GRAND TOTAL" in line.upper():
            nums: list[float] = []
            for j in range(i + 1, min(i + 20, len(lines))):
                v = _parse_num(lines[j])
                if v is not None:
                    nums.append(v)
                elif lines[j].strip() in ("-",):
                    nums.append(0.0)
                # Stop when we hit a text line that isn't a number
                elif lines[j].strip() and not lines[j].strip().replace(",", "").replace(".", "").isdigit():
                    # Allow "Figures for corresponding period..." to end the search
                    if len(nums) >= 4:
                        break
            if len(nums) >= 6:
                return {
                    "total_mobilized": nums[3],
                    "redemption":      nums[5],
                    "net_inflow":      round(nums[3] - nums[5], 2),
                }
            elif len(nums) >= 4:
                return {
                    "total_mobilized": nums[3],
                    "redemption":      None,
                    "net_inflow":      None,
                }
    return None


# ── Top-level extractor ───────────────────────────────────────────────────────

def extract_rows_from_pdf_text(
    pages_text: list[str], year: int, month: int
) -> list[dict]:
    """Extract all amfi_amc_stats rows from a list of page texts.

    Returns a list of dicts with keys matching amfi_amc_stats columns
    (excluding period_year, period_month, source_document_id, id, created_at).
    """
    # Flatten all pages into lines
    all_lines = []
    for page_text in pages_text:
        all_lines.extend(page_text.splitlines())

    # Table 1: GRAND TOTAL row (page 0)
    page0_lines = pages_text[0].splitlines() if pages_text else []
    grand_total = _parse_grand_total(page0_lines)

    # Table 4: AUM by scheme type (last page, or whichever has ASSETS UNDER MANAGEMENT)
    aum_data: dict[str, dict] = {}
    for page_text in reversed(pages_text):
        if "ASSETS UNDER MANAGEMENT" in page_text.upper() or "TABLE 4" in page_text.upper():
            aum_data = _parse_table4(page_text.splitlines())
            if aum_data:
                break

    rows: list[dict] = []

    # Scheme-type rows (AUM only)
    for scheme_type, vals in aum_data.items():
        if scheme_type == "Total":
            continue   # handled below as the aggregate row
        rows.append({
            "scheme_type":      scheme_type,
            "total_mobilized":  None,
            "redemption":       None,
            "net_inflow":       None,
            "aum":              vals.get("aum"),
            "aum_pct":          vals.get("aum_pct"),
        })

    # Total row: combine Grand Total mobilisation + Total AUM
    total_aum = aum_data.get("Total", {})
    rows.append({
        "scheme_type":      "Total",
        "total_mobilized":  grand_total["total_mobilized"] if grand_total else None,
        "redemption":       grand_total["redemption"]      if grand_total else None,
        "net_inflow":       grand_total["net_inflow"]      if grand_total else None,
        "aum":              total_aum.get("aum"),
        "aum_pct":          100.0,
    })

    return rows
