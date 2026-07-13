#!/usr/bin/env python3
"""Train the Vanna text-to-SQL agent on the amfi_fund_stats schema.

Idempotent: clears existing training data before re-training so that
re-runs after schema/example changes always produce a clean state.

Usage:
    cd /Users/sabyasachi/Documents/financial-data-ingestion-pipeline
    .venv/bin/python3 scripts/train_vanna.py [--reset]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import structlog

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from financial_pipeline.config import settings
from financial_pipeline.logging import configure_logging
from financial_pipeline.text_to_sql.schema import DDL
from financial_pipeline.text_to_sql.schema_amc import DDL as AMC_DDL
from financial_pipeline.semantic.semantic_engine import get_engine
from financial_pipeline.text_to_sql.schema_mf_scheme import (
    MF_NAV_HISTORY_DDL,
    MF_SCHEME_MASTER_DDL,
    MF_SCHEME_PERFORMANCE_DDL,
)
from financial_pipeline.text_to_sql.vanna_agent import build_vanna_agent


def _mf_scheme_category_doc() -> str:
    """Build the mf_scheme_master.category reference doc from
    domain/semantic/taxonomy.yaml's mf_scheme_categories at train-time, so it
    can't drift out of sync with the ontology like a hand-copied list would.
    """
    eng = get_engine()
    by_parent: dict[str, list[dict]] = {}
    for entry in eng.mf_scheme_categories_leaf():
        by_parent.setdefault(entry["parent"], []).append(entry)

    lines = [
        "mf_scheme_master.category reference values — use these exact ILIKE patterns "
        "(not the fund_category enum from amfi_fund_stats, which uses different literal "
        "strings for the same real-world concepts, e.g. 'Large Cap Fund' vs "
        "'Equity Scheme - Large Cap Fund'):",
    ]
    for parent_id in ("equity_scheme", "debt_scheme", "hybrid_scheme", "other_scheme"):
        parent_label = (eng.mf_scheme_category(parent_id) or {}).get("label", parent_id)
        lines.append(f"\n{parent_label}:")
        for entry in by_parent.get(parent_id, []):
            lines.append(f"  '{entry['label']}' -> category ILIKE '{entry['category_pattern']}'")
    return "\n".join(lines)

log = structlog.get_logger()

# ── Documentation strings ──────────────────────────────────────────────────────

_DOCS = [
    (
        "amfi_fund_stats table",
        "The amfi_fund_stats table stores one row per (period_year, period_month, fund_category). "
        "CRITICAL, verified against live data: it ONLY has rows for period_year 2020 through 2026 — "
        "NOT 2009 onwards. There is zero row overlap with amfi_amc_stats (2009-2019). "
        "If a question's year(s) are ALL before 2020, this is the WRONG table — use amfi_amc_stats "
        "instead (see its own doc entry). If the question's years span both eras (e.g. 2015 to 2022), "
        "you need a UNION of both tables — see 'amfi_amc_stats table (pre-2020 legacy)' below. "
        "period_year is an integer (e.g. 2024). period_month is 1–12. "
        "fund_category is the full AMFI name (e.g. 'Large Cap Fund', 'Mid Cap Fund', 'ELSS'). "
        "All money columns (funds_mobilized, redemption, net_inflow, aum, avg_aum) are in crore INR (₹). "
        "no_of_folios is the number of investor accounts."
    ),
    (
        "fund_category values",
        "Valid fund_category values include: "
        "'Multi Cap Fund', 'Large Cap Fund', 'Large & Mid Cap Fund', 'Mid Cap Fund', "
        "'Small Cap Fund', 'Dividend Yield Fund', 'Value Fund/Contra Fund', 'Focused Fund', "
        "'Sectoral/Thematic Funds', 'ELSS', 'Flexi Cap Fund', 'Liquid Fund', 'Overnight Fund', "
        "'Ultra Short Duration Fund', 'Low Duration Fund', 'Money Market Fund', "
        "'Short Duration Fund', 'Medium Duration Fund', 'Long Duration Fund', "
        "'Dynamic Bond Fund', 'Corporate Bond Fund', 'Credit Risk Fund', "
        "'Banking and PSU Fund', 'Gilt Fund', 'Floater Fund', "
        "'Conservative Hybrid Fund', 'Balanced Hybrid Fund/Aggressive Hybrid Fund', "
        "'Dynamic Asset Allocation/Balanced Advantage Fund', 'Multi Asset Allocation Fund', "
        "'Arbitrage Fund', 'Equity Savings Fund', 'Index Funds', 'GOLD ETF', 'Other ETFs', "
        "'Fund of funds investing overseas', 'Retirement Fund', 'Childrens Fund'."
    ),
    (
        "month number mapping",
        "period_month values: 1=January, 2=February, 3=March, 4=April, 5=May, 6=June, "
        "7=July, 8=August, 9=September, 10=October, 11=November, 12=December. "
        "Q1 = months 1,2,3 (January–March). Q2 = months 4,5,6. "
        "Q3 = months 7,8,9. Q4 = months 10,11,12."
    ),
    (
        "SUM vs AVG guidance",
        "For total AUM over a year or quarter, use SUM(aum) or AVG(aum) depending on context. "
        "AMFI reports month-end AUM snapshots, not cumulative. "
        "For trend queries, group by period_year, period_month ORDER BY both. "
        "For year-over-year comparison, GROUP BY period_year with SUM or AVG."
    ),
    (
        "amfi_amc_stats table (pre-2020 legacy)",
        "The amfi_amc_stats table stores pre-2020 AMFI monthly report data. "
        "CRITICAL, verified against live data: it ONLY has rows for period_year 2009 through 2019 — "
        "NOT beyond. There is zero row overlap with amfi_fund_stats (2020-2026). "
        "If a question's year(s) are ALL 2020 or later, this is the WRONG table — use amfi_fund_stats "
        "instead. A query against amfi_amc_stats with period_year >= 2020 (or amfi_fund_stats with "
        "period_year < 2020) will silently return zero rows, not an error — always check the year(s) "
        "in the question against this boundary before picking a table. "
        "It has one row per (period_year, period_month, scheme_type). "
        "scheme_type values: 'Income', 'Infrastructure Debt Fund', 'Equity', 'Balanced', "
        "'Liquid/Money Market', 'Gilt', 'ELSS - Equity', 'Gold ETF', 'Other ETFs', "
        "'Fund of Funds Investing Overseas', and 'Total' (industry aggregate). "
        "aum and aum_pct come from Table 4 (AUM by scheme type). "
        "total_mobilized, redemption, net_inflow on the 'Total' row come from the GRAND TOTAL of Table 1. "
        "All money columns are in crore INR. "
        "For queries spanning 2009–2026 you may need to UNION amfi_amc_stats and amfi_fund_stats."
    ),
    (
        "amfi_amc_stats scheme_type values",
        "Valid scheme_type values in amfi_amc_stats: "
        "'Income' (debt/fixed income schemes), "
        "'Equity' (pure equity schemes, predecessor to Large/Mid/Small Cap etc.), "
        "'Balanced' (hybrid schemes), "
        "'Liquid/Money Market' (liquid + money market schemes), "
        "'Gilt' (government securities funds), "
        "'ELSS - Equity' (tax-saving equity linked saving schemes), "
        "'Gold ETF' (gold exchange-traded funds), "
        "'Other ETFs' (non-gold ETFs), "
        "'Infrastructure Debt Fund' (infrastructure debt funds), "
        "'Fund of Funds Investing Overseas' (overseas fund-of-funds), "
        "'Total' (industry grand total row — use for overall industry AUM/mobilization figures). "
        "Pre-2020 data uses broad categories; post-2020 granular categories are in amfi_fund_stats."
    ),
    (
        "business vocabulary and synonyms",
        "Column synonyms — always map these business terms to the exact column names shown:\n"
        "\n"
        "FUNDS MOBILIZED (column: funds_mobilized in amfi_fund_stats, total_mobilized in amfi_amc_stats):\n"
        "  'amount invested', 'money invested', 'capital invested', 'investment amount',\n"
        "  'subscriptions', 'purchases', 'money put in', 'money came in', 'capital inflow',\n"
        "  'sales', 'gross sales', 'gross inflow', 'fresh investments'.\n"
        "\n"
        "NET INFLOW (column: net_inflow):\n"
        "  'inflow', 'net inflow', 'net flow', 'net investment', 'net subscription',\n"
        "  'net money in', 'net capital'.\n"
        "  NOTE: net_inflow = funds_mobilized − redemption. Do NOT confuse with funds_mobilized.\n"
        "\n"
        "REDEMPTION (column: redemption):\n"
        "  'redemptions', 'withdrawals', 'money taken out', 'outflow', 'exits'.\n"
        "\n"
        "AUM (column: aum):\n"
        "  'AUM', 'assets under management', 'corpus', 'fund size', 'total assets',\n"
        "  'assets managed', 'industry AUM', 'category AUM'.\n"
        "  aum is a month-end snapshot, not cumulative. Use AVG(aum) for annual averages.\n"
        "\n"
        "FUND CATEGORY ALIASES (column: fund_category in amfi_fund_stats):\n"
        "  'large cap funds' or 'large cap'         → fund_category = 'Large Cap Fund'\n"
        "  'mid cap funds' or 'mid cap'             → fund_category = 'Mid Cap Fund'\n"
        "  'small cap funds' or 'small cap'         → fund_category = 'Small Cap Fund'\n"
        "  'flexi cap funds' or 'flexi cap'         → fund_category = 'Flexi Cap Fund'\n"
        "  'multi cap funds' or 'multi cap'         → fund_category = 'Multi Cap Fund'\n"
        "  'ELSS', 'tax saving', 'tax saver'        → fund_category = 'ELSS'\n"
        "  'liquid funds' or 'liquid'               → fund_category = 'Liquid Fund'\n"
        "  'index funds' or 'passive funds'         → fund_category = 'Index Funds'\n"
        "  'gold ETF' or 'gold funds'               → fund_category = 'GOLD ETF'\n"
        "  'balanced funds' or 'hybrid funds'       → fund_category IN ('Conservative Hybrid Fund',\n"
        "     'Balanced Hybrid Fund/Aggressive Hybrid Fund', 'Dynamic Asset Allocation/Balanced Advantage Fund')\n"
        "  'debt funds'                             → fund_category IN ('Short Duration Fund',\n"
        "     'Medium Duration Fund', 'Long Duration Fund', 'Corporate Bond Fund',\n"
        "     'Banking and PSU Fund', 'Gilt Fund', 'Credit Risk Fund', 'Dynamic Bond Fund',\n"
        "     'Floater Fund', 'Ultra Short Duration Fund', 'Low Duration Fund')\n"
        "\n"
        "DATE RANGE INTERPRETATION:\n"
        "  'from YYYY to YYYY' or 'between YYYY and YYYY' → period_year BETWEEN YYYY AND YYYY.\n"
        "  This means FULL MONTHLY DATA for every month in that range, NOT a yearly snapshot.\n"
        "  Always ORDER BY period_year, period_month when returning a date range.\n"
        "  Example: 'from 2020 to 2026' → WHERE period_year BETWEEN 2020 AND 2026\n"
        "           then GROUP BY period_year, period_month ORDER BY period_year, period_month.\n"
        "  'yearly' or 'annual' → GROUP BY period_year (collapse months into one row per year).\n"
        "  'monthly trend' or 'each month' → include both period_year and period_month in SELECT.\n"
    ),
    (
        "mf_scheme_master / mf_nav_history / mf_scheme_performance tables (per-scheme dataset)",
        "These three tables are a SEPARATE dataset from amfi_fund_stats/amfi_amc_stats above — "
        "per INDIVIDUAL mutual fund scheme (sourced from mfapi.in), not per fund_category. "
        "Never mix scheme_code-keyed tables with period_year/fund_category-keyed tables in the same query. "
        "\n"
        "mf_scheme_master: one row per scheme_code (the mfapi.in scheme identifier, a TEXT id, e.g. '119551'). "
        "Columns: scheme_code, scheme_name (full name incl. plan/option, e.g. 'HDFC Top 100 Fund - Direct Plan - Growth'), "
        "amc_name (the sponsoring AMC, e.g. 'HDFC Mutual Fund'), category, scheme_type, is_active. "
        "Growth/Dividend/Direct/Regular variants of the same fund are separate scheme_code rows. "
        "\n"
        "mf_nav_history: one row per (scheme_code, nav_date) — the full daily NAV time series back to 2000-01-01. "
        "Columns: scheme_code, nav_date, nav. Use this ONLY for historical/time-series NAV questions "
        "(e.g. 'NAV on a specific date', 'NAV trend over time'). ALWAYS filter by scheme_code — "
        "this table has ~35 million rows across all schemes. "
        "\n"
        "mf_scheme_performance: one row per scheme_code — PRECOMPUTED metrics, refreshed nightly. "
        "Prefer this table over aggregating mf_nav_history yourself whenever the question matches one of "
        "its columns: latest_nav, latest_nav_date, return_1d, return_1w, return_1m, return_3m, return_6m, "
        "return_1y (all simple % returns), return_3y_cagr, return_5y_cagr, return_10y_cagr (all CAGR %, "
        "NOT simple returns — compounded annually), all_time_return (return since the earliest available NAV, "
        "i.e. 'since launch'/'since inception' return), rolling_volatility, rolling_stddev (annualized, "
        "from daily NAV returns), nav_high_52w, nav_low_52w."
    ),
    (
        "mf_scheme_master / mf_scheme_performance JOIN pattern",
        "To answer a question naming a scheme by NAME (not scheme_code), JOIN mf_scheme_master to "
        "mf_scheme_performance on scheme_code, and filter with scheme_name ILIKE '%...%' (scheme names are "
        "long and exact-match rarely works — always use ILIKE with wildcards unless given an exact scheme_code). "
        "Example pattern: "
        "SELECT sm.scheme_name, sm.amc_name, sp.latest_nav, sp.return_1y FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.scheme_name ILIKE '%hdfc top 100%';"
    ),
    (
        "return vs CAGR terminology (mf_scheme_performance)",
        "Do not confuse simple returns with CAGR. return_1m/return_3m/return_6m/return_1y are SIMPLE "
        "percentage changes: ((latest_nav / nav_N_periods_ago) - 1) * 100. "
        "return_3y_cagr/return_5y_cagr/return_10y_cagr are COMPOUND ANNUAL growth rates: "
        "((latest_nav / nav_N_years_ago) ^ (1/N) - 1) * 100 — always fractional-exponent compounded, "
        "never a simple percentage change, even though the underlying formula pattern looks similar. "
        "'3 year return' or '3Y CAGR' or 'three year annualized return' all mean return_3y_cagr. "
        "'since launch return' / 'since inception return' / 'all time return' all mean all_time_return."
    ),
]

# ── Training examples (question → SQL) ────────────────────────────────────────

_EXAMPLES = [
    (
        "What is the total AUM of Large Cap Fund in 2024?",
        "SELECT SUM(aum) AS total_aum_crore FROM amfi_fund_stats "
        "WHERE fund_category = 'Large Cap Fund' AND period_year = 2024;",
    ),
    (
        "Show monthly net inflow for Mid Cap Fund in 2023",
        "SELECT period_month, net_inflow FROM amfi_fund_stats "
        "WHERE fund_category = 'Mid Cap Fund' AND period_year = 2023 "
        "ORDER BY period_month;",
    ),
    (
        "Which fund category had the highest AUM in March 2025?",
        "SELECT fund_category, aum FROM amfi_fund_stats "
        "WHERE period_year = 2025 AND period_month = 3 "
        "ORDER BY aum DESC LIMIT 1;",
    ),
    (
        "Total funds mobilized by ELSS in Q1 2024 (January to March)",
        "SELECT SUM(funds_mobilized) AS q1_mobilized FROM amfi_fund_stats "
        "WHERE fund_category = 'ELSS' AND period_year = 2024 AND period_month IN (1, 2, 3);",
    ),
    (
        "Year-over-year AUM growth for Small Cap Fund from 2020 to 2024",
        "SELECT period_year, SUM(aum) AS annual_aum FROM amfi_fund_stats "
        "WHERE fund_category = 'Small Cap Fund' AND period_year BETWEEN 2020 AND 2024 "
        "GROUP BY period_year ORDER BY period_year;",
    ),
    (
        "Average monthly net inflow for all equity funds in 2022",
        "SELECT fund_category, AVG(net_inflow) AS avg_monthly_inflow FROM amfi_fund_stats "
        "WHERE period_year = 2022 "
        "AND fund_category IN ('Multi Cap Fund','Large Cap Fund','Large & Mid Cap Fund',"
        "'Mid Cap Fund','Small Cap Fund','Dividend Yield Fund','Value Fund/Contra Fund',"
        "'Focused Fund','Sectoral/Thematic Funds','ELSS','Flexi Cap Fund') "
        "GROUP BY fund_category ORDER BY avg_monthly_inflow DESC;",
    ),
    (
        "Number of folios in Large Cap Fund every month in 2021",
        "SELECT period_month, no_of_folios FROM amfi_fund_stats "
        "WHERE fund_category = 'Large Cap Fund' AND period_year = 2021 "
        "ORDER BY period_month;",
    ),
    (
        "Total redemption across all fund categories in 2023",
        "SELECT SUM(redemption) AS total_redemption FROM amfi_fund_stats "
        "WHERE period_year = 2023;",
    ),
    (
        "Compare average AUM of Mid Cap Fund vs Large Cap Fund from 2021 to 2025",
        "SELECT fund_category, period_year, AVG(aum) AS avg_aum FROM amfi_fund_stats "
        "WHERE fund_category IN ('Mid Cap Fund', 'Large Cap Fund') "
        "AND period_year BETWEEN 2021 AND 2025 "
        "GROUP BY fund_category, period_year ORDER BY fund_category, period_year;",
    ),
    (
        "Top 5 fund categories by total AUM in 2024",
        "SELECT fund_category, SUM(aum) AS total_aum FROM amfi_fund_stats "
        "WHERE period_year = 2024 "
        "GROUP BY fund_category ORDER BY total_aum DESC LIMIT 5;",
    ),
    (
        "Monthly funds mobilized trend for Flexi Cap Fund in 2023",
        "SELECT period_month, funds_mobilized FROM amfi_fund_stats "
        "WHERE fund_category = 'Flexi Cap Fund' AND period_year = 2023 "
        "ORDER BY period_month;",
    ),
    (
        "Total net inflow in Q3 2022 for all fund categories",
        "SELECT fund_category, SUM(net_inflow) AS q3_net_inflow FROM amfi_fund_stats "
        "WHERE period_year = 2022 AND period_month IN (7, 8, 9) "
        "GROUP BY fund_category ORDER BY q3_net_inflow DESC;",
    ),
    (
        "Which month in 2024 had the highest funds mobilized for Sectoral/Thematic Funds?",
        "SELECT period_month, funds_mobilized FROM amfi_fund_stats "
        "WHERE fund_category = 'Sectoral/Thematic Funds' AND period_year = 2024 "
        "ORDER BY funds_mobilized DESC LIMIT 1;",
    ),
    (
        "Show average AUM for Index Funds from 2018 to 2024 by year",
        "SELECT period_year, AVG(aum) AS avg_aum FROM amfi_fund_stats "
        "WHERE fund_category = 'Index Funds' AND period_year BETWEEN 2018 AND 2024 "
        "GROUP BY period_year ORDER BY period_year;",
    ),
    (
        "Total number of schemes and folios in ELSS as of December 2024",
        "SELECT no_of_schemes, no_of_folios FROM amfi_fund_stats "
        "WHERE fund_category = 'ELSS' AND period_year = 2024 AND period_month = 12;",
    ),
    (
        "Quarterly net inflow for Large Cap Fund in 2023",
        "SELECT CASE WHEN period_month IN (1,2,3) THEN 'Q1' "
        "            WHEN period_month IN (4,5,6) THEN 'Q2' "
        "            WHEN period_month IN (7,8,9) THEN 'Q3' "
        "            ELSE 'Q4' END AS quarter, "
        "SUM(net_inflow) AS quarterly_net_inflow "
        "FROM amfi_fund_stats "
        "WHERE fund_category = 'Large Cap Fund' AND period_year = 2023 "
        "GROUP BY quarter ORDER BY quarter;",
    ),
    (
        "How many months of data are available for each year?",
        "SELECT period_year, COUNT(DISTINCT period_month) AS months_available "
        "FROM amfi_fund_stats GROUP BY period_year ORDER BY period_year;",
    ),
    (
        "Average AUM of Liquid Fund in 2024",
        "SELECT AVG(aum) AS avg_aum FROM amfi_fund_stats "
        "WHERE fund_category = 'Liquid Fund' AND period_year = 2024;",
    ),
    (
        "Net inflow for all fund categories in January 2025",
        "SELECT fund_category, net_inflow FROM amfi_fund_stats "
        "WHERE period_year = 2025 AND period_month = 1 "
        "ORDER BY net_inflow DESC;",
    ),
    (
        "Total AUM growth from 2019 to 2024 across all fund categories",
        "SELECT period_year, SUM(aum) AS total_industry_aum FROM amfi_fund_stats "
        "WHERE period_year BETWEEN 2019 AND 2024 "
        "GROUP BY period_year ORDER BY period_year;",
    ),
    (
        # Deliberately mirrors "What was the total industry AUM in December
        # 2015?" below, same "total industry AUM" phrasing, so the year
        # range is the only discriminating signal between the two tables —
        # without this pair, few-shot retrieval anchors on the phrase alone
        # and pulls amfi_amc_stats even when every filtered year is 2020+
        # (verified: this exact collision made "Total industry AUM by year
        # from 2020 to 2026" pick amfi_amc_stats deterministically, on every
        # retry, even with an explicit correction hint appended).
        "Total industry AUM by year from 2020 to 2026",
        "SELECT period_year, SUM(aum) AS total_industry_aum FROM amfi_fund_stats "
        "WHERE period_year BETWEEN 2020 AND 2026 "
        "GROUP BY period_year ORDER BY period_year;",
    ),
    # ── Vocabulary demonstration examples ──────────────────────────────────────
    (
        "How much amount was invested in Large Cap Fund in 2023?",
        "SELECT SUM(funds_mobilized) AS amount_invested_crore FROM amfi_fund_stats "
        "WHERE fund_category = 'Large Cap Fund' AND period_year = 2023;",
    ),
    (
        "How much money came in to Mid Cap Fund each month in 2024?",
        "SELECT period_month, funds_mobilized AS money_came_in FROM amfi_fund_stats "
        "WHERE fund_category = 'Mid Cap Fund' AND period_year = 2024 "
        "ORDER BY period_month;",
    ),
    (
        "What was the total inflow for Small Cap Fund in 2022?",
        "SELECT SUM(net_inflow) AS total_inflow FROM amfi_fund_stats "
        "WHERE fund_category = 'Small Cap Fund' AND period_year = 2022;",
    ),
    (
        "Show monthly inflow for ELSS from 2020 to 2026",
        "SELECT period_year, period_month, net_inflow FROM amfi_fund_stats "
        "WHERE fund_category = 'ELSS' AND period_year BETWEEN 2020 AND 2026 "
        "ORDER BY period_year, period_month;",
    ),
    (
        "What is the AUM of large cap funds in each month from 2020 to 2026?",
        "SELECT period_year, period_month, aum FROM amfi_fund_stats "
        "WHERE fund_category = 'Large Cap Fund' AND period_year BETWEEN 2020 AND 2026 "
        "ORDER BY period_year, period_month;",
    ),
    (
        "Show the total AUM of the mutual fund industry from 2020 to 2026 by month",
        "SELECT period_year, period_month, SUM(aum) AS industry_aum FROM amfi_fund_stats "
        "WHERE period_year BETWEEN 2020 AND 2026 "
        "GROUP BY period_year, period_month "
        "ORDER BY period_year, period_month;",
    ),
    (
        "What were the total subscriptions (amount invested) in tax saving funds in 2021?",
        "SELECT SUM(funds_mobilized) AS subscriptions_crore FROM amfi_fund_stats "
        "WHERE fund_category = 'ELSS' AND period_year = 2021;",
    ),
    (
        "Show redemptions vs inflow for liquid funds in 2023",
        "SELECT period_month, redemption, net_inflow FROM amfi_fund_stats "
        "WHERE fund_category = 'Liquid Fund' AND period_year = 2023 "
        "ORDER BY period_month;",
    ),
    (
        "What is the corpus (AUM) of index funds in December 2024?",
        "SELECT aum AS corpus_crore FROM amfi_fund_stats "
        "WHERE fund_category = 'Index Funds' AND period_year = 2024 AND period_month = 12;",
    ),
    (
        "Show net inflow and redemption for Gold ETF in each month of 2023",
        "SELECT period_month, net_inflow, redemption FROM amfi_fund_stats "
        "WHERE fund_category = 'GOLD ETF' AND period_year = 2023 "
        "ORDER BY period_month;",
    ),
    (
        "How much money came in versus went out for all equity funds in 2024?",
        "SELECT fund_category, SUM(funds_mobilized) AS money_came_in, "
        "SUM(redemption) AS withdrawals, SUM(net_inflow) AS net_flow "
        "FROM amfi_fund_stats "
        "WHERE fund_category IN ('Multi Cap Fund','Large Cap Fund','Large & Mid Cap Fund',"
        "'Mid Cap Fund','Small Cap Fund','Dividend Yield Fund','Value Fund/Contra Fund',"
        "'Focused Fund','Sectoral/Thematic Funds','ELSS','Flexi Cap Fund') "
        "AND period_year = 2024 "
        "GROUP BY fund_category ORDER BY net_flow DESC;",
    ),
    # ── amfi_amc_stats examples (pre-2020) ─────────────────────────────────────
    (
        "What was the total industry AUM in December 2015?",
        "SELECT aum FROM amfi_amc_stats "
        "WHERE period_year = 2015 AND period_month = 12 AND scheme_type = 'Total';",
    ),
    (
        "Show equity AUM by year from 2010 to 2019",
        "SELECT period_year, AVG(aum) AS avg_aum FROM amfi_amc_stats "
        "WHERE scheme_type = 'Equity' AND period_year BETWEEN 2010 AND 2019 "
        "GROUP BY period_year ORDER BY period_year;",
    ),
    (
        "Monthly AUM of Liquid/Money Market schemes in 2014",
        "SELECT period_month, aum FROM amfi_amc_stats "
        "WHERE scheme_type = 'Liquid/Money Market' AND period_year = 2014 "
        "ORDER BY period_month;",
    ),
    (
        "Total funds mobilized by the industry in 2013",
        "SELECT SUM(total_mobilized) AS total_mobilized_crore FROM amfi_amc_stats "
        "WHERE period_year = 2013 AND scheme_type = 'Total';",
    ),
    (
        "Compare equity vs income AUM in 2016",
        "SELECT scheme_type, AVG(aum) AS avg_aum FROM amfi_amc_stats "
        "WHERE scheme_type IN ('Equity', 'Income') AND period_year = 2016 "
        "GROUP BY scheme_type;",
    ),
    (
        "Net inflow for the industry each year from 2009 to 2019",
        "SELECT period_year, SUM(net_inflow) AS annual_net_inflow FROM amfi_amc_stats "
        "WHERE scheme_type = 'Total' AND period_year BETWEEN 2009 AND 2019 "
        "GROUP BY period_year ORDER BY period_year;",
    ),
    (
        "Which scheme type had the highest AUM share (aum_pct) in March 2018?",
        "SELECT scheme_type, aum_pct FROM amfi_amc_stats "
        "WHERE period_year = 2018 AND period_month = 3 "
        "ORDER BY aum_pct DESC LIMIT 1;",
    ),
    (
        "Gold ETF AUM trend from 2012 to 2018",
        "SELECT period_year, AVG(aum) AS avg_aum FROM amfi_amc_stats "
        "WHERE scheme_type = 'Gold ETF' AND period_year BETWEEN 2012 AND 2018 "
        "GROUP BY period_year ORDER BY period_year;",
    ),
    (
        "Industry AUM growth from 2009 to 2019 (full pre-2020 history)",
        "SELECT period_year, AVG(aum) AS avg_annual_aum FROM amfi_amc_stats "
        "WHERE scheme_type = 'Total' AND period_year BETWEEN 2009 AND 2019 "
        "GROUP BY period_year ORDER BY period_year;",
    ),
    (
        "Long-term industry AUM trend combining pre-2020 and post-2020 data from 2009 to 2024",
        "SELECT period_year, AVG(aum) AS avg_aum FROM amfi_amc_stats "
        "WHERE scheme_type = 'Total' AND period_year BETWEEN 2009 AND 2019 "
        "GROUP BY period_year "
        "UNION ALL "
        "SELECT period_year, SUM(aum) AS avg_aum FROM amfi_fund_stats "
        "WHERE period_year BETWEEN 2020 AND 2024 "
        "GROUP BY period_year "
        "ORDER BY period_year;",
    ),

    # ── mf_scheme_master / mf_nav_history / mf_scheme_performance examples ──
    (
        "What is the latest NAV of HDFC Top 100 Fund?",
        "SELECT sm.scheme_name, sm.amc_name, sp.latest_nav, sp.latest_nav_date "
        "FROM mf_scheme_master sm JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.scheme_name ILIKE '%hdfc top 100%' LIMIT 20;",
    ),
    (
        "What is the AMC name for scheme code 119551?",
        "SELECT scheme_code, scheme_name, amc_name FROM mf_scheme_master WHERE scheme_code = '119551';",
    ),
    (
        "What is the 1 year return of SBI Bluechip Fund?",
        "SELECT sm.scheme_name, sp.return_1y FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.scheme_name ILIKE '%sbi bluechip%' LIMIT 20;",
    ),
    (
        "Show the 3 year CAGR and 5 year CAGR for Axis Long Term Equity Fund",
        "SELECT sm.scheme_name, sp.return_3y_cagr, sp.return_5y_cagr FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.scheme_name ILIKE '%axis long term equity%' LIMIT 20;",
    ),
    (
        "What is the since launch return of scheme code 100027?",
        "SELECT scheme_code, all_time_return AS since_launch_return "
        "FROM mf_scheme_performance WHERE scheme_code = '100027';",
    ),
    (
        "List the top 10 schemes by 1 year return for HDFC Mutual Fund",
        "SELECT sm.scheme_name, sp.return_1y FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.amc_name ILIKE '%hdfc%' AND sp.return_1y IS NOT NULL "
        "ORDER BY sp.return_1y DESC LIMIT 10;",
    ),
    (
        "Compare the 3 month and 6 month returns of Parag Parikh Flexi Cap Fund",
        "SELECT sm.scheme_name, sp.return_3m, sp.return_6m FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.scheme_name ILIKE '%parag parikh flexi cap%' LIMIT 20;",
    ),
    (
        "What was the NAV of scheme code 119551 on 2024-01-15?",
        "SELECT scheme_code, nav_date, nav FROM mf_nav_history "
        "WHERE scheme_code = '119551' AND nav_date = '2024-01-15';",
    ),
    (
        "Show the NAV history of scheme code 100027 for the last 30 days",
        "SELECT nav_date, nav FROM mf_nav_history WHERE scheme_code = '100027' "
        "ORDER BY nav_date DESC LIMIT 30;",
    ),
    (
        "What is the 52 week high and low NAV for Mirae Asset Large Cap Fund?",
        "SELECT sm.scheme_name, sp.nav_high_52w, sp.nav_low_52w FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.scheme_name ILIKE '%mirae asset large cap%' LIMIT 20;",
    ),
    (
        "Which AMCs have the most active schemes?",
        "SELECT amc_name, COUNT(*) AS scheme_count FROM mf_scheme_master "
        "WHERE is_active = TRUE GROUP BY amc_name ORDER BY scheme_count DESC LIMIT 20;",
    ),
    (
        "List all scheme names and codes for Quant Mutual Fund",
        "SELECT scheme_code, scheme_name FROM mf_scheme_master "
        "WHERE amc_name ILIKE '%quant%' LIMIT 50;",
    ),

    # ── Fund-level comparison patterns ───────────────────────────────────────
    (
        "Compare the 1 year return of Parag Parikh Flexi Cap Fund vs Axis Bluechip Fund",
        "SELECT sm.scheme_name, sp.return_1y FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.scheme_name ILIKE '%parag parikh flexi cap%' "
        "   OR sm.scheme_name ILIKE '%axis bluechip%' "
        "ORDER BY sm.scheme_name;",
    ),
    (
        "Compare the 3 year CAGR of HDFC Top 100 Fund and SBI Bluechip Fund",
        "SELECT sm.scheme_name, sp.return_3y_cagr FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.scheme_name ILIKE '%hdfc top 100%' "
        "   OR sm.scheme_name ILIKE '%sbi bluechip%' "
        "ORDER BY sm.scheme_name;",
    ),
    (
        "What are the top 10 funds by 1 year return in the Large Cap category?",
        "SELECT sm.scheme_name, sm.amc_name, sp.return_1y FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.category ILIKE '%large cap%' AND sp.return_1y IS NOT NULL "
        "ORDER BY sp.return_1y DESC LIMIT 10;",
    ),
    (
        "Show the best performing Flexi Cap funds by 3 year CAGR",
        "SELECT sm.scheme_name, sm.amc_name, sp.return_3y_cagr FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.category ILIKE '%flexi cap%' AND sp.return_3y_cagr IS NOT NULL "
        "ORDER BY sp.return_3y_cagr DESC LIMIT 10;",
    ),
    (
        "Compare average 1 year return across AMCs for Large Cap schemes",
        "SELECT sm.amc_name, AVG(sp.return_1y) AS avg_return_1y, COUNT(*) AS scheme_count "
        "FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.category ILIKE '%large cap%' AND sp.return_1y IS NOT NULL "
        "GROUP BY sm.amc_name ORDER BY avg_return_1y DESC LIMIT 20;",
    ),
    (
        "Rank all ELSS schemes by 5 year CAGR",
        "SELECT sm.scheme_name, sm.amc_name, sp.return_5y_cagr FROM mf_scheme_master sm "
        "JOIN mf_scheme_performance sp ON sp.scheme_code = sm.scheme_code "
        "WHERE sm.category ILIKE '%elss%' AND sp.return_5y_cagr IS NOT NULL "
        "ORDER BY sp.return_5y_cagr DESC;",
    ),
]


def main(reset: bool) -> None:
    configure_logging(level="INFO", fmt="console")

    vn = build_vanna_agent(
        anthropic_api_key=settings.openai_api_key,
        postgres_url=settings.postgres_url,
    )

    if reset:
        log.info("train.resetting_training_data")
        try:
            existing = vn.get_training_data()
            if existing is not None and not existing.empty:
                for tid in existing["id"]:
                    vn.remove_training_data(id=tid)
                log.info("train.cleared", count=len(existing))
        except Exception as exc:
            log.warning("train.reset_failed", error=str(exc))

    # 1 — DDL (all five tables)
    all_ddls = [
        ("amfi_fund_stats", DDL),
        ("amfi_amc_stats", AMC_DDL),
        ("mf_scheme_master", MF_SCHEME_MASTER_DDL),
        ("mf_nav_history", MF_NAV_HISTORY_DDL),
        ("mf_scheme_performance", MF_SCHEME_PERFORMANCE_DDL),
    ]
    for table, ddl in all_ddls:
        log.info("train.ddl", table=table)
        vn.train(ddl=ddl)

    # 2 — Documentation
    for subject, doc in _DOCS:
        log.info("train.doc", subject=subject)
        vn.train(documentation=doc)

    log.info("train.doc", subject="mf_scheme_master.category reference (generated)")
    vn.train(documentation=_mf_scheme_category_doc())

    # 3 — Question–SQL pairs
    for i, (question, sql) in enumerate(_EXAMPLES, 1):
        log.info("train.example", n=i, question=question[:60])
        vn.train(question=question, sql=sql)

    total_docs = len(_DOCS) + 1  # +1 for the generated category-reference doc
    log.info("train.done",
             ddl=len(all_ddls),
             docs=total_docs,
             examples=len(_EXAMPLES))
    print(f"\nTraining complete: {len(all_ddls)} DDLs + {total_docs} docs + {len(_EXAMPLES)} SQL examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear all existing training data before re-training (use after schema changes)",
    )
    args = parser.parse_args()
    main(reset=args.reset)
