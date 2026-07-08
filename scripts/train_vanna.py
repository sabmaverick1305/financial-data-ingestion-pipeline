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
from financial_pipeline.text_to_sql.vanna_agent import build_vanna_agent

log = structlog.get_logger()

# ── Documentation strings ──────────────────────────────────────────────────────

_DOCS = [
    (
        "amfi_fund_stats table",
        "The amfi_fund_stats table stores one row per (period_year, period_month, fund_category). "
        "It is populated from AMFI monthly reports (2009–2026). "
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
        "The amfi_amc_stats table stores pre-2020 AMFI monthly report data (2009–2019). "
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

    # 1 — DDL (both tables)
    log.info("train.ddl", table="amfi_fund_stats")
    vn.train(ddl=DDL)
    log.info("train.ddl", table="amfi_amc_stats")
    vn.train(ddl=AMC_DDL)

    # 2 — Documentation
    for subject, doc in _DOCS:
        log.info("train.doc", subject=subject)
        vn.train(documentation=doc)

    # 3 — Question–SQL pairs
    for i, (question, sql) in enumerate(_EXAMPLES, 1):
        log.info("train.example", n=i, question=question[:60])
        vn.train(question=question, sql=sql)

    log.info("train.done",
             ddl=2,
             docs=len(_DOCS),
             examples=len(_EXAMPLES))
    print(f"\nTraining complete: 2 DDLs + {len(_DOCS)} docs + {len(_EXAMPLES)} SQL examples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset", action="store_true",
        help="Clear all existing training data before re-training (use after schema changes)",
    )
    args = parser.parse_args()
    main(reset=args.reset)
