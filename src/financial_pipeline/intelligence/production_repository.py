"""Production SQL repository for FIES reasoning capabilities."""
from __future__ import annotations
from sqlalchemy import text
from sqlalchemy.engine import Engine
from financial_pipeline.intelligence.category_ontology import CategoryOntology
from financial_pipeline.intelligence.data_quality import FundDataQualityGate

class ReasoningProductionRepository:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._categories = CategoryOntology()
        self._quality = FundDataQualityGate()

    def discover_categories(self) -> list[str]:
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT DISTINCT category
                FROM mf_scheme_master
                WHERE is_active = TRUE
                  AND category IS NOT NULL
                  AND BTRIM(category) <> ''
            """)).all()
        canonical = [
            self._categories.canonicalize(str(row[0]))
            for row in rows
        ]
        return sorted({value for value in canonical if value})

    def discover_funds(self, *, category: str | None = None, limit: int = 20) -> list[dict]:
        where = "WHERE m.is_active = TRUE AND m.category IS NOT NULL"
        params: dict[str, object] = {"limit": max(limit * 20, 500)}
        if category:
            where += " AND LOWER(COALESCE(m.category, '')) LIKE LOWER(:category_pattern)"
            params["category_pattern"] = f"%{category}%"
        sql = f"""
            SELECT m.scheme_code, m.scheme_name, m.amc_name, m.category, m.scheme_type,
                   MIN(n.nav_date) AS inception_date,
                   p.latest_nav, p.latest_nav_date, p.return_1y, p.return_3y_cagr,
                   p.return_5y_cagr, p.return_10y_cagr, p.rolling_volatility
            FROM mf_scheme_master m
            LEFT JOIN mf_nav_history n ON n.scheme_code = m.scheme_code
            LEFT JOIN mf_scheme_performance p ON p.scheme_code = m.scheme_code
            {where}
            GROUP BY m.scheme_code, m.scheme_name, m.amc_name, m.category, m.scheme_type,
                     p.latest_nav, p.latest_nav_date, p.return_1y, p.return_3y_cagr,
                     p.return_5y_cagr, p.return_10y_cagr, p.rolling_volatility
            ORDER BY p.return_3y_cagr DESC NULLS LAST, p.return_1y DESC NULLS LAST, m.scheme_name ASC
            LIMIT :limit
        """
        with self._engine.connect() as conn:
            rows = [dict(row) for row in conn.execute(text(sql), params).mappings().all()]
        filtered: list[dict] = []
        for row in rows:
            raw_category = str(row.get("category") or "")
            canonical = self._categories.canonicalize(raw_category)
            if canonical is None:
                continue
            row["raw_category"] = raw_category
            row["category"] = canonical
            quality = self._quality.validate_candidate(row)
            if not quality.valid:
                continue
            row["data_quality"] = {"valid": True, "issues": []}
            filtered.append(row)
            if len(filtered) >= limit:
                break
        return filtered

    def nav_history(self, scheme_code: str) -> list[tuple]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                text("SELECT nav_date, nav FROM mf_nav_history WHERE scheme_code=:code ORDER BY nav_date"),
                {"code": scheme_code},
            ).all()
        return [(row[0], float(row[1])) for row in rows]

    def performance(self, scheme_code: str) -> dict | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                text("""
                    SELECT p.*, m.scheme_name, m.amc_name, m.category
                    FROM mf_scheme_performance p
                    JOIN mf_scheme_master m ON m.scheme_code=p.scheme_code
                    WHERE p.scheme_code=:code
                """), {"code": scheme_code}
            ).mappings().first()
        return dict(row) if row else None

    def peer_performance(self, *, category: str, limit: int = 100) -> list[dict]:
        with self._engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT m.scheme_code, m.scheme_name, m.amc_name, m.category,
                       p.return_1y, p.return_3y_cagr, p.return_5y_cagr,
                       p.return_10y_cagr, p.rolling_volatility
                FROM mf_scheme_master m
                JOIN mf_scheme_performance p ON p.scheme_code=m.scheme_code
                WHERE m.is_active=TRUE AND LOWER(COALESCE(m.category, '')) LIKE LOWER(:category_pattern)
                ORDER BY p.return_3y_cagr DESC NULLS LAST
                LIMIT :limit
            """), {"category_pattern": f"%{category}%", "limit": limit}).mappings().all()
        return [dict(row) for row in rows]

    def latest_category_facts(self, *, metric: str, category: str | None = None) -> list[dict]:
        if metric not in {"aum", "net_inflow", "redemption", "funds_mobilized", "avg_aum"}:
            raise ValueError(f"unsupported category metric: {metric}")
        where = f"WHERE afs.{metric} IS NOT NULL"
        params: dict[str, object] = {}
        if category:
            where += " AND LOWER(afs.fund_category) LIKE LOWER(:category_pattern)"
            params["category_pattern"] = f"%{category}%"
        sql = f"""
            SELECT DISTINCT ON (afs.fund_category)
                   afs.fund_category, afs.period_year, afs.period_month, afs.{metric} AS value,
                   afs.source_document_id, dm.original_url, dm.s3_raw_key, dm.s3_processed_key
            FROM amfi_fund_stats afs
            LEFT JOIN document_metadata dm ON dm.document_id=afs.source_document_id
            {where}
            ORDER BY afs.fund_category, afs.period_year DESC, afs.period_month DESC
        """
        with self._engine.connect() as conn:
            return [dict(row) for row in conn.execute(text(sql), params).mappings().all()]
