import pandas as pd
import structlog
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from financial_pipeline.storage.base import BaseStorage

log = structlog.get_logger()


class DatabaseStorage(BaseStorage):
    """SQLAlchemy-backed relational database storage."""

    def __init__(self, connection_url: str) -> None:
        self._engine: Engine = create_engine(connection_url, pool_pre_ping=True)
        self._log = log.bind(backend="database")

    def save(self, df: pd.DataFrame, table: str, if_exists: str = "append") -> int:
        rows = len(df)
        df.to_sql(table, con=self._engine, if_exists=if_exists, index=False)
        self._log.info("storage.saved", table=table, rows=rows)
        return rows

    def load(self, table: str) -> pd.DataFrame:
        with self._engine.connect() as conn:
            df = pd.read_sql(text(f"SELECT * FROM {table}"), conn)  # noqa: S608
        self._log.info("storage.loaded", table=table, rows=len(df))
        return df
