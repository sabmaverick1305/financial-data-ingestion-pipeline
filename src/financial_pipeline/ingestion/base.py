from abc import ABC, abstractmethod
from typing import Any

import structlog

log = structlog.get_logger()


class BaseIngester(ABC):
    """Abstract base for all data ingesters."""

    def __init__(self, source_name: str) -> None:
        self.source_name = source_name
        self._log = log.bind(source=source_name)

    @abstractmethod
    def fetch(self, **kwargs: Any) -> Any:
        """Fetch raw data from the source."""
        ...

    def ingest(self, **kwargs: Any) -> Any:
        self._log.info("ingestion.started")
        try:
            data = self.fetch(**kwargs)
            self._log.info("ingestion.completed", records=len(data) if hasattr(data, "__len__") else "?")
            return data
        except Exception:
            self._log.exception("ingestion.failed")
            raise
