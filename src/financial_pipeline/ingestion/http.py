from typing import Any

import httpx

from financial_pipeline.ingestion.base import BaseIngester
from financial_pipeline.utils.retry import with_retry


class HttpIngester(BaseIngester):
    """Ingests data from an HTTP/REST endpoint."""

    def __init__(self, source_name: str, base_url: str, headers: dict[str, str] | None = None) -> None:
        super().__init__(source_name)
        self.base_url = base_url
        self._headers = headers or {}

    @with_retry(attempts=3, wait_seconds=2)
    def fetch(self, path: str = "", params: dict[str, Any] | None = None) -> Any:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        with httpx.Client(headers=self._headers, timeout=30) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.json()

    @with_retry(attempts=3, wait_seconds=2)
    def fetch_raw(self, path: str = "", params: dict[str, Any] | None = None) -> bytes:
        """Fetch raw bytes — use this for text, CSV, XLS, PDF endpoints."""
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        with httpx.Client(headers=self._headers, timeout=60, follow_redirects=True) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
            return response.content
