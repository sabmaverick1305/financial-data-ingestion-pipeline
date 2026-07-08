"""Thin client for api.mfapi.in.

Self-contained: deliberately not built on ingestion/http.py's HttpIngester
(that's shared AMFI-PDF-pipeline infra) — this ingestion source is a
different dataset and stays independent.

Each call is a single HTTP request with no built-in retry — retry/backoff
lives in sync.py's per-scheme loop so every attempt's outcome can be recorded
against mf_scheme_sync_status.
"""

from __future__ import annotations

import httpx

MFAPI_BASE_URL = "https://api.mfapi.in"


class RateLimitedError(Exception):
    """Raised when mfapi.in returns HTTP 429."""


class MFApiClient:
    def __init__(self, base_url: str = MFAPI_BASE_URL, timeout: float = 30.0) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    def fetch_nav_history(self, scheme_code: str, start_date: str, end_date: str) -> dict:
        """GET /mf/{scheme_code}?startDate=...&endDate=... — returns the parsed JSON body.

        mfapi's `meta` block carries fund_house/scheme_type/scheme_category/
        isin_*; `data` is a list of {date: "DD-MM-YYYY", nav: "123.4567"},
        most-recent first.
        """
        url = f"{self._base_url}/mf/{scheme_code}"
        with httpx.Client(timeout=self._timeout) as client:
            response = client.get(url, params={"startDate": start_date, "endDate": end_date})
        if response.status_code == 429:
            raise RateLimitedError(f"scheme {scheme_code}: rate limited (429)")
        response.raise_for_status()
        return response.json()
