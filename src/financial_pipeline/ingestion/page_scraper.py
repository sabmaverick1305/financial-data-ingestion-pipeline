from __future__ import annotations

import re
from html.parser import HTMLParser
from pathlib import PurePosixPath
from urllib.parse import urljoin, urlparse

import httpx
import structlog

log = structlog.get_logger()

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,*/*",
}

_RSC_HEADERS = {
    **_DEFAULT_HEADERS,
    "RSC": "1",
    "Next-Router-State-Tree": "%5B%22%22%2C%7B%7D%5D",
    "Next-Url": "/",
}

# Matches http(s) URLs ending in a file extension of interest
_FILE_URL_RE = re.compile(
    r'https?://[^\s"\'\\<>]+\.(?:pdf|xls|xlsx)',
    re.IGNORECASE,
)


class _LinkExtractor(HTMLParser):
    """Walks static HTML and collects <a href> values matching given extensions."""

    def __init__(self, base_url: str, extensions: tuple[str, ...]) -> None:
        super().__init__()
        self._base = base_url
        self._exts = tuple(e.lower() for e in extensions)
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        for name, value in attrs:
            if name == "href" and value:
                url = urljoin(self._base, value.strip())
                if any(url.lower().split("?")[0].endswith(ext) for ext in self._exts):
                    self.links.append(url)


class PageScraper:
    """
    Fetches an HTML page and extracts file download links by extension.

    Handles two rendering modes:
      - Static HTML / server-rendered pages  →  HTML <a href> parsing
      - Next.js (React Server Components)    →  RSC payload regex extraction
        Detected by presence of `/_next/` in page HTML.
    """

    FILE_EXTENSIONS: tuple[str, ...] = (".pdf", ".xls", ".xlsx")

    def __init__(self, timeout: int = 30, extra_headers: dict[str, str] | None = None) -> None:
        self._timeout = timeout
        self._headers = {**_DEFAULT_HEADERS, **(extra_headers or {})}
        self._log = log.bind(component="page_scraper")

    # ------------------------------------------------------------------

    def find_links(
        self,
        page_url: str,
        extensions: tuple[str, ...] | None = None,
    ) -> list[str]:
        """Return deduplicated absolute file URLs found on the page."""
        exts = extensions or self.FILE_EXTENSIONS

        self._log.info("scraper.fetching_page", url=page_url)
        with httpx.Client(headers=self._headers, timeout=self._timeout, follow_redirects=True) as client:
            html_resp = client.get(page_url)
            html_resp.raise_for_status()
            html = html_resp.text

        links: list[str] = []

        # --- Try static HTML parsing first ---
        extractor = _LinkExtractor(base_url=page_url, extensions=exts)
        extractor.feed(html)
        links.extend(extractor.links)

        # --- If Next.js detected, also fetch RSC payload ---
        if "/_next/" in html:
            self._log.info("scraper.nextjs_detected", url=page_url)
            rsc_links = self._fetch_rsc_links(page_url, exts, client_headers=self._headers)
            links.extend(rsc_links)

        unique = _dedupe(links)
        self._log.info("scraper.links_found", count=len(unique), extensions=exts)
        return unique

    def _fetch_rsc_links(
        self,
        page_url: str,
        extensions: tuple[str, ...],
        client_headers: dict[str, str],
    ) -> list[str]:
        """Hit the Next.js RSC endpoint and extract file URLs via regex."""
        rsc_headers = {**client_headers, **_RSC_HEADERS, "Next-Url": urlparse(page_url).path}
        with httpx.Client(headers=rsc_headers, timeout=self._timeout, follow_redirects=True) as client:
            resp = client.get(page_url)
            resp.raise_for_status()

        raw_urls = _FILE_URL_RE.findall(resp.text)
        # Keep only URLs whose extension matches what was requested
        return [u for u in raw_urls if any(u.lower().split("?")[0].endswith(ext) for ext in extensions)]

    # ------------------------------------------------------------------

    def download(self, url: str) -> bytes:
        """Download a single file and return raw bytes."""
        filename = self.filename_from_url(url)
        self._log.info("download.started", url=url, filename=filename)
        with httpx.Client(headers=self._headers, timeout=120, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
        self._log.info("download.completed", filename=filename, bytes=len(resp.content))
        return resp.content

    # ------------------------------------------------------------------

    @staticmethod
    def filename_from_url(url: str) -> str:
        return PurePosixPath(urlparse(url).path).name

    @staticmethod
    def content_type(filename: str) -> str:
        ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
        return {
            "pdf": "application/pdf",
            "xls": "application/vnd.ms-excel",
            "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        }.get(ext, "application/octet-stream")


_MONTHLY_RE = re.compile(r"^am([a-z]{3})(\d{4})repo\.", re.IGNORECASE)
_QUARTERLY_RE = re.compile(r"^aqu-vol(\d+)-issue([IVXLCDM]+)\.", re.IGNORECASE)

_MONTH_MAP = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_ROMAN_TO_QUARTER = {"I": "Q1", "II": "Q2", "III": "Q3", "IV": "Q4"}


def classify_filename(filename: str) -> str:
    """Return 'monthly', 'quarterly', or 'unknown' based on the filename pattern."""
    name = PurePosixPath(filename).name
    if _MONTHLY_RE.match(name):
        return "monthly"
    if _QUARTERLY_RE.match(name):
        return "quarterly"
    return "unknown"


def parse_filename_metadata(filename: str) -> dict:
    """Extract period metadata from a filename. Returns a partial dict suitable
    for merging into document_metadata fields."""
    name = PurePosixPath(filename).name
    m = _MONTHLY_RE.match(name)
    if m:
        month_str, year_str = m.group(1).lower(), m.group(2)
        return {
            "period_month": _MONTH_MAP.get(month_str),
            "period_year": int(year_str),
        }
    m = _QUARTERLY_RE.match(name)
    if m:
        vol, roman = m.group(1), m.group(2).upper()
        return {
            "volume": vol,
            "issue": roman,
            "period_quarter": _ROMAN_TO_QUARTER.get(roman),
        }
    return {}


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    return [x for x in items if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
