"""Discover authoritative fund-document links from an official source page."""
from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import urljoin, urlparse

import httpx


class AuthoritativeSourceDiscoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class DiscoveredAuthoritativeDocument:
    url: str
    document_type: str
    anchor_text: str


class _AnchorParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self._href: str | None = None
        self._text: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag.lower() != "a":
            return
        attributes = dict(attrs)
        self._href = attributes.get("href")
        self._text = []

    def handle_data(self, data: str) -> None:
        if self._href is not None:
            self._text.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._href is not None:
            self.links.append((self._href, " ".join(self._text).strip()))
            self._href = None
            self._text = []


class AuthoritativeSourcePageDiscoverer:
    _TYPE_KEYWORDS = {
        "scheme_information_document": (
            "scheme information document",
            "sid",
            "scheme document",
            "offer document",
        ),
        "fund_prospectus": (
            "prospectus",
            "offer document",
        ),
        "key_information_memorandum": (
            "key information memorandum",
            "kim",
        ),
        "fund_fact_sheet": (
            "factsheet",
            "fact sheet",
            "monthly factsheet",
        ),
        "fund_strategy_document": (
            "investment strategy",
            "investment approach",
            "strategy document",
        ),
        "portfolio_disclosure": (
            "portfolio disclosure",
            "portfolio",
            "holding",
        ),
    }

    def discover(
        self,
        *,
        source_page_url: str,
        authoritative_domain: str,
        scheme_family_key: str,
    ) -> list[DiscoveredAuthoritativeDocument]:
        parsed = urlparse(source_page_url)
        hostname = (parsed.hostname or "").lower()
        allowed = authoritative_domain.lower().strip()
        if parsed.scheme != "https" or not (
            hostname == allowed or hostname.endswith("." + allowed)
        ):
            raise ValueError("source page must be HTTPS on the authoritative domain")

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/151.0 Safari/537.36"
            ),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-IN,en;q=0.9",
        }
        try:
            response = httpx.get(
                source_page_url,
                timeout=30,
                follow_redirects=True,
                headers=headers,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise AuthoritativeSourceDiscoveryError(
                f"official source page could not be fetched: {source_page_url}: {exc}"
            ) from exc

        parser = _AnchorParser()
        parser.feed(response.text)
        scheme_tokens = {
            token
            for token in scheme_family_key.lower().replace("&", " and ").split()
            if len(token) > 2
        }

        discovered: list[DiscoveredAuthoritativeDocument] = []
        seen: set[tuple[str, str]] = set()
        for href, anchor_text in parser.links:
            absolute = urljoin(source_page_url, href)
            link_host = (urlparse(absolute).hostname or "").lower()
            if not (link_host == allowed or link_host.endswith("." + allowed)):
                continue

            haystack = f"{anchor_text} {absolute}".lower().replace("-", " ")
            scheme_overlap = sum(token in haystack for token in scheme_tokens)
            if scheme_tokens and scheme_overlap < max(1, len(scheme_tokens) // 2):
                continue

            for document_type, keywords in self._TYPE_KEYWORDS.items():
                if any(keyword in haystack for keyword in keywords):
                    key = (absolute, document_type)
                    if key not in seen:
                        seen.add(key)
                        discovered.append(
                            DiscoveredAuthoritativeDocument(
                                url=absolute,
                                document_type=document_type,
                                anchor_text=anchor_text,
                            )
                        )
        return discovered
