"""Cheap per-page profiling used by `DocumentProfiler`.

Takes a live PyMuPDF `Page` — an internal collaborator, not part of the
preflight -> routing boundary (only `DocumentProfiler`'s `DocumentProfile`
output crosses that boundary). Mirrors the per-page signal
`financial_pipeline.processing.extractor.TextExtractor._has_text_layer`
already samples: embedded fonts or a minimum word count means "has text".
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import structlog

from fies_parser.preflight.models import PageProfile

log = structlog.get_logger()

DEFAULT_MIN_WORDS_PER_PAGE = 10


class PageProfiler:
    def __init__(self, min_words_per_page: int = DEFAULT_MIN_WORDS_PER_PAGE) -> None:
        self._min_words_per_page = min_words_per_page

    def profile(self, page: Any, page_number: int) -> PageProfile:
        word_count = len((page.get_text("text") or "").split())
        has_text = bool(page.get_fonts()) or word_count >= self._min_words_per_page
        image_count = len(page.get_images())

        return PageProfile(
            page_number=page_number,
            has_text=has_text,
            word_count=word_count,
            image_count=image_count,
            table_count=self._count_tables(page, page_number),
            drawing_count=len(page.get_drawings()),
        )

    def _count_tables(self, page: Any, page_number: int) -> int:
        try:
            # PyMuPDF's table finder prints an advisory line to stdout on
            # every call — harmless but noisy at document scale.
            with contextlib.redirect_stdout(io.StringIO()):
                return len(page.find_tables().tables)
        except Exception as exc:
            log.debug("page_profiler.table_detection_failed", page_number=page_number, error=str(exc))
            return 0
