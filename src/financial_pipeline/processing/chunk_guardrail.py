"""ChunkGuardrail — quality gate between chunking and S3 upload.

Runs immediately after chunk_text() and before any chunk is written to S3 or
the database.  If the guardrail blocks, the document reverts to
tables_extracted so the next worker pick-up can retry with a better strategy.

Checks
------
1. min_chunks          : at least MIN_CHUNKS produced (extraction probably failed)
2. table_header_coverage: table chunks must carry their column header
3. fragment_rate       : chunks must not start mid-row or mid-sentence
4. self_containment    : chunk must be interpretable without neighbouring chunks
5. empty_chunk_rate    : too many near-empty chunks signals extractor noise
6. text_coverage       : chunks must cover a reasonable fraction of source text

Outcome
-------
  PASS    — all checks green; proceed to S3 upload
  WARN    — non-critical issues; log and proceed, do not block
  BLOCK   — one or more checks failed; revert document, log for human review
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

import structlog

log = structlog.get_logger()

# ── Thresholds ────────────────────────────────────────────────────────────────

MIN_CHUNKS              = 2      # fewer than this → extraction probably failed
MAX_FRAGMENT_RATE       = 0.20   # >20% fragments → block
MIN_TABLE_HEADER_RATE   = 0.80   # <80% table chunks have header → block
MIN_TEXT_COVERAGE       = 0.40   # chunks cover <40% of source text → block
MAX_EMPTY_CHUNK_RATE    = 0.15   # >15% near-empty chunks → warn
MIN_SELF_CONTAIN_RATE   = 0.70   # <70% self-contained → warn

# A table chunk: has 4+ pipe chars
_TABLE_RE   = re.compile(r"\|.*\|.*\|.*\|")
# A header separator row: |---|---|...
_SEP_RE     = re.compile(r"\|[-| :]+\|")
# Fragment: chunk starts with a continuation cell or bare number
_FRAGMENT_RE = re.compile(r"^\s*(\|[^-]|\d[\d,\.]*\s*\|)")


class Outcome(str, Enum):
    PASS  = "pass"
    WARN  = "warn"
    BLOCK = "block"


@dataclass
class ChunkGuardrailResult:
    outcome:        Outcome
    errors:         list[str] = field(default_factory=list)   # blocking
    warnings:       list[str] = field(default_factory=list)   # non-blocking
    stats:          dict      = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.outcome != Outcome.BLOCK

    def log(self, doc_id: str, file_name: str) -> None:
        bound = log.bind(document_id=doc_id, file_name=file_name, outcome=self.outcome.value)
        if self.errors:
            bound.warning("chunk_guardrail.blocked", errors=self.errors, stats=self.stats)
        elif self.warnings:
            bound.info("chunk_guardrail.warned", warnings=self.warnings, stats=self.stats)
        else:
            bound.debug("chunk_guardrail.passed", stats=self.stats)


# ── Individual checks ─────────────────────────────────────────────────────────

def _check_min_chunks(chunks: list[dict]) -> str | None:
    if len(chunks) < MIN_CHUNKS:
        return f"only {len(chunks)} chunk(s) produced — extraction likely failed"
    return None


def _is_fragment(text: str) -> bool:
    """A chunk is a fragment if it starts with a table continuation cell
    (pipe char) but contains no header separator row (|---|).
    A chunk that starts with | but includes |---| is a complete table chunk
    (header + data rows) — not a fragment.
    """
    if not text.startswith("|"):
        return False
    if _SEP_RE.search(text):
        return False   # has a separator row → proper table chunk, not a fragment
    # Starts with | and has no separator: bare continuation row
    return True


def _check_fragments(chunks: list[dict]) -> tuple[float, str | None]:
    if not chunks:
        return 0.0, None
    fragments = [c for c in chunks if _is_fragment(c["text"])]
    rate = len(fragments) / len(chunks)
    if rate > MAX_FRAGMENT_RATE:
        examples = [c["text"][:60] for c in fragments[:2]]
        return rate, f"fragment rate {rate:.0%} > {MAX_FRAGMENT_RATE:.0%}: {examples}"
    return rate, None


def _check_table_header_coverage(chunks: list[dict]) -> tuple[float, str | None]:
    table_chunks = [c for c in chunks if _TABLE_RE.search(c["text"])]
    if not table_chunks:
        return 1.0, None   # no tables → not applicable

    with_header = [c for c in table_chunks if _SEP_RE.search(c["text"])]
    rate = len(with_header) / len(table_chunks)

    if rate < MIN_TABLE_HEADER_RATE:
        return rate, (
            f"only {rate:.0%} of {len(table_chunks)} table chunks carry a header row "
            f"(need >= {MIN_TABLE_HEADER_RATE:.0%}) — column labels will be missing from LLM context"
        )
    return rate, None


def _check_text_coverage(chunks: list[dict], source_text: str) -> tuple[float, str | None]:
    if not source_text.strip():
        return 1.0, None
    covered = sum(len(c["text"]) for c in chunks)
    rate    = covered / len(source_text)
    if rate < MIN_TEXT_COVERAGE:
        return rate, f"chunks cover only {rate:.0%} of source text — significant content may be lost"
    return rate, None


def _check_empty_chunks(chunks: list[dict]) -> tuple[float, str | None]:
    if not chunks:
        return 0.0, None
    empty = [c for c in chunks if len(c["text"].split()) < 5]
    rate  = len(empty) / len(chunks)
    if rate > MAX_EMPTY_CHUNK_RATE:
        return rate, f"{rate:.0%} of chunks have fewer than 5 words — extractor noise"
    return rate, None


def _check_broken_rows(chunks: list[dict]) -> tuple[float, str | None]:
    """Detect chunks that contain table content but end mid-cell.

    A complete table row in GitHub-flavoured markdown ends with '|'.
    If the last non-empty line of a table chunk does NOT end with '|',
    the chunker cut through a cell — numbers and text will be split
    across chunk boundaries, making the context uninterpretable.
    """
    table_chunks = [c for c in chunks if _TABLE_RE.search(c["text"])]
    if not table_chunks:
        return 0.0, None

    broken = []
    for c in table_chunks:
        lines = [l for l in c["text"].splitlines() if l.strip()]
        if not lines:
            continue
        last = lines[-1].rstrip()
        # last line of a table chunk should close with |
        if "|" in last and not last.endswith("|"):
            broken.append(c)

    rate = len(broken) / len(table_chunks)
    if rate > MAX_FRAGMENT_RATE:
        examples = [c["text"][-50:] for c in broken[:2]]
        return rate, (
            f"{rate:.0%} of table chunks end mid-cell (no closing '|') — "
            f"numbers and text are split across chunk boundaries: {examples}"
        )
    return rate, None


def _check_self_containment(chunks: list[dict]) -> tuple[float, str | None]:
    """A chunk is self-contained if it is either:
    - A prose chunk (no leading pipe)
    - A table chunk that carries its header separator (|---|)
    A chunk that starts with | but has no separator is an orphaned data row.
    """
    if not chunks:
        return 1.0, None
    bad = [c for c in chunks if _is_fragment(c["text"])]
    rate = 1.0 - len(bad) / len(chunks)
    if rate < MIN_SELF_CONTAIN_RATE:
        return rate, f"self-containment {rate:.0%} < {MIN_SELF_CONTAIN_RATE:.0%} — chunks need more context"
    return rate, None


# ── Main guardrail ────────────────────────────────────────────────────────────

def check_chunks(
    chunks:      list[dict],
    source_text: str,
    doc_id:      str  = "",
    file_name:   str  = "",
) -> ChunkGuardrailResult:
    """Run all quality checks and return a ChunkGuardrailResult.

    Call this between chunk_text() and the S3 upload / DB insert.
    If result.passed is False, revert the document status and skip upload.
    """
    errors:   list[str] = []
    warnings: list[str] = []

    # 1. Minimum chunk count (hard fail)
    err = _check_min_chunks(chunks)
    if err:
        errors.append(err)

    # 2. Fragment rate (hard fail)
    fragment_rate, err = _check_fragments(chunks)
    if err:
        errors.append(err)

    # 3. Table header coverage (hard fail — this was the root cause bug)
    header_rate, err = _check_table_header_coverage(chunks)
    if err:
        errors.append(err)

    # 4. Text coverage (hard fail)
    coverage_rate, err = _check_text_coverage(chunks, source_text)
    if err:
        errors.append(err)

    # 4b. Broken table rows — chunks ending mid-cell (hard fail)
    broken_row_rate, err = _check_broken_rows(chunks)
    if err:
        errors.append(err)

    # 5. Empty chunk rate (warning only)
    empty_rate, warn = _check_empty_chunks(chunks)
    if warn:
        warnings.append(warn)

    # 6. Self-containment (warning only)
    self_contain_rate, warn = _check_self_containment(chunks)
    if warn:
        warnings.append(warn)

    stats = {
        "total_chunks":      len(chunks),
        "fragment_rate":     round(fragment_rate, 3),
        "broken_row_rate":   round(broken_row_rate, 3),
        "table_header_rate": round(header_rate, 3),
        "text_coverage":     round(coverage_rate, 3),
        "empty_rate":        round(empty_rate, 3),
        "self_contain_rate": round(self_contain_rate, 3),
    }

    if errors:
        outcome = Outcome.BLOCK
    elif warnings:
        outcome = Outcome.WARN
    else:
        outcome = Outcome.PASS

    result = ChunkGuardrailResult(outcome=outcome, errors=errors, warnings=warnings, stats=stats)
    result.log(doc_id, file_name)
    return result
