from __future__ import annotations

import re


# ── Table-aware chunking ──────────────────────────────────────────────────────
# Docling renders PDF tables as GitHub-flavored markdown (lines starting with
# "|"). The generic sentence-boundary chunker cuts across table rows, leaving
# data cells in one chunk and their column headers in another.  The LLM then
# sees raw numbers like "| 35 33 | 1,71,79,356 1,39,42,522 |" with no idea
# which column is which.
#
# Fix: detect markdown table blocks, extract their header row, then emit
# chunks of TABLE_ROWS_PER_CHUNK rows each — with the header prepended to
# every chunk so the LLM always has column context.

TABLE_ROWS_PER_CHUNK = 3    # data rows per table chunk (excl. header); keeps chunks under ~1500 chars for AMFI wide tables
_TABLE_LINE = re.compile(r"^\s*\|")   # any line that starts with |


def _is_table_line(line: str) -> bool:
    return bool(_TABLE_LINE.match(line))


def _split_table_block(block_lines: list[str]) -> list[str]:
    """Split a markdown table block into chunks with header prepended.

    Returns a list of chunk texts.  If the table is small enough to fit in one
    chunk the entire block is returned as-is.
    """
    if not block_lines:
        return []

    # First line = column headers; second line = separator (|---|---| …)
    # Everything after that is data rows.
    header_rows: list[str] = []
    sep_idx = -1
    for i, line in enumerate(block_lines):
        if re.match(r"^\s*\|[-| :]+\|?\s*$", line):  # separator row
            header_rows = block_lines[: i + 1]
            sep_idx = i
            break

    if sep_idx == -1:
        # No separator found — treat as plain text block
        return ["\n".join(block_lines)]

    data_rows = block_lines[sep_idx + 1 :]
    header_text = "\n".join(header_rows)

    if not data_rows:
        return [header_text]

    chunks: list[str] = []
    for i in range(0, len(data_rows), TABLE_ROWS_PER_CHUNK):
        rows = data_rows[i : i + TABLE_ROWS_PER_CHUNK]
        chunks.append(header_text + "\n" + "\n".join(rows))

    return chunks


def _segment_text(text: str) -> list[tuple[str, str]]:
    """Split text into (kind, content) segments: 'table' or 'prose'."""
    segments: list[tuple[str, str]] = []
    lines = text.splitlines(keepends=True)

    i = 0
    while i < len(lines):
        if _is_table_line(lines[i]):
            # Collect contiguous table lines
            j = i
            while j < len(lines) and (_is_table_line(lines[j]) or lines[j].strip() == ""):
                j += 1
            block = [l.rstrip() for l in lines[i:j] if l.strip()]
            segments.append(("table", block))
            i = j
        else:
            # Collect prose until a table line appears
            j = i
            while j < len(lines) and not _is_table_line(lines[j]):
                j += 1
            segments.append(("prose", "".join(lines[i:j])))
            i = j

    return segments


def _chunk_prose(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split prose text into overlapping character-level chunks."""
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = end - overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """Split text into chunks, preserving table structure.

    Markdown table blocks (produced by Docling) are split row-by-row with the
    header row prepended to every chunk.  Prose sections use the original
    sentence-boundary strategy with character-level overlap.

    Returns a list of dicts:
        {"chunk_id": int, "text": str, "start": int, "end": int}
    where start/end are *approximate* character offsets (exact for prose,
    estimated for tables).
    """
    if not text.strip():
        return []

    segments = _segment_text(text)
    raw_chunks: list[str] = []

    for kind, content in segments:
        if kind == "table":
            raw_chunks.extend(_split_table_block(content))
        else:
            raw_chunks.extend(_chunk_prose(content, chunk_size, overlap))

    # Assign chunk_ids and approximate character offsets
    results: list[dict] = []
    cursor = 0
    for i, chunk_text_val in enumerate(raw_chunks):
        if not chunk_text_val.strip():
            continue
        start = text.find(chunk_text_val[:40], cursor)
        if start == -1:
            start = cursor
        end = start + len(chunk_text_val)
        results.append({
            "chunk_id": i,
            "text": chunk_text_val,
            "start": start,
            "end": end,
        })
        cursor = max(cursor, start)

    return results
