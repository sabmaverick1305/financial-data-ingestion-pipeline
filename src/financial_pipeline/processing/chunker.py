from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[dict]:
    """Split text into overlapping chunks.

    Returns a list of dicts: {"chunk_id": int, "text": str, "start": int, "end": int}
    where start/end are character offsets into the original text.
    """
    if not text.strip():
        return []

    chunks: list[dict] = []
    start = 0
    chunk_id = 0
    text_len = len(text)

    while start < text_len:
        end = min(start + chunk_size, text_len)

        # Walk back to the nearest sentence boundary to avoid mid-sentence splits
        if end < text_len:
            boundary = text.rfind(". ", start, end)
            if boundary != -1 and boundary > start + overlap:
                end = boundary + 1

        chunks.append(
            {
                "chunk_id": chunk_id,
                "text": text[start:end].strip(),
                "start": start,
                "end": end,
            }
        )
        chunk_id += 1

        if end >= text_len:
            # Reached the end of the text — stop, rather than recompute a
            # `start` that can fail to advance once the remaining tail is
            # shorter than `overlap` (was an infinite loop: `end` pinned at
            # text_len and `end - overlap` stuck at or behind the current
            # `start`, so the loop never terminated and `chunks` grew until
            # the process was OOM-killed).
            break

        start = end - overlap

    return [c for c in chunks if c["text"]]
