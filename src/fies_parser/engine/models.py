"""Parser-independent request/document models.

These carry no parser-specific configuration — a `ParseRequest.configuration`
dict is the only place adapter-specific knobs live, and adapters are free to
ignore keys they don't recognize.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fies_parser.engine.exceptions import InvalidPageSelectionError, UnsupportedDocumentError


class SourceDocument(BaseModel):
    """A document to be parsed, independent of any parser implementation."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    document_id: str
    file_path: Path
    file_name: str
    mime_type: str
    file_hash: str
    source: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_file(self) -> SourceDocument:
        if not self.file_path.exists():
            raise UnsupportedDocumentError(self.document_id, f"file does not exist: {self.file_path}")
        if not self.file_path.is_file():
            raise UnsupportedDocumentError(self.document_id, f"not a regular file: {self.file_path}")
        return self


class ParseRequest(BaseModel):
    """A request to parse a `SourceDocument`, either in full or by page.

    `pages` uses a one-based public API; `None` means the complete document.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    document: SourceDocument
    pages: tuple[int, ...] | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    timeout_seconds: int = 60
    max_memory_mb: int = 1024

    @field_validator("pages")
    @classmethod
    def _validate_pages(cls, pages: tuple[int, ...] | None) -> tuple[int, ...] | None:
        if pages is None:
            return None
        if not pages:
            raise InvalidPageSelectionError("page selection must not be empty when provided", pages)
        invalid = tuple(p for p in pages if p <= 0)
        if invalid:
            raise InvalidPageSelectionError("page numbers are one-based and must be >= 1", invalid)
        return tuple(sorted(set(pages)))
