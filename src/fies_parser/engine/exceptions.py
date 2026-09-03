"""Typed exception hierarchy for the Parser Engine.

Every failure surfaced to callers of `ParserEngine`/`ParserRegistry` is one of
these types — never a bare `Exception`. Adapters raising unexpected errors
have them wrapped in `ParserExecutionError` by the engine.
"""

from __future__ import annotations

from typing import Any


class ParserEngineError(Exception):
    """Base type for every error the Parser Engine can raise."""


class UnknownParserError(ParserEngineError):
    """Raised when a requested parser name is not registered."""

    def __init__(self, parser_name: str, available: tuple[str, ...] = ()) -> None:
        self.parser_name = parser_name
        self.available = available
        super().__init__(f"Unknown parser {parser_name!r}. Available parsers: {list(available)}")


class DuplicateParserError(ParserEngineError):
    """Raised when registering a parser name that is already registered."""

    def __init__(self, parser_name: str) -> None:
        self.parser_name = parser_name
        super().__init__(f"Parser {parser_name!r} is already registered")


class UnsupportedDocumentError(ParserEngineError):
    """Raised when a document is missing, not a regular file, or the selected
    parser does not support it (mime type, structure, etc.)."""

    def __init__(self, document_id: str, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"Document {document_id!r} is unsupported: {reason}")


class InvalidPageSelectionError(ParserEngineError):
    """Raised for structurally invalid page numbers (<= 0) or pages outside
    the document's actual page range."""

    def __init__(self, reason: str, pages: tuple[int, ...] = ()) -> None:
        self.reason = reason
        self.pages = pages
        super().__init__(f"Invalid page selection {list(pages)}: {reason}")


class NoParserAvailableError(ParserEngineError):
    """Raised by a `RoutingPolicy` when no registered parser can handle a
    document — no adapter supports its mime type, or the document needs a
    capability (e.g. OCR) that nothing registered declares."""

    def __init__(self, document_id: str, reason: str) -> None:
        self.document_id = document_id
        self.reason = reason
        super().__init__(f"No parser available for document {document_id!r}: {reason}")


class ParserTimeoutError(ParserEngineError):
    """Raised when a parser execution exceeds `ParseRequest.timeout_seconds`."""

    def __init__(self, parser_name: str, timeout_seconds: int) -> None:
        self.parser_name = parser_name
        self.timeout_seconds = timeout_seconds
        super().__init__(f"Parser {parser_name!r} exceeded timeout of {timeout_seconds}s")


class ParserResourceLimitError(ParserEngineError):
    """Raised when a parser execution exceeds `ParseRequest.max_memory_mb`."""

    def __init__(self, parser_name: str, max_memory_mb: int) -> None:
        self.parser_name = parser_name
        self.max_memory_mb = max_memory_mb
        super().__init__(f"Parser {parser_name!r} exceeded memory limit of {max_memory_mb}MB")


class InvalidCandidateError(ParserEngineError):
    """Raised when a `ParserCandidate` returned by an adapter fails structural
    validation (`CandidateValidator`) — duplicate ids, dangling cross-page
    references, degenerate geometry. Not raised for financial-value issues."""

    def __init__(self, parser_name: str, document_id: str, issues: list[str]) -> None:
        self.parser_name = parser_name
        self.document_id = document_id
        self.issues = issues
        super().__init__(f"Parser {parser_name!r} produced an invalid candidate for document {document_id!r}: {issues}")


class ParserExecutionError(ParserEngineError):
    """Wraps any unexpected exception raised by an adapter's `parse()` call."""

    def __init__(self, parser_name: str, document_id: str, original_error: Exception) -> None:
        self.parser_name = parser_name
        self.document_id = document_id
        self.original_error = original_error
        super().__init__(
            f"Parser {parser_name!r} failed on document {document_id!r}: {type(original_error).__name__}: {original_error}"
        )


def error_context(exc: ParserEngineError) -> dict[str, Any]:
    """Best-effort structured fields for logging, without the message string."""
    return {k: v for k, v in vars(exc).items() if k != "args"}
