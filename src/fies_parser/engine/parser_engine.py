"""The Parser Engine — executes a registered adapter for a `ParseRequest`.

Scope is intentionally narrow: validate the document, resolve and run one
adapter, stamp identity/timing metadata onto the result. It does not decide
*which* parser to use, retry through a different one, validate financial
correctness, chunk, embed, or publish anything — those are later layers.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any
from uuid import uuid4

import structlog

from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.engine.candidate_validator import CandidateValidator
from fies_parser.engine.exceptions import (
    InvalidCandidateError,
    ParserEngineError,
    ParserExecutionError,
    UnsupportedDocumentError,
)
from fies_parser.engine.models import ParseRequest
from fies_parser.engine.registry import ParserRegistry

log = structlog.get_logger()


def compute_configuration_hash(parser_name: str, parser_version: str, configuration: dict[str, Any]) -> str:
    """Deterministic SHA-256 of (parser name, version, configuration).

    Uses canonical JSON (sorted keys, no incidental whitespace) so the same
    logical configuration always hashes the same way regardless of dict
    insertion order.
    """
    payload = {
        "parser_name": parser_name,
        "parser_version": parser_version,
        "configuration": configuration,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class ParserEngine:
    """Executes registered `ParserAdapter`s through a common contract."""

    def __init__(self, registry: ParserRegistry, validator: CandidateValidator | None = None) -> None:
        self._registry = registry
        self._validator = validator or CandidateValidator()

    def run(self, parser_name: str, request: ParseRequest) -> ParserCandidate:
        document = request.document
        self._validate_document(document.document_id, document.file_path)

        adapter = self._registry.get(parser_name)

        if not adapter.supports(request):
            raise UnsupportedDocumentError(
                document.document_id,
                f"parser {adapter.name!r} does not support mime type {document.mime_type!r}",
            )

        configuration_hash = compute_configuration_hash(adapter.name, adapter.version, request.configuration)
        parser_run_id = str(uuid4())

        log.info(
            "parser_engine.execution_started",
            parser_run_id=parser_run_id,
            parser_name=adapter.name,
            parser_version=adapter.version,
            document_id=document.document_id,
            pages_requested=len(request.pages) if request.pages else None,
            configuration_hash=configuration_hash,
        )

        start = time.perf_counter()
        try:
            candidate = adapter.parse(request)
        except ParserEngineError as exc:
            log.warning(
                "parser_engine.execution_failed",
                parser_run_id=parser_run_id,
                parser_name=adapter.name,
                document_id=document.document_id,
                configuration_hash=configuration_hash,
                duration=round(time.perf_counter() - start, 4),
                error=type(exc).__name__,
            )
            raise
        except Exception as exc:
            log.warning(
                "parser_engine.execution_failed",
                parser_run_id=parser_run_id,
                parser_name=adapter.name,
                document_id=document.document_id,
                configuration_hash=configuration_hash,
                duration=round(time.perf_counter() - start, 4),
                error=type(exc).__name__,
            )
            raise ParserExecutionError(adapter.name, document.document_id, exc) from exc

        duration = time.perf_counter() - start

        final_candidate = candidate.model_copy(
            update={
                "parser_run_id": parser_run_id,
                "document_id": document.document_id,
                "parser_name": adapter.name,
                "parser_version": adapter.version,
                "configuration_hash": configuration_hash,
                "metrics": candidate.metrics.model_copy(update={"duration_seconds": duration}),
            }
        )

        issues = self._validator.validate(final_candidate)
        if issues:
            log.warning(
                "parser_engine.candidate_invalid",
                parser_run_id=parser_run_id,
                parser_name=adapter.name,
                document_id=document.document_id,
                issue_count=len(issues),
            )
            raise InvalidCandidateError(adapter.name, document.document_id, issues)

        log.info(
            "parser_engine.execution_completed",
            parser_run_id=parser_run_id,
            parser_name=adapter.name,
            parser_version=adapter.version,
            document_id=document.document_id,
            pages_requested=final_candidate.metrics.pages_requested,
            pages_processed=final_candidate.metrics.pages_processed,
            duration=round(duration, 4),
            configuration_hash=configuration_hash,
        )

        return final_candidate

    def _validate_document(self, document_id: str, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            raise UnsupportedDocumentError(document_id, f"file is no longer accessible: {file_path}")
