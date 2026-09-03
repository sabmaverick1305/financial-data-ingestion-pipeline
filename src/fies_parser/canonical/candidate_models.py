"""Parser-independent candidate models.

Every adapter converts its parser-specific output into these models before
returning from `ParserAdapter.parse()`. Nothing downstream of the adapter
boundary should ever see a PyMuPDF/Docling/LlamaParse object.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ElementType(StrEnum):
    TITLE = "title"
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    LIST = "list"
    FOOTNOTE = "footnote"
    IMAGE = "image"
    CAPTION = "caption"
    FORMULA = "formula"
    UNKNOWN = "unknown"


class BoundingBox(BaseModel):
    model_config = ConfigDict(frozen=True)

    x0: float
    y0: float
    x1: float
    y1: float


class CandidateElement(BaseModel):
    element_id: str
    element_type: ElementType
    page_number: int
    text: str | None = None
    bbox: BoundingBox | None = None
    parent_element_id: str | None = None
    parser_confidence: float | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidatePage(BaseModel):
    page_number: int
    width: float | None = None
    height: float | None = None
    rotation: int = 0
    text: str | None = None
    element_ids: list[str] = Field(default_factory=list)
    table_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class CandidateTable(BaseModel):
    table_id: str
    page_start: int
    page_end: int
    headers: list[str] = Field(default_factory=list)
    rows: list[list[str]] = Field(default_factory=list)
    title: str | None = None
    units: dict[str, str] = Field(default_factory=dict)
    footnotes: list[str] = Field(default_factory=list)
    bbox: BoundingBox | None = None
    parser_confidence: float | None = None
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParserExecutionMetrics(BaseModel):
    duration_seconds: float = 0.0
    pages_requested: int = 0
    pages_processed: int = 0
    peak_memory_mb: float | None = None
    cpu_seconds: float | None = None
    external_cost: float | None = None


class ParserCandidate(BaseModel):
    """The unit of output the Parser Engine returns for a single parser run.

    `parser_run_id` and `configuration_hash` are stamped by `ParserEngine`
    after the adapter returns, since only the engine has that context —
    adapters should leave them as empty strings.
    """

    parser_run_id: str = ""
    document_id: str
    parser_name: str
    parser_version: str
    configuration_hash: str = ""
    pages: list[CandidatePage] = Field(default_factory=list)
    elements: list[CandidateElement] = Field(default_factory=list)
    tables: list[CandidateTable] = Field(default_factory=list)
    metrics: ParserExecutionMetrics = Field(default_factory=ParserExecutionMetrics)
    warnings: list[str] = Field(default_factory=list)
    raw_artifact_uri: str | None = None
    # Document-level metadata the parser exposed (PDF info dict, Docling doc
    # metadata, etc.) — distinct from per-page/element/table `metadata`.
    document_metadata: dict[str, Any] = Field(default_factory=dict)
