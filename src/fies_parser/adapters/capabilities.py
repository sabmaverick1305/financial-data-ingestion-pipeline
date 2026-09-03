"""Declarative capabilities an adapter can advertise about itself.

Not consumed by anything in this milestone — `ParserEngine` never branches on
it. It exists so a future preflight/routing layer can pick an adapter (or
validate a request against one) without hardcoding parser names or importing
adapter-specific modules.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ParserCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    supports_page_selection: bool = False
    supports_tables: bool = False
    supports_ocr: bool = False
    supports_headings: bool = False
    supported_mime_types: tuple[str, ...] = Field(default_factory=tuple)
