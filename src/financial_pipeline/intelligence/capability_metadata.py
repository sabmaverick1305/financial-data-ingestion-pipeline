"""Operational metadata for FIES capabilities."""
from __future__ import annotations
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

class CapabilityMode(StrEnum):
    DETERMINISTIC = "deterministic"
    RETRIEVAL = "retrieval"
    LLM = "llm"

@dataclass(frozen=True)
class CapabilityMetadata:
    required_inputs: tuple[str, ...] = ()
    output_schema: dict[str, Any] | None = None
    source_authority: str = "internal"
    freshness_seconds: int | None = None
    mode: CapabilityMode = CapabilityMode.DETERMINISTIC
    estimated_cost_usd: float = 0.0
    estimated_latency_ms: int = 0
    allow_partial: bool = False
