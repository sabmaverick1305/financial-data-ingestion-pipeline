"""Output of a `RoutingPolicy` decision."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class RoutingDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    parser_name: str
    reason: str
    # Other registered parsers that could also handle this document, in
    # preference order. Informational only in this milestone — nothing
    # automatically retries through them; that's fallback execution, a
    # separate later layer.
    fallback_parser_names: tuple[str, ...] = Field(default_factory=tuple)
