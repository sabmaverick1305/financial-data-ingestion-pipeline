"""Typed configuration for the Parser Engine.

Deliberately separate from `financial_pipeline.config.Settings` — the engine
must stay usable standalone, without pulling in the rest of the application's
configuration surface.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class ParserSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FIES_PARSER_", env_file=".env", extra="ignore")

    default_timeout_seconds: int = 60
    default_max_memory_mb: int = 1024
    supported_mime_types: tuple[str, ...] = ("application/pdf",)
    raw_artifact_enabled: bool = False


@lru_cache(maxsize=1)
def get_parser_settings() -> ParserSettings:
    return ParserSettings()
