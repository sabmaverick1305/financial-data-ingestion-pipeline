"""Parser adapter registry.

An instance, not a module-level global — callers construct one and inject the
adapters they want available, so multiple registries (e.g. per environment,
per test) never share state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from fies_parser.engine.exceptions import DuplicateParserError, UnknownParserError

if TYPE_CHECKING:
    # Deferred to break the adapters.base <-> engine (registry -> parser_engine
    # -> engine.__init__ -> models, imported from adapters.base) import cycle.
    # Only used in type hints, which `from __future__ import annotations`
    # already evaluates lazily.
    from fies_parser.adapters.base import ParserAdapter


class ParserRegistry:
    """Holds parser adapters keyed by their unique `name`."""

    def __init__(self) -> None:
        self._adapters: dict[str, ParserAdapter] = {}

    def register(self, adapter: ParserAdapter) -> None:
        if adapter.name in self._adapters:
            raise DuplicateParserError(adapter.name)
        self._adapters[adapter.name] = adapter

    def get(self, name: str) -> ParserAdapter:
        try:
            return self._adapters[name]
        except KeyError:
            raise UnknownParserError(name, tuple(self._adapters)) from None

    def list_parsers(self) -> tuple[str, ...]:
        return tuple(self._adapters)
