"""The parser adapter contract.

Every concrete parser (PyMuPDF, Docling, LlamaParse, ...) implements this
interface and only this interface — `ParserEngine` and `ParserRegistry` never
import a concrete adapter or a third-party parsing library directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from fies_parser.adapters.capabilities import ParserCapabilities
from fies_parser.canonical.candidate_models import ParserCandidate
from fies_parser.engine.models import ParseRequest


class ParserAdapter(ABC):
    """Common interface every parser implementation must satisfy."""

    name: str
    version: str

    @abstractmethod
    def supports(self, request: ParseRequest) -> bool:
        """Whether this adapter can handle the given request (mime type, etc.).

        Must not raise; return False for anything it cannot handle.
        """

    @abstractmethod
    def parse(self, request: ParseRequest) -> ParserCandidate:
        """Parse the requested document/pages into common candidate models.

        Implementations must not leak parser-specific objects — only
        `ParserCandidate` and the models it's composed of may be returned.
        Raise `fies_parser.engine.exceptions` types for expected failures
        (invalid pages, etc.); let unexpected exceptions propagate, they are
        wrapped by `ParserEngine` into `ParserExecutionError`.
        """

    def healthcheck(self) -> bool:
        """Cheap liveness check. Default assumes the adapter is always usable."""
        return True

    @property
    def capabilities(self) -> ParserCapabilities:
        """What this adapter can do. Default declares nothing — override to
        advertise real capabilities so a future routing layer can pick an
        adapter without hardcoding its name."""
        return ParserCapabilities()
