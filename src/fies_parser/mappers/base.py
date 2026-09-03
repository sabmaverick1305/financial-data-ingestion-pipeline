"""Generic mapping contract from a `ParserCandidate` to a target shape.

Concrete mappers translate the engine's parser-agnostic output into whatever
shape a specific consumer needs — a legacy result type, a future FIES
Document IR, an export format, etc. `fies_parser` only defines the contract;
consumer-specific target types (and their imports) live with the consumer,
keeping the dependency direction one-way (consumer -> fies_parser, never
back).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from fies_parser.canonical.candidate_models import ParserCandidate

T = TypeVar("T")


class CandidateMapper(ABC, Generic[T]):
    """Maps a `ParserCandidate` into a target representation `T`."""

    @abstractmethod
    def map(self, candidate: ParserCandidate) -> T:
        """Convert `candidate` into the target shape. Must not mutate `candidate`."""
