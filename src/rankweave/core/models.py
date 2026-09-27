from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class RankedDocument:
    """Internal representation of a ranked document."""

    document: Any
    document_id: str
    rank: int
    score: float | None = None


@dataclass(frozen=True, slots=True)
class FusedResult:
    """Results produced by a fusion algorithm."""

    document: Any
    document_id: str
    score: float
