from typing import Protocol, Sequence

from .models import FusedResult, RankedDocument


class FusionAlgorithm(Protocol):
    """Contract that every fusion algorithm must adhere to."""

    @property
    def requires_scores(self) -> bool:
        """Whether the algorithm requires scores to be provided."""
        ...

    @property
    def supports_weights(self) -> bool:
        """Whether the algorithm supports weights to be provided."""
        ...

    def fuse(
        self,
        rankings: Sequence[Sequence[RankedDocument]],
        *,
        weights: Sequence[float] | None = None,
        top_k: int | None = None,
        **kwargs,
    ) -> list[FusedResult]:
        """Fuse ranked document lists into a single ranked list."""
        ...
