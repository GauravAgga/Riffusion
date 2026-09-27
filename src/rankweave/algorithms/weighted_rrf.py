"""Weighted Reciprocal Rank Fusion (Weighted RRF) for combining ranked lists.

Weighted RRF extends Reciprocal Rank Fusion by applying a per-ranking weight
to each reciprocal-rank contribution. It remains score-agnostic: only rank
positions and list weights are used, so heterogeneous retrievers (for
example, BM25 and dense vector search) can be fused without normalizing
incompatible score scales.

For each document ``d`` that appears in one or more input rankings, the fused
score is:

.. math::

    \\text{WRRF}(d) = \\sum_{i} \\frac{w_i}{k + r_i(d)}

where ``w_i`` is the weight for ranking ``i``, ``r_i(d)`` is the 1-based rank
of ``d`` in ranking ``i``, and ``k`` is a smoothing constant (default ``60``).
Documents absent from a ranking contribute ``0`` for that ranking.

References
----------
Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
*Reciprocal rank fusion outperforms condorcet and individual rank learning
methods.* SIGIR '09.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rankweave.core.models import FusedResult, RankedDocument


class WeightedRRF:
    """Fuse multiple ranked document lists using Weighted Reciprocal Rank Fusion.

    This implementation treats each input list as an ordered ranking only; raw
    retrieval scores are ignored. Per-list weights control how much each
    ranking contributes to the final score.

    Parameters
    ----------
    k:
        Smoothing constant added to each rank in the denominator. Larger values
        reduce the influence of top-ranked items and flatten score differences.
        The literature and common practice use ``k=60``.

    Examples
    --------
    >>> from rankweave.core.models import RankedDocument
    >>> from rankweave.algorithms.weighted_rrf import WeightedRRF
    >>>
    >>> bm25 = [
    ...     RankedDocument(document={"text": "doc-a"}, document_id="a", rank=1),
    ...     RankedDocument(document={"text": "doc-b"}, document_id="b", rank=2),
    ... ]
    >>> dense = [
    ...     RankedDocument(document={"text": "doc-b"}, document_id="b", rank=1),
    ...     RankedDocument(document={"text": "doc-c"}, document_id="c", rank=2),
    ... ]
    >>> fused = WeightedRRF(k=60).fuse([bm25, dense], weights=[0.3, 0.7])
    >>> fused[0].document_id
    'b'
    """

    def __init__(self, k: int = 60) -> None:
        if k <= 0:
            raise ValueError("k must be a positive integer")
        self._k = k

    @property
    def supports_weights(self) -> bool:
        """Whether the algorithm accepts per-ranking weights."""
        return True

    @property
    def requires_scores(self) -> bool:
        """Whether the algorithm requires raw retrieval scores on input."""
        return False

    def fuse(
        self,
        rankings: Sequence[Sequence[RankedDocument]],
        *,
        weights: Sequence[float] | None = None,
        top_k: int | None = None,
        **kwargs: object,
    ) -> list[FusedResult]:
        """Combine ranked lists into a single list ordered by weighted RRF score.

        Each inner sequence is one retriever's ranking. Ranks are taken from
        :attr:`~rankweave.core.models.RankedDocument.rank` and are expected to
        be 1-based (rank ``1`` is the best result). Within a single ranking,
        each ``document_id`` contributes once (first occurrence wins if
        duplicates appear).

        Parameters
        ----------
        rankings:
            One ranked list per source retriever. A document may appear in
            multiple lists; its final score is the weighted sum of
            reciprocal-rank contributions from every list in which it appears.
        weights:
            Required weight for each ranking. Must have the same length as
            ``rankings``. Each weight must be a finite number satisfying
            ``0 < weight <= 1``, and the weights must sum to ``1``
            (checked with :func:`math.isclose`).
        top_k:
            If given, return at most this many fused results after sorting by
            score descending. Must be a positive integer when provided.
        **kwargs:
            Ignored. Accepted for compatibility with the fusion algorithm
            protocol.

        Returns
        -------
        list[FusedResult]
            Fused documents sorted by descending weighted RRF score. Ties
            preserve the relative order produced by Python's stable sort.

        Raises
        ------
        ValueError
            If ``weights`` is missing, length does not match ``rankings``, a
            weight is not finite or outside ``(0, 1]``, weights do not sum to
            ``1``, or ``top_k`` is not positive.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        if weights is None:
            raise ValueError("weights are required for weighted RRF")

        num_rankings = len(rankings)
        if len(weights) != num_rankings:
            raise ValueError(
                f"weights length ({len(weights)}) must match "
                f"number of rankings ({num_rankings})"
            )

        resolved_weights: list[float] = []
        for index, weight in enumerate(weights):
            if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                raise ValueError(
                    f"weight at index {index} must be a finite number"
                )
            value = float(weight)
            if not math.isfinite(value):
                raise ValueError(
                    f"weight at index {index} must be a finite number"
                )
            if not (0.0 < value <= 1.0):
                raise ValueError(
                    f"weight at index {index} must satisfy 0 < weight <= 1"
                )
            resolved_weights.append(value)

        if not math.isclose(sum(resolved_weights), 1.0):
            raise ValueError("weights must sum to 1")

        scores: dict[str, float] = {}
        documents: dict[str, object] = {}

        for ranking, weight in zip(rankings, resolved_weights):
            seen_ids: set[str] = set()
            for ranked_document in ranking:
                document_id = ranked_document.document_id
                if document_id in seen_ids:
                    continue
                seen_ids.add(document_id)

                contribution = weight / (self._k + ranked_document.rank)
                scores[document_id] = scores.get(document_id, 0.0) + contribution
                documents.setdefault(document_id, ranked_document.document)

        results = [
            FusedResult(
                document=documents[document_id],
                score=score,
                document_id=document_id,
            )
            for document_id, score in scores.items()
        ]
        results.sort(key=lambda result: result.score, reverse=True)

        if top_k is not None:
            results = results[:top_k]

        return results
