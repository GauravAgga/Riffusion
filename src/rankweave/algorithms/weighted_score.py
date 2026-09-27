"""Weighted Score Fusion for combining multiple scored result lists.

Weighted Score Fusion is a score-based fusion method used in hybrid search
and RAG pipelines when retriever scores are meaningful and comparable after
optional normalization. Unlike Reciprocal Rank Fusion (RRF), which ignores
raw scores, this method combines (optionally min-max normalized) scores with
per-ranking weights.

For each document ``d`` that appears in one or more input rankings, the fused
score is:

.. math::

    \\text{WSF}(d) = \\sum_{i} w_i \\cdot s'_i(d)

where ``w_i`` is the weight for ranking ``i``, and ``s'_i(d)`` is the
(optionally normalized) score of ``d`` in ranking ``i``. Documents absent
from a ranking contribute ``0`` for that ranking.

When ``normalize=True`` (default), each ranking is independently min-max
scaled to ``[0, 1]`` before weighting so heterogeneous score scales (for
example, BM25 and dense similarity) can be combined. When all scores in a
ranking are equal, every document in that ranking contributes ``0.0``
(no relative signal).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from rankweave.core.models import FusedResult, RankedDocument


class WeightedScoreFusion:
    """Fuse multiple scored document lists using weighted score fusion.

    This implementation uses each document's raw retrieval score. Per-list
    weights control how much each ranking contributes to the final score.
    Optional min-max normalization makes scores from different retrievers
    comparable before weighting.

    Parameters
    ----------
    normalize:
        If ``True`` (default), min-max normalize each ranking's scores to
        ``[0, 1]`` before applying weights. If ``False``, use raw scores
        (weighted CombSUM).

    Examples
    --------
    >>> from rankweave.core.models import RankedDocument
    >>> from rankweave.algorithms.weighted_score import WeightedScoreFusion
    >>>
    >>> bm25 = [
    ...     RankedDocument(document={"text": "doc-a"}, document_id="a", rank=1, score=10.0),
    ...     RankedDocument(document={"text": "doc-b"}, document_id="b", rank=2, score=5.0),
    ... ]
    >>> dense = [
    ...     RankedDocument(document={"text": "doc-b"}, document_id="b", rank=1, score=0.9),
    ...     RankedDocument(document={"text": "doc-c"}, document_id="c", rank=2, score=0.5),
    ... ]
    >>> fused = WeightedScoreFusion(normalize=True).fuse(
    ...     [bm25, dense], weights=[0.3, 0.7]
    ... )
    >>> fused[0].document_id
    'b'
    """

    def __init__(self, normalize: bool = True) -> None:
        self._normalize = normalize

    @property
    def supports_weights(self) -> bool:
        """Whether the algorithm accepts per-ranking weights."""
        return True

    @property
    def requires_scores(self) -> bool:
        """Whether the algorithm requires raw retrieval scores on input."""
        return True

    def fuse(
        self,
        rankings: Sequence[Sequence[RankedDocument]],
        *,
        weights: Sequence[float] | None = None,
        top_k: int | None = None,
        **kwargs: object,
    ) -> list[FusedResult]:
        """Combine scored lists into a single list ordered by fused score.

        Each inner sequence is one retriever's ranking. Scores are taken from
        :attr:`~rankweave.core.models.RankedDocument.score`. Within a single
        ranking, each ``document_id`` contributes once (last occurrence wins
        if duplicates appear).

        Parameters
        ----------
        rankings:
            One scored list per source retriever. A document may appear in
            multiple lists; its final score is the weighted sum of
            (optionally normalized) contributions from every list.
        weights:
            Optional weight for each ranking. Must have the same length as
            ``rankings`` when provided. Each weight must be a finite number.
            If omitted, every ranking uses weight ``1.0``.
        top_k:
            If given, return at most this many fused results after sorting by
            score descending. Must be a positive integer when provided.
        **kwargs:
            Ignored. Accepted for compatibility with the fusion algorithm
            protocol.

        Returns
        -------
        list[FusedResult]
            Fused documents sorted by descending fused score. Ties preserve the
            relative order produced by Python's stable sort.

        Raises
        ------
        ValueError
            If ``weights`` length does not match ``rankings``, a weight is not
            finite, a document is missing a numeric score, or ``top_k`` is not
            positive.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        num_rankings = len(rankings)
        if weights is None:
            resolved_weights = [1.0] * num_rankings
        else:
            if len(weights) != num_rankings:
                raise ValueError(
                    f"weights length ({len(weights)}) must match "
                    f"number of rankings ({num_rankings})"
                )
            resolved_weights = []
            for index, weight in enumerate(weights):
                if not isinstance(weight, (int, float)) or isinstance(weight, bool):
                    raise ValueError(
                        f"weight at index {index} must be a finite number"
                    )
                if not math.isfinite(float(weight)):
                    raise ValueError(
                        f"weight at index {index} must be a finite number"
                    )
                resolved_weights.append(float(weight))

        scores: dict[str, float] = {}
        documents: dict[str, object] = {}

        for ranking, weight in zip(rankings, resolved_weights):
            ranking_scores = self._scores_for_ranking(ranking)

            for document_id, contribution in ranking_scores.items():
                scores[document_id] = (
                    scores.get(document_id, 0.0) + weight * contribution
                )

            for ranked_document in ranking:
                documents.setdefault(
                    ranked_document.document_id, ranked_document.document
                )

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

    def _scores_for_ranking(
        self, ranking: Sequence[RankedDocument]
    ) -> dict[str, float]:
        """Return per-document scores for one ranking (optionally normalized).

        Duplicate ``document_id`` values within a ranking keep the last score.
        """
        raw_scores: dict[str, float] = {}

        for ranked_document in ranking:
            score = ranked_document.score
            if score is None:
                raise ValueError(
                    f"document '{ranked_document.document_id}' is missing a score"
                )
            if not isinstance(score, (int, float)) or isinstance(score, bool):
                raise ValueError(
                    f"document '{ranked_document.document_id}' has a non-numeric score"
                )
            if not math.isfinite(float(score)):
                raise ValueError(
                    f"document '{ranked_document.document_id}' has a non-finite score"
                )
            raw_scores[ranked_document.document_id] = float(score)

        if not raw_scores:
            return {}

        if not self._normalize:
            return raw_scores

        values = list(raw_scores.values())
        min_score = min(values)
        max_score = max(values)
        if max_score == min_score:
            return {document_id: 0.0 for document_id in raw_scores}

        scale = max_score - min_score
        return {
            document_id: (score - min_score) / scale
            for document_id, score in raw_scores.items()
        }
