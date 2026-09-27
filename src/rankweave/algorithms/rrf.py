"""Reciprocal Rank Fusion (RRF) for combining multiple ranked result lists.

RRF is a score-agnostic fusion method commonly used in hybrid search and RAG
pipelines to merge rankings from heterogeneous retrievers (for example, BM25
and dense vector search) without normalizing incompatible score scales.

For each document ``d`` that appears in one or more input rankings, the fused
score is:

.. math::

    \\text{RRF}(d) = \\sum_{i} \\frac{1}{k + r_i(d)}

where ``r_i(d)`` is the 1-based rank of ``d`` in ranking ``i``, and ``k`` is a
smoothing constant (default ``60``). Documents that appear in multiple lists
accumulate contributions from each list.

References
----------
Cormack, G. V., Clarke, C. L. A., & Buettcher, S. (2009).
*Reciprocal rank fusion outperforms condorcet and individual rank learning
methods.* SIGIR '09.
"""

from __future__ import annotations

from collections.abc import Sequence

from rankweave.core.models import FusedResult, RankedDocument


class RRFAlgorithm:
    """Fuse multiple ranked document lists using Reciprocal Rank Fusion.

    This implementation treats each input list as an ordered ranking only; raw
    retrieval scores are ignored. All lists contribute equally—per-list weights
    are not supported.

    Parameters
    ----------
    k:
        Smoothing constant added to each rank in the denominator. Larger values
        reduce the influence of top-ranked items and flatten score differences.
        The literature and common practice use ``k=60``.

    Examples
    --------
    >>> from rankweave.core.models import RankedDocument
    >>> from rankweave.algorithms.rrf import RRFAlgorithm
    >>>
    >>> bm25 = [
    ...     RankedDocument(document={"text": "doc-a"}, document_id="a", rank=1),
    ...     RankedDocument(document={"text": "doc-b"}, document_id="b", rank=2),
    ... ]
    >>> dense = [
    ...     RankedDocument(document={"text": "doc-b"}, document_id="b", rank=1),
    ...     RankedDocument(document={"text": "doc-c"}, document_id="c", rank=2),
    ... ]
    >>> fused = RRFAlgorithm(k=60).fuse([bm25, dense])
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
        return False

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
        """Combine ranked lists into a single list ordered by RRF score.

        Each inner sequence is one retriever's ranking. Ranks are taken from
        :attr:`~rankweave.core.models.RankedDocument.rank` and are expected to
        be 1-based (rank ``1`` is the best result).

        Parameters
        ----------
        rankings:
            One ranked list per source retriever. A document may appear in
            multiple lists; its final score is the sum of reciprocal-rank
            contributions from every list in which it appears.
        weights:
            Not supported for plain RRF. Passing a non-``None`` value raises
            :class:`ValueError`.
        top_k:
            If given, return at most this many fused results after sorting by
            score descending. Must be a positive integer when provided.
        **kwargs:
            Ignored. Accepted for compatibility with the fusion algorithm
            protocol.

        Returns
        -------
        list[FusedResult]
            Fused documents sorted by descending RRF score. Ties preserve the
            relative order produced by Python's stable sort.

        Raises
        ------
        ValueError
            If ``weights`` is provided, or if ``top_k`` is not positive.
        """
        if weights is not None:
            raise ValueError("weights are not supported for RRF")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        scores: dict[str, float] = {}
        documents: dict[str, object] = {}

        for ranking in rankings:
            for ranked_document in ranking:
                document_id = ranked_document.document_id
                rank = ranked_document.rank

                # Each appearance in a ranking adds 1 / (k + rank).
                rrf_score = 1.0 / (self._k + rank)
                scores[document_id] = scores.get(document_id, 0.0) + rrf_score

                # Preserve the first seen document payload for each ID.
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
