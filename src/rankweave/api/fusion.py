from collections.abc import Callable, Iterable, Sequence
import inspect
from typing import Any

from rankweave.algorithms.rrf import RRFAlgorithm
from rankweave.algorithms.weighted_rrf import WeightedRRF
from rankweave.algorithms.weighted_score import WeightedScoreFusion
from rankweave.core.models import RankedDocument

_ALGORITHMS = {
    "rrf": RRFAlgorithm,
    "weighted_score": WeightedScoreFusion,
    "weighted_rrf": WeightedRRF,
}


def fuse(
    rankings: Iterable[Iterable[Any]],
    *,
    method: str = "rrf",
    id_field: str | Callable[[Any], str] = "id",
    score_field: str | Callable[[Any], float] | None = None,
    weights: Sequence[float] | None = None,
    k: int = 60,
    normalize: bool = True,
    top_k: int | None = None,
):
    """
    Fuse multiple ranked result lists into a single ranked result list.

    Parameters
    ----------
    rankings:
        Collection of ranked document lists. The order within each
        ranking represents the document's rank.

    method:
        Fusion algorithm to use, e.g. "rrf", "weighted_score",
        "weighted_rrf", etc.

    id_field:
        Field used to identify a document, or a callable that extracts
        the document ID.

    score_field:
        Optional field containing the original document score, or a
        callable that extracts the score. Required only by algorithms
        that operate on scores.

    weights:
        Optional weight for each ranking. Required by "weighted_rrf".
        Supported (optional) by "weighted_score". Not supported by plain RRF.

    k:
        Algorithm-specific parameter. For RRF and weighted RRF, this is the
        RRF smoothing constant.

    normalize:
        Algorithm-specific parameter. For weighted score fusion, controls
        whether each ranking's scores are min-max normalized to [0, 1]
        before weighting.

    top_k:
        Maximum number of results to return. None means return all
        results.
    """
    normalized_rankings = _normalize_rankings(
        rankings, id_field=id_field, score_field=score_field
    )
    algorithm_class = _ALGORITHMS.get(method)
    if algorithm_class is None:
        raise ValueError(f"Invalid method: {method}")
    algorithm = _instantiate_algorithm(
        algorithm_class, k=k, normalize=normalize
    )

    if algorithm.requires_scores and score_field is None:
        raise ValueError(f"Fusion method '{method}' requires scores")

    if algorithm.requires_scores:
        _validate_scores_present(normalized_rankings, method=method)

    if weights is not None and not algorithm.supports_weights:
        raise ValueError(f"Fusion method '{method}' does not support weights")
    results = algorithm.fuse(normalized_rankings, weights=weights, top_k=top_k)
    return results


def _instantiate_algorithm(algorithm_class: type, **params: object) -> object:
    """Construct an algorithm, passing only parameters its ``__init__`` accepts."""
    signature = inspect.signature(algorithm_class.__init__)
    accepted = {
        name
        for name, parameter in signature.parameters.items()
        if name != "self"
        and parameter.kind
        in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    }
    # If __init__ accepts **kwargs, forward everything.
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return algorithm_class(**params)
    filtered = {
        name: value for name, value in params.items() if name in accepted
    }
    return algorithm_class(**filtered)


def _validate_scores_present(
    rankings: Sequence[Sequence[RankedDocument]],
    *,
    method: str,
) -> None:
    for ranking in rankings:
        for ranked_document in ranking:
            if ranked_document.score is None:
                raise ValueError(
                    f"Fusion method '{method}' requires scores; "
                    f"document '{ranked_document.document_id}' is missing a score"
                )


def _normalize_rankings(
    rankings: Iterable[Iterable[Any]],
    *,
    id_field: str | Callable[[Any], str] | None = None,
    score_field: str | Callable[[Any], float] | None = None,
) -> list[list[RankedDocument]]:
    normalized_rankings: list[list[RankedDocument]] = []

    for ranking in rankings:
        normalized_ranking: list[RankedDocument] = []

        for rank, item in enumerate(ranking, start=1):
            if id_field is not None:
                document_id = (
                    id_field(item) if callable(id_field) else item.get(id_field)
                )
            else:
                document_id = item.get("id")

            if document_id is None:
                raise ValueError("id_field is required to identify documents")

            if score_field is not None:
                score = (
                    score_field(item)
                    if callable(score_field)
                    else item.get(score_field)
                )
            else:
                score = None

            normalized_ranking.append(
                RankedDocument(
                    document=item,
                    document_id=str(document_id),
                    rank=rank,
                    score=score,
                )
            )

        normalized_rankings.append(normalized_ranking)

    return normalized_rankings
