import pytest

from rankfuse.api.fusion import fuse


def test_rrf_fuses_multiple_rankings():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
        {"id": "C"},
    ]

    ranking2 = [
        {"id": "B"},
        {"id": "A"},
        {"id": "D"},
    ]

    results = fuse([ranking1, ranking2])

    assert [result.document_id for result in results] == [
        "A",
        "B",
        "C",
        "D",
    ]


def test_rrf_calculates_scores():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
    ]

    ranking2 = [
        {"id": "B"},
        {"id": "A"},
    ]

    results = fuse([ranking1, ranking2], k=60)

    scores = {
        result.document_id: result.score
        for result in results
    }

    expected = (1 / 61) + (1 / 62)

    assert scores["A"] == expected
    assert scores["B"] == expected


def test_rrf_top_k():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
        {"id": "C"},
    ]

    ranking2 = [
        {"id": "B"},
        {"id": "C"},
        {"id": "D"},
    ]

    results = fuse(
        [ranking1, ranking2],
        top_k=2,
    )

    assert len(results) == 2


def test_custom_id_field():
    ranking = [
        {"product_id": "P1"},
        {"product_id": "P2"},
    ]

    results = fuse(
        [ranking],
        id_field="product_id",
    )

    assert [result.document_id for result in results] == [
        "P1",
        "P2",
    ]


def test_callable_id_field():
    ranking = [
        {"product": {"id": "P1"}},
        {"product": {"id": "P2"}},
    ]

    results = fuse(
        [ranking],
        id_field=lambda item: item["product"]["id"],
    )

    assert [result.document_id for result in results] == [
        "P1",
        "P2",
    ]


def test_missing_id_raises_error():
    ranking = [
        {"name": "Product A"},
    ]

    with pytest.raises(ValueError):
        fuse([ranking])


def test_unsupported_algorithm_raises_error():
    ranking = [
        {"id": "A"},
    ]

    with pytest.raises(ValueError, match="Invalid method"):
        fuse(
            [ranking],
            method="does_not_exist",
        )


def test_rrf_does_not_support_weights():
    ranking = [
        {"id": "A"},
        {"id": "B"},
    ]

    with pytest.raises(ValueError, match="does not support weights"):
        fuse(
            [ranking],
            weights=[1.0],
        )


def test_rrf_does_not_require_scores():
    ranking = [
        {"id": "A"},
        {"id": "B"},
    ]

    results = fuse([ranking])

    assert len(results) == 2


def test_original_document_is_preserved():
    document = {
        "id": "A",
        "name": "Running Shoes",
        "price": 100,
    }

    results = fuse([[document]])

    assert results[0].document == document