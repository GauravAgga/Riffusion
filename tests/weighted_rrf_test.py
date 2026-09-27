import pytest

from rankweave.api.fusion import fuse


def test_weighted_rrf_fuses_multiple_rankings():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
    ]

    ranking2 = [
        {"id": "B"},
        {"id": "C"},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_rrf",
        weights=[0.3, 0.7],
        k=60,
    )

    assert [result.document_id for result in results] == [
        "B",
        "C",
        "A",
    ]


def test_weighted_rrf_calculates_scores():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
    ]

    ranking2 = [
        {"id": "B"},
        {"id": "C"},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_rrf",
        weights=[0.3, 0.7],
        k=60,
    )

    scores = {
        result.document_id: result.score
        for result in results
    }

    # A: only ranking1 rank 1 → 0.3 / 61
    # B: ranking1 rank 2 + ranking2 rank 1 → 0.3/62 + 0.7/61
    # C: only ranking2 rank 2 → 0.7 / 62
    assert scores["A"] == pytest.approx(0.3 / 61)
    assert scores["B"] == pytest.approx(0.3 / 62 + 0.7 / 61)
    assert scores["C"] == pytest.approx(0.7 / 62)


def test_weighted_rrf_weights_change_order():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
    ]

    ranking2 = [
        {"id": "B"},
        {"id": "A"},
    ]

    dense_heavy = fuse(
        [ranking1, ranking2],
        method="weighted_rrf",
        weights=[0.1, 0.9],
        k=60,
    )

    bm25_heavy = fuse(
        [ranking1, ranking2],
        method="weighted_rrf",
        weights=[0.9, 0.1],
        k=60,
    )

    # Equal-ish [0.5, 0.5]: both get 0.5/61 + 0.5/62 → tie; dense-heavy favors B
    assert dense_heavy[0].document_id == "B"
    assert bm25_heavy[0].document_id == "A"


def test_weighted_rrf_equal_weights():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
    ]

    ranking2 = [
        {"id": "B"},
        {"id": "A"},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_rrf",
        weights=[0.5, 0.5],
        k=60,
    )

    scores = {
        result.document_id: result.score
        for result in results
    }

    expected = 0.5 / 61 + 0.5 / 62
    assert scores["A"] == pytest.approx(expected)
    assert scores["B"] == pytest.approx(expected)


def test_weighted_rrf_document_in_only_one_list():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
    ]

    ranking2 = [
        {"id": "C"},
        {"id": "D"},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_rrf",
        weights=[0.5, 0.5],
        k=60,
    )

    document_ids = {result.document_id for result in results}
    assert document_ids == {"A", "B", "C", "D"}

    scores = {
        result.document_id: result.score
        for result in results
    }
    assert scores["A"] == pytest.approx(0.5 / 61)
    assert scores["B"] == pytest.approx(0.5 / 62)
    assert scores["C"] == pytest.approx(0.5 / 61)
    assert scores["D"] == pytest.approx(0.5 / 62)


def test_weighted_rrf_top_k():
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
        method="weighted_rrf",
        weights=[0.5, 0.5],
        top_k=2,
    )

    assert len(results) == 2


def test_weighted_rrf_duplicate_id_uses_first_occurrence():
    ranking = [
        {"id": "A"},
        {"id": "A"},
        {"id": "B"},
    ]

    results = fuse(
        [ranking],
        method="weighted_rrf",
        weights=[1.0],
        k=60,
    )

    scores = {
        result.document_id: result.score
        for result in results
    }

    # First A is rank 1; second A is ignored (not summed).
    assert scores["A"] == pytest.approx(1.0 / 61)
    assert scores["B"] == pytest.approx(1.0 / 63)
    assert [result.document_id for result in results] == ["A", "B"]


def test_weighted_rrf_requires_weights():
    ranking = [
        {"id": "A"},
        {"id": "B"},
    ]

    with pytest.raises(ValueError, match="weights are required"):
        fuse(
            [ranking],
            method="weighted_rrf",
        )


def test_weighted_rrf_weight_length_mismatch_raises():
    ranking1 = [
        {"id": "A"},
    ]
    ranking2 = [
        {"id": "B"},
    ]

    with pytest.raises(ValueError, match="weights length"):
        fuse(
            [ranking1, ranking2],
            method="weighted_rrf",
            weights=[1.0],
        )


def test_weighted_rrf_weight_zero_raises():
    ranking = [
        {"id": "A"},
    ]

    with pytest.raises(ValueError, match="0 < weight <= 1"):
        fuse(
            [ranking],
            method="weighted_rrf",
            weights=[0.0],
        )


def test_weighted_rrf_weight_above_one_raises():
    ranking1 = [
        {"id": "A"},
    ]
    ranking2 = [
        {"id": "B"},
    ]

    with pytest.raises(ValueError, match="0 < weight <= 1"):
        fuse(
            [ranking1, ranking2],
            method="weighted_rrf",
            weights=[1.1, -0.1],
        )


def test_weighted_rrf_single_weight_one_is_valid():
    ranking = [
        {"id": "A"},
        {"id": "B"},
    ]

    results = fuse(
        [ranking],
        method="weighted_rrf",
        weights=[1.0],
        k=60,
    )

    scores = {
        result.document_id: result.score
        for result in results
    }
    assert scores["A"] == pytest.approx(1.0 / 61)
    assert scores["B"] == pytest.approx(1.0 / 62)


def test_weighted_rrf_non_finite_weight_raises():
    ranking = [
        {"id": "A"},
    ]

    with pytest.raises(ValueError, match="finite number"):
        fuse(
            [ranking],
            method="weighted_rrf",
            weights=[float("nan")],
        )


def test_weighted_rrf_weights_must_sum_to_one():
    ranking1 = [
        {"id": "A"},
    ]
    ranking2 = [
        {"id": "B"},
    ]

    with pytest.raises(ValueError, match="weights must sum to 1"):
        fuse(
            [ranking1, ranking2],
            method="weighted_rrf",
            weights=[0.3, 0.3],
        )


def test_weighted_rrf_custom_id_field():
    ranking = [
        {"product_id": "P1"},
        {"product_id": "P2"},
    ]

    results = fuse(
        [ranking],
        method="weighted_rrf",
        id_field="product_id",
        weights=[1.0],
    )

    assert [result.document_id for result in results] == [
        "P1",
        "P2",
    ]


def test_weighted_rrf_callable_id_field():
    ranking = [
        {"product": {"id": "P1"}},
        {"product": {"id": "P2"}},
    ]

    results = fuse(
        [ranking],
        method="weighted_rrf",
        id_field=lambda item: item["product"]["id"],
        weights=[1.0],
    )

    assert [result.document_id for result in results] == [
        "P1",
        "P2",
    ]


def test_weighted_rrf_does_not_require_scores():
    ranking = [
        {"id": "A"},
        {"id": "B"},
    ]

    results = fuse(
        [ranking],
        method="weighted_rrf",
        weights=[1.0],
    )

    assert len(results) == 2


def test_weighted_rrf_preserves_original_document():
    document = {
        "id": "A",
        "name": "Running Shoes",
        "price": 100,
    }

    results = fuse(
        [[document]],
        method="weighted_rrf",
        weights=[1.0],
    )

    assert results[0].document == document


def test_rrf_still_rejects_weights():
    ranking = [
        {"id": "A"},
        {"id": "B"},
    ]

    with pytest.raises(ValueError, match="does not support weights"):
        fuse(
            [ranking],
            method="rrf",
            weights=[1.0],
        )


def test_weighted_score_still_requires_scores():
    ranking = [
        {"id": "A"},
    ]

    with pytest.raises(ValueError, match="requires scores"):
        fuse(
            [ranking],
            method="weighted_score",
        )
