import pytest

from rankweave.api.fusion import fuse


def test_weighted_score_fuses_multiple_rankings():
    ranking1 = [
        {"id": "A", "score": 10.0},
        {"id": "B", "score": 5.0},
    ]

    ranking2 = [
        {"id": "B", "score": 0.9},
        {"id": "C", "score": 0.5},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
        weights=[0.3, 0.7],
    )

    assert [result.document_id for result in results] == [
        "B",
        "A",
        "C",
    ]


def test_weighted_score_calculates_normalized_scores():
    ranking1 = [
        {"id": "A", "score": 10.0},
        {"id": "B", "score": 5.0},
    ]

    ranking2 = [
        {"id": "B", "score": 0.9},
        {"id": "C", "score": 0.5},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
        weights=[0.3, 0.7],
        normalize=True,
    )

    scores = {
        result.document_id: result.score
        for result in results
    }

    # ranking1 min-max: A=1.0, B=0.0
    # ranking2 min-max: B=1.0, C=0.0
    assert scores["A"] == pytest.approx(0.3 * 1.0)
    assert scores["B"] == pytest.approx(0.3 * 0.0 + 0.7 * 1.0)
    assert scores["C"] == pytest.approx(0.7 * 0.0)


def test_weighted_score_without_normalization():
    ranking1 = [
        {"id": "A", "score": 10.0},
        {"id": "B", "score": 5.0},
    ]

    ranking2 = [
        {"id": "B", "score": 0.9},
        {"id": "C", "score": 0.5},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
        weights=[0.3, 0.7],
        normalize=False,
    )

    scores = {
        result.document_id: result.score
        for result in results
    }

    assert scores["A"] == pytest.approx(0.3 * 10.0)
    assert scores["B"] == pytest.approx(0.3 * 5.0 + 0.7 * 0.9)
    assert scores["C"] == pytest.approx(0.7 * 0.5)
    assert [result.document_id for result in results] == [
        "A",
        "B",
        "C",
    ]


def test_weighted_score_default_equal_weights():
    ranking1 = [
        {"id": "A", "score": 10.0},
        {"id": "B", "score": 5.0},
    ]

    ranking2 = [
        {"id": "B", "score": 0.9},
        {"id": "C", "score": 0.5},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
    )

    scores = {
        result.document_id: result.score
        for result in results
    }

    # Equal weights (1.0, 1.0) with normalization:
    # A=1.0, B=1.0, C=0.0
    assert scores["A"] == pytest.approx(1.0)
    assert scores["B"] == pytest.approx(1.0)
    assert scores["C"] == pytest.approx(0.0)


def test_weighted_score_weights_change_order():
    ranking1 = [
        {"id": "A", "score": 10.0},
        {"id": "B", "score": 5.0},
    ]

    ranking2 = [
        {"id": "B", "score": 0.9},
        {"id": "A", "score": 0.1},
    ]

    dense_heavy = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
        weights=[0.1, 0.9],
    )

    bm25_heavy = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
        weights=[0.9, 0.1],
    )

    # ranking1: A=1.0, B=0.0; ranking2: B=1.0, A=0.0
    assert dense_heavy[0].document_id == "B"
    assert bm25_heavy[0].document_id == "A"


def test_weighted_score_constant_scores_contribute_zero():
    ranking1 = [
        {"id": "A", "score": 5.0},
        {"id": "B", "score": 5.0},
    ]

    ranking2 = [
        {"id": "B", "score": 0.9},
        {"id": "C", "score": 0.5},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
        weights=[0.5, 0.5],
    )

    scores = {
        result.document_id: result.score
        for result in results
    }

    # Constant ranking1 contributes 0 for A and B.
    assert scores["A"] == pytest.approx(0.0)
    assert scores["B"] == pytest.approx(0.5 * 1.0)
    assert scores["C"] == pytest.approx(0.0)


def test_weighted_score_document_in_only_one_list():
    ranking1 = [
        {"id": "A", "score": 10.0},
        {"id": "B", "score": 5.0},
    ]

    ranking2 = [
        {"id": "C", "score": 0.8},
        {"id": "D", "score": 0.2},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
        weights=[0.5, 0.5],
    )

    document_ids = {result.document_id for result in results}
    assert document_ids == {"A", "B", "C", "D"}

    scores = {
        result.document_id: result.score
        for result in results
    }
    assert scores["A"] == pytest.approx(0.5 * 1.0)
    assert scores["B"] == pytest.approx(0.0)
    assert scores["C"] == pytest.approx(0.5 * 1.0)
    assert scores["D"] == pytest.approx(0.0)


def test_weighted_score_top_k():
    ranking1 = [
        {"id": "A", "score": 10.0},
        {"id": "B", "score": 5.0},
        {"id": "C", "score": 1.0},
    ]

    ranking2 = [
        {"id": "B", "score": 0.9},
        {"id": "C", "score": 0.5},
        {"id": "D", "score": 0.1},
    ]

    results = fuse(
        [ranking1, ranking2],
        method="weighted_score",
        score_field="score",
        top_k=2,
    )

    assert len(results) == 2


def test_weighted_score_requires_score_field():
    ranking = [
        {"id": "A", "score": 1.0},
    ]

    with pytest.raises(ValueError, match="requires scores"):
        fuse(
            [ranking],
            method="weighted_score",
        )


def test_weighted_score_missing_score_value_raises():
    ranking = [
        {"id": "A", "score": 1.0},
        {"id": "B"},
    ]

    with pytest.raises(ValueError, match="missing a score"):
        fuse(
            [ranking],
            method="weighted_score",
            score_field="score",
        )


def test_weighted_score_callable_score_field():
    ranking = [
        {"id": "A", "relevance": 10.0},
        {"id": "B", "relevance": 5.0},
    ]

    results = fuse(
        [ranking],
        method="weighted_score",
        score_field=lambda item: item["relevance"],
    )

    assert [result.document_id for result in results] == ["A", "B"]
    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.0)


def test_weighted_score_custom_id_field():
    ranking = [
        {"product_id": "P1", "score": 10.0},
        {"product_id": "P2", "score": 5.0},
    ]

    results = fuse(
        [ranking],
        method="weighted_score",
        id_field="product_id",
        score_field="score",
    )

    assert [result.document_id for result in results] == [
        "P1",
        "P2",
    ]


def test_weighted_score_weight_length_mismatch_raises():
    ranking1 = [
        {"id": "A", "score": 1.0},
    ]
    ranking2 = [
        {"id": "B", "score": 1.0},
    ]

    with pytest.raises(ValueError, match="weights length"):
        fuse(
            [ranking1, ranking2],
            method="weighted_score",
            score_field="score",
            weights=[1.0],
        )


def test_weighted_score_non_finite_weight_raises():
    ranking = [
        {"id": "A", "score": 1.0},
    ]

    with pytest.raises(ValueError, match="finite number"):
        fuse(
            [ranking],
            method="weighted_score",
            score_field="score",
            weights=[float("nan")],
        )


def test_weighted_score_preserves_original_document():
    document = {
        "id": "A",
        "name": "Running Shoes",
        "price": 100,
        "score": 0.95,
    }

    results = fuse(
        [[document]],
        method="weighted_score",
        score_field="score",
    )

    assert results[0].document == document


def test_rrf_still_works_alongside_weighted_score():
    ranking1 = [
        {"id": "A"},
        {"id": "B"},
    ]

    ranking2 = [
        {"id": "B"},
        {"id": "A"},
    ]

    results = fuse([ranking1, ranking2], method="rrf", k=60)

    expected = (1 / 61) + (1 / 62)
    scores = {
        result.document_id: result.score
        for result in results
    }
    assert scores["A"] == expected
    assert scores["B"] == expected


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
