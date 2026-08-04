from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

from src.model_functions.multi_similarity_siamese import (
    MultiSimilaritySiamese,
    comparison_features,
    multi_similarity_features,
)
from src.one_shot import cli
from src.one_shot_data import (
    build_balanced_pair_indices,
    build_one_shot_split,
    sample_class_references,
)
from src.one_shot_gallery import (
    ReferenceGallery,
    ScoreNormalizer,
    attach_semantic_suggestions,
    baseline_scores_against_gallery,
    calibrate_unknown_threshold,
    classify_from_pair_scores,
    fit_score_normalizer,
)
from src.zero_shot_data import INPUT_COLUMNS, build_outer_fold


def _frame(rows_per_class: int = 14) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for class_id in range(8):
        for row_id in range(rows_per_class):
            row: dict[str, float | int] = {
                "Class": class_id,
                "Position": row_id,
                "loss": class_id,
                "Reflectance": -class_id,
                "SNR": class_id * 100 + row_id,
            }
            row.update({f"P{index}": class_id * 1000 + row_id * 10 + index for index in range(1, 31)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_multi_similarity_features_are_symmetric_and_retain_product_coordinates() -> None:
    left = torch.randn(5, 16)
    right = torch.randn(5, 16)

    forward = multi_similarity_features(left, right)
    reverse = multi_similarity_features(right, left)

    assert forward.shape == (5, 19)
    assert torch.allclose(forward, reverse, atol=1e-6)
    assert torch.isfinite(forward).all()


def test_siamese_pair_score_is_order_invariant() -> None:
    model = MultiSimilaritySiamese(embedding_dim=16, dropout=0.0).eval()
    left = torch.randn(4, 2, 30)
    right = torch.randn(4, 2, 30)

    assert torch.allclose(model(left, right), model(right, left), atol=1e-6)


@pytest.mark.parametrize("mode,width", [("l1", 1), ("l2", 1), ("cosine", 1), ("product", 16), ("multi", 19)])
def test_similarity_ablation_feature_modes(mode: str, width: int) -> None:
    left = torch.randn(3, 16)
    right = torch.randn(3, 16)
    assert comparison_features(left, right, mode).shape == (3, width)
    assert MultiSimilaritySiamese(embedding_dim=16, similarity_mode=mode)(
        torch.randn(3, 2, 30), torch.randn(3, 2, 30)
    ).shape == (3,)


def test_balanced_pairs_are_deterministic_equal_and_never_self_pairs() -> None:
    labels = np.repeat(np.arange(4), 5)
    first = build_balanced_pair_indices(labels, pair_count=40, seed=9)
    second = build_balanced_pair_indices(labels, pair_count=40, seed=9)

    assert np.array_equal(first.left, second.left)
    assert np.array_equal(first.right, second.right)
    assert np.array_equal(first.targets, second.targets)
    assert int(first.targets.sum()) == 20
    assert np.all(first.left != first.right)
    assert np.all((labels[first.left] == labels[first.right]) == first.targets.astype(bool))


def test_one_shot_support_pool_and_query_are_group_disjoint() -> None:
    outer = build_outer_fold(_frame(), holdout=(1, 2), seed=42)
    split = build_one_shot_split(outer, support_fraction=0.2, seed=42)

    assert set(split.support_pool["Class"]) == {1, 2}
    assert set(split.query["Class"]) == {1, 2}
    assert set(split.support_pool["_input_group"]).isdisjoint(set(split.query["_input_group"]))
    assert len(split.support_pool) + len(split.query) == len(outer.unseen_test)

    references = sample_class_references(split.support_pool, references_per_class=1, seed=7)
    assert references["Class"].value_counts().to_dict() == {1: 1, 2: 1}


def test_gallery_rejects_unknown_then_enrollment_makes_it_classifiable() -> None:
    gallery = ReferenceGallery(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([0, 1]),
        row_indices=torch.tensor([10, 11]),
    )
    pair_scores = torch.tensor([[0.9, 0.2], [0.3, 0.4]])
    predicted, confidence, accepted = classify_from_pair_scores(
        pair_scores, gallery.labels, threshold=0.6, top_k=1
    )

    assert predicted.tolist() == [0, -1]
    assert confidence.tolist() == pytest.approx([0.9, 0.4])
    assert accepted.tolist() == [True, False]

    enrolled = gallery.enroll(torch.tensor([[0.5, 0.5]]), class_id=2, row_indices=torch.tensor([12]))
    post_scores = torch.tensor([[0.1, 0.2, 0.95]])
    post_predicted, _, post_accepted = classify_from_pair_scores(
        post_scores, enrolled.labels, threshold=0.6, top_k=1
    )
    assert post_predicted.tolist() == [2]
    assert post_accepted.tolist() == [True]


def test_threshold_calibration_maximizes_known_unknown_harmonic_mean() -> None:
    scores = np.array([0.95, 0.85, 0.8, 0.45, 0.35, 0.1])
    is_known = np.array([True, True, True, False, False, False])

    result = calibrate_unknown_threshold(scores, is_known)

    assert 0.45 < result.threshold <= 0.8
    assert result.known_acceptance == 1.0
    assert result.unknown_recall == 1.0
    assert result.harmonic_mean == 1.0


def test_robust_score_normalization_transfers_across_affine_score_scales() -> None:
    source = np.array([1.0, 2.0, 3.0, 4.0])
    shifted = source * 7.0 + 19.0
    source_norm = fit_score_normalizer(source)
    shifted_norm = fit_score_normalizer(shifted)

    assert np.allclose(source_norm.transform(source), shifted_norm.transform(shifted))
    assert ScoreNormalizer.from_dict(source_norm.to_dict()) == source_norm


def test_cosine_and_euclidean_1nn_baselines_score_expected_reference() -> None:
    gallery = ReferenceGallery(
        embeddings=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        labels=torch.tensor([0, 1]),
        row_indices=torch.tensor([0, 1]),
    )
    queries = torch.tensor([[0.9, 0.1], [0.1, 0.9]])

    cosine = baseline_scores_against_gallery(queries, gallery, method="cosine_1nn")
    euclidean = baseline_scores_against_gallery(queries, gallery, method="euclidean_1nn")

    assert cosine.argmax(1).tolist() == [0, 1]
    assert euclidean.argmax(1).tolist() == [0, 1]


def test_semantic_labels_are_suggestions_only_for_gallery_rejections() -> None:
    suggestions, sources = attach_semantic_suggestions(
        torch.tensor([2, -1, 4]),
        torch.tensor([7, 6, 5]),
    )

    assert suggestions.tolist() == [2, 6, 4]
    assert sources == ["gallery", "semantic_suggestion", "gallery"]


def test_one_shot_cli_exposes_training_evaluation_enrollment_and_classification() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in (
        "train-fold",
        "evaluate-detection",
        "evaluate-one-shot",
        "benchmark",
        "enroll-reference",
        "classify",
    ):
        assert command in result.output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_siamese_runs_real_cuda_forward_backward() -> None:
    model = MultiSimilaritySiamese(embedding_dim=16).cuda()
    left = torch.randn(8, 2, 30, device="cuda")
    right = torch.randn(8, 2, 30, device="cuda")
    loss = model(left, right).square().mean()
    loss.backward()

    assert loss.is_cuda
    assert next(model.parameters()).grad is not None
