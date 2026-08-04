from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

from src.model_functions.study_models import EpisodicMetricModel, PhysicsSemanticModel, SelfSupervisedTraceModel
from src.model_functions.zero_shot import require_cuda
from src.otdr_study import cli
from src.study_data import deterministic_support_indices, group_stratified_inner_splits, prepare_fold, validate_model_features
from src.study_experiment import class_prototypes
from src.study_experiment import DensityScorer
from src.study_metrics import conformal_p_values, open_set_metrics, post_enrollment_metrics, threshold_at_normal_far
from src.study_semantics import load_physics_prototypes
from src.study_state import stable_run_id, validate_run, write_manifest
from src.study_training import ApproachAConfig
from src.zero_shot_data import INPUT_COLUMNS


def _frame(rows_per_class: int = 18) -> pd.DataFrame:
    rows = []
    for class_id in range(8):
        for row in range(rows_per_class):
            values = {"SNR": class_id + row / 100}
            values.update({f"P{i}": class_id * 0.1 + row * 0.001 + i * 0.00001 for i in range(1, 31)})
            values.update({"Class": class_id, "Position": 0.1, "Reflectance": 0.2, "loss": 0.3})
            rows.append(values)
    return pd.DataFrame(rows)


def test_features_are_exact_and_forbidden_inputs_rejected() -> None:
    validate_model_features(INPUT_COLUMNS)
    with pytest.raises(ValueError):
        validate_model_features([*INPUT_COLUMNS, "loss"])


def test_outer_and_enrollment_groups_are_disjoint_and_holdouts_absent() -> None:
    prepared = prepare_fold(_frame(), holdout=(1, 7), seed=42)
    parts = [prepared.outer.train, prepared.outer.validation, prepared.outer.seen_test,
             prepared.enrollment.support_pool, prepared.enrollment.query]
    groups = [set(part["_input_group"]) for part in parts]
    assert all(not (groups[i] & groups[j]) for i in range(len(groups)) for j in range(i + 1, len(groups)))
    assert not set(prepared.outer.train["Class"]) & {1, 7}
    assert not set(prepared.outer.validation["Class"]) & {1, 7}


def test_inner_folds_are_group_disjoint_class_valid_and_deterministic() -> None:
    prepared = prepare_fold(_frame(), holdout=(2, 6), seed=123)
    first = list(group_stratified_inner_splits(prepared.outer.train, seed=9))
    second = list(group_stratified_inner_splits(prepared.outer.train, seed=9))
    for (train_a, val_a), (train_b, val_b) in zip(first, second):
        assert np.array_equal(train_a, train_b) and np.array_equal(val_a, val_b)
        assert not set(prepared.outer.train.iloc[train_a]["_input_group"]) & set(prepared.outer.train.iloc[val_a]["_input_group"])
        assert set(prepared.outer.train.iloc[train_a]["Class"]) == set(prepared.outer.train.iloc[val_a]["Class"])


def test_support_draws_deterministic_and_reference_counts() -> None:
    labels = np.repeat([1, 4], 10)
    left = deterministic_support_indices(labels, (1, 4), count=3, draw=2, seed=42)
    right = deterministic_support_indices(labels, (1, 4), count=3, draw=2, seed=42)
    assert np.array_equal(left, right)
    assert np.bincount(labels[left])[1] == 3 and np.bincount(labels[left])[4] == 3
    embeddings = torch.nn.functional.normalize(torch.randn(20, 16), dim=-1)
    prototypes = class_prototypes(embeddings, torch.from_numpy(labels), [1, 4], strategy="equal", count=3)
    assert prototypes.shape == (2, 16)


def test_knn_density_scores_have_expected_shape_and_order() -> None:
    embeddings = torch.tensor([[0.0, 0.0], [0.1, 0.0], [4.0, 4.0], [4.1, 4.0]])
    labels = torch.tensor([0, 0, 1, 1])
    scorer = DensityScorer(embeddings, labels, [0, 1], density="knn", shrinkage=0.1, knn_k=1)
    distances = scorer.distances(torch.tensor([[0.05, 0.0], [4.05, 4.0]]))
    assert distances.shape == (2, 2)
    assert distances[0, 0] < distances[0, 1] and distances[1, 1] < distances[1, 0]


def test_cuda_enforcement() -> None:
    with pytest.raises(ValueError):
        require_cuda("cpu")
    if torch.cuda.is_available():
        assert require_cuda("cuda:0").type == "cuda"


@pytest.mark.parametrize("kind", ["a", "b", "c"])
def test_all_models_forward_backward(kind: str) -> None:
    x = torch.randn(12, 2, 30)
    if kind == "a":
        model = EpisodicMetricModel(class_count=6, embedding_dim=64)
        logits, z = model(x)
        loss = logits.square().mean() + z.square().mean()
    elif kind == "b":
        model = PhysicsSemanticModel(attribute_dim=12, latent_dim=64)
        logits, attributes, z = model(x, torch.rand(6, 12))
        loss = logits.square().mean() + attributes.square().mean() + z.square().mean()
    else:
        model = SelfSupervisedTraceModel(embedding_dim=64)
        z, reconstruction = model(x)
        loss = z.square().mean() + reconstruction.square().mean()
    loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad)


def test_physics_semantic_prototypes_validate() -> None:
    path = Path(__file__).parents[1] / "src" / "corpus" / "otdr_physics_prototypes.json"
    attributes, names, matrix = load_physics_prototypes(path)
    assert len(attributes) == 12 and len(names) == 8 and matrix.shape == (8, 12)
    assert torch.unique(matrix, dim=0).shape[0] == 8


def test_conformal_calibration_is_finite_sample_and_monotone() -> None:
    calibration = np.arange(1.0, 101.0)
    values = conformal_p_values(calibration, np.asarray([0.0, 50.0, 200.0]))
    assert np.all(np.diff(values) <= 0)
    assert values[-1] == pytest.approx(1 / 101)
    threshold = threshold_at_normal_far(values[:2], 0.01)
    assert np.mean(values[:2] < threshold) <= 0.01


def test_open_set_metrics_executes_oscr_with_current_numpy() -> None:
    metrics = open_set_metrics(
        is_known=np.asarray([True, True, False, False]),
        confidence=np.asarray([0.9, 0.6, 0.4, 0.1]),
        predicted=np.asarray([0, 1, 0, 1]),
        true_labels=np.asarray([0, 1, 2, 2]),
        threshold=0.5,
    )
    assert 0.0 <= metrics["oscr_auc"] <= 1.0


def test_artifact_hash_and_resume_validation(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "metrics.json").write_text('{"value": 1}\n', encoding="utf-8")
    write_manifest(run, {"run_id": "stable"})
    assert validate_run(run, {"run_id": "stable"}) == (True, "valid")
    (run / "metrics.json").write_text('{"value": 2}\n', encoding="utf-8")
    valid, reason = validate_run(run, {"run_id": "stable"})
    assert not valid and "mismatch" in reason


def test_run_ids_and_metrics_reconstruct_independently() -> None:
    config = ApproachAConfig(seed=42)
    assert stable_run_id("a", (1, 2), 42, config) == stable_run_id("a", (1, 2), 42, replace(config))
    y_true = np.asarray([0, 0, 1, 1, 2, 2])
    y_pred = np.asarray([0, -1, 1, 0, 2, 1])
    metrics = post_enrollment_metrics(y_true, y_pred, seen_ids=[0, 1], unseen_ids=[2])
    seen = np.mean([0.5, 0.5])
    unseen = 0.5
    assert metrics["seen_accuracy"] == pytest.approx(seen)
    assert metrics["unseen_accuracy"] == pytest.approx(unseen)
    assert metrics["harmonic_mean"] == pytest.approx(0.5)


def test_balanced_accuracy_dependency_used_by_semantic_sweep() -> None:
    from src import study_sweep
    assert study_sweep.balanced_accuracy_score(np.asarray([1, 1, 2, 2]), np.asarray([1, 2, 2, 2])) == pytest.approx(0.75)


def test_study_cli_exposes_required_commands() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    for command in ("audit", "sweep", "benchmark", "analyze"):
        assert command in result.output
