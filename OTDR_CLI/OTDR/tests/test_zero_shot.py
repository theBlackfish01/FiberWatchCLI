from __future__ import annotations

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from click.testing import CliRunner

from src.model_functions.tcn import OTDR_TCN
from src.model_functions.zero_shot import (
    OTDRZeroShotEncoder,
    aggregate_prototype_scores,
    require_cuda,
)
from src.zero_shot_data import (
    INPUT_COLUMNS,
    build_outer_fold,
    load_fault_prototypes,
)
from src.zero_shot_training import (
    apply_seen_penalty,
    choose_seen_penalty,
    compute_gzsl_metrics,
    fit_seen_scaler,
    gpu_metadata,
    _load_sentence_encoder,
    transform_frame,
)
from src.zero_shot import cli, fault_pairs


def _frame(rows_per_class: int = 12) -> pd.DataFrame:
    rows = []
    for cls in range(8):
        for idx in range(rows_per_class):
            row = {"Class": cls, "Position": float(idx), "loss": cls, "Reflectance": -cls}
            row["SNR"] = float(cls * 100 + idx)
            row.update({f"P{i}": float(cls * 1000 + idx * 10 + i) for i in range(1, 31)})
            rows.append(row)
    # Exact duplicate input must remain in the same partition.
    rows.append(dict(rows[0]))
    rows[-1]["Position"] = 999.0
    return pd.DataFrame(rows)


def test_prototypes_define_five_descriptions_for_all_classes(tmp_path: Path) -> None:
    path = tmp_path / "prototypes.json"
    payload = {
        "schema_version": 1,
        "classes": [
            {"id": cls, "name": f"class-{cls}", "descriptions": [f"class {cls} view {i}" for i in range(5)]}
            for cls in range(8)
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")

    prototypes = load_fault_prototypes(path)

    assert [item.class_id for item in prototypes] == list(range(8))
    assert all(len(item.descriptions) == 5 for item in prototypes)


def test_outer_fold_excludes_unseen_classes_and_forbidden_features() -> None:
    fold = build_outer_fold(_frame(), holdout=(1, 2), seed=42)

    assert INPUT_COLUMNS == ["SNR", *[f"P{i}" for i in range(1, 31)]]
    assert not {1, 2}.intersection(set(fold.train["Class"].unique()))
    assert set(fold.unseen_test["Class"].unique()) == {1, 2}
    assert not {"loss", "Reflectance", "Position", "Class"}.intersection(fold.feature_columns)

    partitions = [fold.train, fold.validation, fold.seen_test, fold.unseen_test]
    group_sets = [set(part["_input_group"]) for part in partitions]
    for left in range(len(group_sets)):
        for right in range(left + 1, len(group_sets)):
            assert group_sets[left].isdisjoint(group_sets[right])


def test_outer_fold_rejects_identical_inputs_with_conflicting_labels() -> None:
    frame = _frame()
    conflicting = frame.iloc[0].copy()
    conflicting["Class"] = 1
    frame = pd.concat([frame, conflicting.to_frame().T], ignore_index=True)
    with pytest.raises(ValueError, match="conflicting class labels"):
        build_outer_fold(frame, holdout=(1, 2), seed=42)


def test_zero_shot_encoder_and_prototype_scores_are_normalized() -> None:
    model = OTDRZeroShotEncoder(in_ch=2, embedding_dim=16)
    inputs = torch.randn(4, 2, 30)
    embeddings = model(inputs)
    prototypes = torch.nn.functional.normalize(torch.randn(8, 5, 16), dim=-1)

    scores = aggregate_prototype_scores(embeddings, prototypes, temperature=torch.tensor(0.1))

    assert embeddings.shape == (4, 16)
    assert torch.allclose(embeddings.norm(dim=1), torch.ones(4), atol=1e-5)
    assert scores.shape == (4, 8)
    assert torch.isfinite(scores).all()


def test_tcn_state_dict_keys_remain_checkpoint_compatible() -> None:
    model = OTDR_TCN()
    clone = OTDR_TCN()
    clone.load_state_dict(model.state_dict(), strict=True)
    assert "tcn.0.net.0.weight" in model.state_dict()
    assert "attn_pool.score.0.weight" in model.state_dict()


def test_require_cuda_rejects_cpu() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        require_cuda("cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_require_cuda_runs_real_cuda_forward_backward() -> None:
    device = require_cuda("cuda:0")
    model = OTDRZeroShotEncoder(in_ch=2, embedding_dim=16).to(device)
    batch = torch.randn(8, 2, 30, device=device)
    result = model(batch)
    loss = result.square().mean()
    loss.backward()

    assert device.type == "cuda"
    assert result.is_cuda
    assert next(model.parameters()).grad is not None


def test_scaler_is_fit_only_on_seen_training_rows() -> None:
    fold = build_outer_fold(_frame(), holdout=(1, 2), seed=42)
    scaler = fit_seen_scaler(fold)
    expected = fold.train[INPUT_COLUMNS].to_numpy(dtype=np.float32).mean(axis=0)
    assert np.allclose(scaler.mean_, expected)


def test_transformed_tensors_are_writable_without_numpy_warning() -> None:
    fold = build_outer_fold(_frame(), holdout=(1, 2), seed=42)
    scaler = fit_seen_scaler(fold)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        features, labels = transform_frame(fold.train, scaler)
        labels[0] = labels[0]
        features[0, 0] = features[0, 0]
    assert not caught


def test_seen_penalty_calibration_maximizes_harmonic_mean() -> None:
    scores = torch.tensor([[4.0, 3.0], [3.0, 4.0], [4.0, 3.8], [4.0, 4.2]])
    labels = np.array([0, 1, 1, 1])
    gamma, rows = choose_seen_penalty(scores, labels, seen_class_ids={0}, candidate_class_ids=[0, 1])
    adjusted = apply_seen_penalty(scores, gamma, seen_class_ids={0}, candidate_class_ids=[0, 1])
    assert rows
    assert adjusted.argmax(1).tolist() == [0, 1, 1, 1]


def test_gzsl_metrics_report_seen_unseen_harmonic_mean() -> None:
    metrics = compute_gzsl_metrics(
        y_true=np.array([0, 0, 1, 1]),
        y_pred=np.array([0, 0, 1, 0]),
        seen_class_ids={0},
        unseen_class_ids={1},
    )
    assert metrics["seen_accuracy"] == 1.0
    assert metrics["unseen_accuracy"] == 0.5
    assert metrics["harmonic_mean"] == pytest.approx(2.0 / 3.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_metadata_identifies_actual_cuda_device() -> None:
    metadata = gpu_metadata(require_cuda("cuda:0"))
    assert metadata["device"] == "cuda:0"
    assert metadata["gpu_name"] == torch.cuda.get_device_name(0)
    assert metadata["gpu_name"]
    assert metadata["cuda_available"] is True


def test_fault_pairs_cover_every_unordered_pair() -> None:
    pairs = fault_pairs()
    assert len(pairs) == 21
    assert pairs[0] == (1, 2)
    assert pairs[-1] == (6, 7)
    assert all(sum(class_id in pair for pair in pairs) == 6 for class_id in range(1, 8))


def test_zero_shot_cli_exposes_train_evaluate_and_benchmark() -> None:
    result = CliRunner().invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "train-fold" in result.output
    assert "evaluate-fold" in result.output
    assert "benchmark" in result.output


def test_text_encoder_loads_cached_model_before_network_fallback() -> None:
    calls = []

    def factory(model_name, **kwargs):
        calls.append((model_name, kwargs))
        return object()

    result = _load_sentence_encoder("example/model", torch.device("cuda:0"), factory=factory)
    assert result is not None
    assert calls == [("example/model", {"device": "cuda:0", "local_files_only": True})]
