from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from OTDR_CLI.OTDR.src.event_openworld_data import (
    attach_input_groups,
    build_event_openworld_fold,
    fit_tensor_fold,
    validate_fold_isolation,
    write_exact_group_manifest,
)
from OTDR_CLI.OTDR.src.event_openworld_baselines import _balanced_indices as closed_set_balanced_indices
from OTDR_CLI.OTDR.src.event_openworld_analysis import _rate_difference_in_samples, paired_previous_comparisons
from OTDR_CLI.OTDR.src.event_openworld_graph import graph_propagate, mutual_knn_graph, seeded_graph_enrollment
from OTDR_CLI.OTDR.src.event_openworld_external import _evaluate_external, _external_summary_rows
from OTDR_CLI.OTDR.src.event_openworld_metrics import NormalOnlyCalibrator, evaluate_zero_day, raw_partial_auroc
from OTDR_CLI.OTDR.src.event_openworld_training import PC2Config, SGMEConfig, cvar_ranking_loss, train_pc2_oe
from OTDR_CLI.OTDR.src.event_openworld_sweep import _inner_sgme_task, _inner_task
from OTDR_CLI.OTDR.src.model_functions.event_openworld import (
    EventCompositionalModel,
    PhysicsEventRenderer,
    load_event_recipes,
    novelty_components,
)
from OTDR_CLI.OTDR.src.model_functions.zero_shot import require_cuda
from OTDR_CLI.OTDR.src.study_state import atomic_json, validate_run, write_manifest
from OTDR_CLI.OTDR.src.zero_shot_data import FORBIDDEN_FEATURES, INPUT_COLUMNS


ROOT = Path(__file__).resolve().parents[1]
RECIPE_PATH = ROOT / "experiments" / "otdr_event_openworld_study" / "configs" / "event_recipes.json"


def small_frame(groups_per_class: int = 80) -> pd.DataFrame:
    rows = []
    for class_id in range(8):
        for group in range(groups_per_class):
            row = {"Class": class_id, "SNR": 10 + class_id + group / 1000}
            row.update({f"P{i}": class_id * 0.1 + group * 0.001 + i * 0.00001 for i in range(1, 31)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_exact_feature_whitelist_is_frozen() -> None:
    protocol = json.loads((RECIPE_PATH.parent / "protocol.json").read_text(encoding="utf-8"))
    assert protocol["features"] == INPUT_COLUMNS
    assert set(protocol["forbidden"]) == FORBIDDEN_FEATURES
    assert not set(INPUT_COLUMNS) & FORBIDDEN_FEATURES


def test_duplicate_groups_and_all_partitions_are_isolated() -> None:
    frame = small_frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    grouped = attach_input_groups(frame)
    assert grouped["_input_group"].nunique() == len(frame) - 1
    fold = build_event_openworld_fold(grouped, holdout=(2, 6), seed=42)
    validate_fold_isolation(fold)
    parts = fold.partitions()
    groups = {name: set(value["_input_group"]) for name, value in parts.items()}
    for left, left_name in enumerate(groups):
        for right_name in list(groups)[left + 1:]:
            assert groups[left_name].isdisjoint(groups[right_name])
    assert not (set(fold.train["Class"]) | set(fold.validation["Class"])) & {2, 6}
    assert set(fold.reference_pool["Class"]) == {2, 6}
    assert set(fold.adaptation_pool["Class"]) == {2, 6}
    assert set(fold.query["Class"]) == {2, 6}


def test_conflicting_duplicate_group_is_rejected() -> None:
    frame = small_frame(10)
    conflict = frame.iloc[[0]].copy()
    conflict["Class"] = 1
    with pytest.raises(ValueError, match="Conflicting-label"):
        attach_input_groups(pd.concat([frame, conflict], ignore_index=True))


def test_forbidden_metadata_cannot_change_splits_scaling_or_model_tensors() -> None:
    left = small_frame()
    left["Position"] = np.arange(len(left))
    left["Loss"] = np.linspace(-100, 100, len(left))
    left["Reflectance"] = 0.0
    right = left.copy()
    right[["Position", "Loss", "Reflectance"]] = np.random.default_rng(9).normal(size=(len(right), 3))
    left_fold = fit_tensor_fold(build_event_openworld_fold(left, holdout=(1, 7), seed=123))
    right_fold = fit_tensor_fold(build_event_openworld_fold(right, holdout=(1, 7), seed=123))
    assert left_fold.split.partitions().keys() == right_fold.split.partitions().keys()
    for name in left_fold.tensors:
        assert torch.equal(left_fold.tensors[name][0], right_fold.tensors[name][0])
        assert torch.equal(left_fold.tensors[name][1], right_fold.tensors[name][1])


def test_inner_selector_has_two_nontrivial_pseudo_unseen_classes() -> None:
    outer = build_event_openworld_fold(small_frame(100), holdout=(1, 2), seed=42)
    task = _inner_task(outer.train, outer.validation, (3, 4))
    assert not set(task["train"]["Class"]) & {3, 4}
    assert set(task["support"]["Class"]) == {3, 4}
    assert set(task["adaptation"]["Class"]) == {3, 4}
    assert set(task["pseudo_query"]["Class"]) == {3, 4}
    names = list(task)
    group_sets = {name: set(value["_input_group"]) for name, value in task.items()}
    for left, left_name in enumerate(names):
        for right_name in names[left + 1:]:
            assert group_sets[left_name].isdisjoint(group_sets[right_name])


def test_outer_closed_encoder_sampler_never_requires_heldout_classes() -> None:
    labels = torch.tensor([0] * 20 + [2] * 20 + [4] * 20 + [7] * 20)
    indices = closed_set_balanced_indices(labels, 32, np.random.default_rng(4))
    sampled = labels[indices]
    assert set(sampled.tolist()) == {0, 2, 4, 7}
    assert all(int((sampled == class_id).sum()) == 8 for class_id in (0, 2, 4, 7))


def test_sgme_inner_support_adaptation_and_query_are_exactly_disjoint() -> None:
    outer = build_event_openworld_fold(small_frame(180), holdout=(1, 2), seed=42)
    task = _inner_sgme_task(outer.train, outer.validation, (3, 4))
    assert not set(task["train"]["Class"]) & {3, 4}
    assert set(task["support"]["Class"]) == {3, 4}
    assert set(task["adaptation"]["Class"]) == {3, 4}
    assert set(task["pseudo_query"]["Class"]) == {3, 4}
    groups = {name: set(part["_input_group"]) for name, part in task.items()}
    for left, left_name in enumerate(groups):
        for right_name in list(groups)[left + 1:]:
            assert groups[left_name].isdisjoint(groups[right_name])


def test_exact_split_manifest_round_trips_every_group(tmp_path: Path) -> None:
    fold = build_event_openworld_fold(small_frame(80), holdout=(2, 5), seed=7)
    path = write_exact_group_manifest(fold, tmp_path / "groups.npz")
    with np.load(path) as payload:
        for name, part in fold.partitions().items():
            recovered = {bytes(value).hex() for value in payload[name]}
            assert recovered == set(part["_input_group"])


def test_event_centering_is_deterministic_and_soft_alignment_has_gradients() -> None:
    recipes = load_event_recipes(RECIPE_PATH)
    torch.manual_seed(11)
    model = EventCompositionalModel(width=24, latent_dim=20, patch_size=11, soft_alignment=True)
    features = torch.randn(6, 31, requires_grad=True)
    left = model(features, recipes["means"], recipes["stds"])
    right = model(features, recipes["means"], recipes["stds"])
    assert torch.equal(left["center"], right["center"])
    left["logits"].sum().backward()
    assert features.grad is not None and float(features.grad.abs().sum()) > 0
    assert model.canonicalizer.saliency[0].weight.grad is not None
    assert float(model.canonicalizer.saliency[0].weight.grad.abs().sum()) > 0


def test_bounded_residual_penalty_is_nonconstant_and_has_gradients() -> None:
    recipes = load_event_recipes(RECIPE_PATH)
    torch.manual_seed(17)
    model = EventCompositionalModel(width=24, latent_dim=20, residual_enabled=True)
    features = torch.randn(12, 31)
    model.residual_scale.data.fill_(10)
    output = model(features, recipes["means"], recipes["stds"])
    norms = output["residual_norm"]
    assert torch.all(norms <= 1 + 1e-6)
    assert float(norms.std().detach()) > 1e-5
    norms.mean().backward()
    assert model.trace_residual.weight.grad is not None
    assert float(model.trace_residual.weight.grad.abs().sum()) > 0
    original_logits = output["logits"].detach()
    with torch.no_grad():
        model.trace_residual.weight.zero_()
        model.trace_residual.bias.zero_()
    zero_residual_logits = model(features, recipes["means"], recipes["stds"])["logits"].detach()
    assert float((original_logits - zero_residual_logits).abs().max()) > 1e-5


def test_front_end_retains_bounded_absolute_energy_information() -> None:
    model = EventCompositionalModel(width=24, latent_dim=20)
    with torch.no_grad():
        for parameter in model.snr_gate.parameters():
            parameter.zero_()
    base = torch.sin(torch.linspace(0, 3.14, 30))[None, :].repeat(4, 1)
    low = torch.cat([torch.zeros(4, 1), base * 0.25], dim=1)
    high = torch.cat([torch.zeros(4, 1), base * 2.0], dim=1)
    low_energy = model._channels(low).square().mean((1, 2)).sqrt()
    high_energy = model._channels(high).square().mean((1, 2)).sqrt()
    assert torch.all(high_energy > low_energy)
    assert torch.all(high_energy / low_energy < 2.0)  # bounded conditioning, not raw amplification


def test_zero_day_novelty_ignores_semantic_only_unseen_recipes() -> None:
    recipes = load_event_recipes(RECIPE_PATH)
    torch.manual_seed(31)
    model = EventCompositionalModel(width=24, latent_dim=20)
    output = model(torch.randn(10, 31), recipes["means"], recipes["stds"])
    known = [0, 1, 2, 3, 4, 5]
    before = novelty_components(output, known)
    changed = dict(output)
    changed["logits"] = output["logits"].clone()
    changed["reconstruction_residual_per_class"] = output["reconstruction_residual_per_class"].clone()
    changed["logits"][:, 6:] += 1000
    changed["reconstruction_residual_per_class"][:, 6:] = 0
    after = novelty_components(changed, known)
    assert torch.equal(before, after)
    assert not torch.allclose(novelty_components(output), novelty_components(changed))


def test_renderer_is_seeded_bounded_and_atom_morphology_is_directional() -> None:
    recipes = load_event_recipes(RECIPE_PATH)
    renderer = PhysicsEventRenderer(recipes["means"], recipes["stds"], snr_mean=20, snr_scale=5)
    left, right = renderer._different_fault_pairs(4096, torch.Generator().manual_seed(8))
    assert torch.all(left != right)
    assert set(left.tolist()) == set(range(1, 8)) == set(right.tolist())
    labels = torch.tensor([0] * 128 + [5] * 128 + [7] * 128)
    one = renderer.render_named(labels, generator=torch.Generator().manual_seed(9))[0]
    two = renderer.render_named(labels, generator=torch.Generator().manual_seed(9))[0]
    assert torch.equal(one, two)
    assert torch.isfinite(one).all() and one.shape == (384, 31)
    normal, cut, reflector = one[:128, 1:], one[128:256, 1:], one[256:, 1:]
    assert reflector.amax(1).mean() > normal.amax(1).mean()
    assert cut[:, -3:].mean() < normal[:, -3:].mean()
    boundary, factors = renderer.render_boundary(64, generator=torch.Generator().manual_seed(3))
    assert boundary.shape == (64, 31)
    assert torch.all((factors >= 0) & (factors <= 1))
    centered = boundary[:, 1:] - boundary[:, 1:].mean(1, keepdim=True)
    rendered_rms = centered.square().mean(1).sqrt()
    expected_rms = renderer.trace_rms_target * (0.4 + 1.2 * factors[:, 9])
    assert torch.allclose(rendered_rms, expected_rms, atol=2e-5, rtol=2e-5)
    base = recipes["means"][6:7].repeat(64, 1)
    for factor_index in (5, 7, 8, 9):
        low, high = base.clone(), base.clone()
        low[:, factor_index], high[:, factor_index] = 0.05, 0.95
        low_trace = renderer._render(low, generator=torch.Generator().manual_seed(81))[:, 1:]
        high_trace = renderer._render(high, generator=torch.Generator().manual_seed(81))[:, 1:]
        assert float((low_trace - high_trace).abs().mean()) > 1e-3
    payload = recipes["payload"]
    assert payload["basis"].startswith("General OTDR morphology")
    assert "real" not in PhysicsEventRenderer.__init__.__code__.co_varnames


def test_cvar_focuses_on_hardest_ranking_violations() -> None:
    normal = torch.tensor([0.0, 0.0, 0.0, 0.0])
    outlier = torch.tensor([2.0, 1.0, 0.0, -1.0])
    full = cvar_ranking_loss(normal, outlier, margin=1.0, fraction=1.0)
    tail = cvar_ranking_loss(normal, outlier, margin=1.0, fraction=0.25)
    assert tail >= full
    assert tail == pytest.approx(2.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-specific training regression")
def test_pc2_small_batch_keeps_real_and_synthetic_losses_finite() -> None:
    recipes = load_event_recipes(RECIPE_PATH)
    features = torch.randn(240, 31)
    labels = torch.arange(240) % 6
    _, metadata = train_pc2_oe(
        features, labels, recipes["means"], recipes["stds"], snr_mean=20, snr_scale=5,
        device=torch.device("cuda:0"),
        config=PC2Config(epochs=1, steps_per_epoch=1, batch_size=32, width=24, latent_dim=20, seed=19),
    )
    assert all(np.isfinite(value) for key, value in metadata["history"][0].items() if key != "epoch")


def test_calibration_uses_only_supplied_normals_and_small_bins_fall_back() -> None:
    normal_score = np.linspace(-1, 1, 500)
    normal_snr = np.repeat(np.arange(5), 100)
    global_cal = NormalOnlyCalibrator("global").fit(normal_score, normal_snr)
    first = global_cal.threshold(np.array([0, 100]), 0.01)
    second = global_cal.threshold(np.array([0, 100]), 0.01)
    assert np.array_equal(first, second)
    mondrian = NormalOnlyCalibrator("mondrian", bins=10, minimum_bin_size=128).fit(normal_score, normal_snr)
    thresholds = mondrian.threshold(np.array([-100, 100]), 0.01)
    assert thresholds[0] == thresholds[1]
    assert mondrian.describe()["small_bin_fallback"] == "global"


def test_partial_auc_and_zero_day_metrics_reconstruct() -> None:
    labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    score = np.array([0.0, 0.1, 0.2, 0.3, 0.7, 0.8, 0.9, 1.0])
    assert raw_partial_auroc(labels, score, 0.05) == pytest.approx(1.0)
    true = np.array([0] * 200 + [1] * 100 + [2] * 100)
    test_score = np.r_[np.linspace(-1, 0, 200), np.linspace(0.5, 1.0, 200)]
    result = evaluate_zero_day(
        validation_normal_score=np.linspace(-1, 0, 300), validation_normal_snr=np.linspace(-2, 2, 300),
        test_score=test_score, test_snr=np.linspace(-2, 2, 400), true_labels=true,
        predicted=true, holdout=(1, 2), calibration="global",
    )
    assert result["auroc"] == pytest.approx(1.0)
    assert result["operating_points"]["far_0.010"]["unknown_recall"] == pytest.approx(1.0)


def test_external_reflection_attenuation_score_uses_amax_tensors() -> None:
    logits = torch.zeros(6, 8)
    logits[:2, 0] = 4.0
    logits[2:4, 4] = 4.0
    logits[4:, 1] = 4.0
    result = _evaluate_external(
        {"logits": logits, "center": torch.full((6,), 14.5)},
        np.array([0, 0, 1, 1, 2, 2]),
        [{} for _ in range(6)],
        np.array([0.0, 0.1, 0.8, 0.9, 0.7, 0.6]),
    )
    assert result["reflection_vs_attenuation_auroc_assuming_1_reflection_2_attenuation"] == pytest.approx(1.0)


def test_external_summary_rows_keep_performance_and_calibration_tasks_distinct() -> None:
    performance = {
        "examples": 6, "event_examples": 4, "no_event_examples": 2, "event_auroc": 0.7,
        "event_pauroc_0_05": 0.1, "event_balanced_accuracy_at_0_5": 0.6,
        "event_location_mae_bins": 0.2, "zero_day_novelty_auroc_for_event_vs_no_event": 0.55,
        "zero_day_novelty_pauroc_0_05": 0.02,
    }
    operating = {"far_0.010": {"target_far": 0.01, "observed_no_event_far": 0.02, "event_recall": 0.3}}
    rows = _external_summary_rows({"approaches": {"ec": {
        "simulated": performance, "measured": performance,
        "local_normal_calibration_transfer": {"simulated": operating, "measured": operating},
        "synthetic_no_event_to_measured_calibration_transfer": operating,
    }}})
    assert [row["row_type"] for row in rows].count("performance") == 2
    assert [row["row_type"] for row in rows].count("calibration_transfer") == 3
    assert {row.get("calibration_source") for row in rows if row["row_type"] == "calibration_transfer"} == {
        "local_normal", "simulated_no_event"
    }


def test_paired_comparisons_handle_metrics_with_matching_column_names(tmp_path: Path) -> None:
    study_root = tmp_path / "otdr_event_openworld_study"
    (study_root / "tables").mkdir(parents=True)
    previous_tables = tmp_path / "otdr_three_approach_study" / "tables"
    previous_tables.mkdir(parents=True)
    rows = []
    previous_rows = []
    for fold in ("1-2", "1-3", "1-4", "1-5", "1-6"):
        for seed in (42, 123, 2026):
            rows.append({"approach": "ec", "fold": fold, "seed": seed,
                         "far_0.010_unknown_recall": 0.4, "strict_balanced": 0.6, "gzsl_h": 0.3})
            for approach in ("a", "b", "c"):
                previous_rows.append({"approach": approach, "fold": fold, "seed": seed,
                                      "pre_unknown_recall": 0.2, "strict_balanced": 0.5,
                                      "gzsl_h": 0.2, "post_h": 0.3})
    pd.DataFrame(previous_rows).to_csv(previous_tables / "per_run_results.csv", index=False)
    result = paired_previous_comparisons(
        {"runs": pd.DataFrame(rows), "inductive": pd.DataFrame(), "sgme": pd.DataFrame()}, study_root
    )
    assert set(result["metric"]) == {"unknown_recall", "strict_zsl", "gzsl_h"}
    assert len(result) == 4


def test_reconstruction_uses_a_one_sample_rate_tolerance() -> None:
    assert _rate_difference_in_samples(1 / 13061, 13061) == pytest.approx(1.0)
    assert _rate_difference_in_samples(2 / 13061, 13061) == pytest.approx(2.0)
    with pytest.raises(ValueError, match="denominator"):
        _rate_difference_in_samples(0.0, 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA-specific integrity test")
def test_graph_propagation_is_deterministic_and_confidence_guards_apply() -> None:
    device = require_cuda("cuda:0")
    torch.manual_seed(5)
    seen = torch.randn(16, 12)
    seen_y = torch.tensor([0] * 8 + [1] * 8)
    refs = F_normalize(torch.randn(4, 12))
    refs_y = torch.tensor([2, 2, 3, 3])
    adaptation = torch.cat([refs[:1] + 0.01 * torch.randn(6, 12), refs[2:3] + 0.01 * torch.randn(6, 12)])
    semantic = torch.zeros(12, 8)
    semantic[:6, 2] = 0.95
    semantic[6:, 3] = 0.95
    augmentation = semantic.clone()
    config = SGMEConfig(k_neighbors=3, confidence_threshold=0.4, agreement_threshold=0.3,
                        augmentation_threshold=0.7, semantic_threshold=0.5, seen_rejection_threshold=0.0)
    kwargs = dict(seen_anchor_embeddings=seen, seen_anchor_labels=seen_y, reference_embeddings=refs,
                  reference_labels=refs_y, adaptation_embeddings=adaptation,
                  semantic_probabilities=semantic, augmentation_probabilities=augmentation,
                  holdout=(2, 3), device=device, config=config)
    left = seeded_graph_enrollment(**kwargs)
    right = seeded_graph_enrollment(**kwargs)
    assert torch.equal(left.accepted_indices, right.accepted_indices)
    bad = seeded_graph_enrollment(**{**kwargs, "augmentation_probabilities": torch.zeros_like(augmentation)})
    assert len(bad.accepted_indices) == 0


def F_normalize(value: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(value, dim=-1)


def test_manifest_tampering_is_detected_and_resume_validation_is_stable(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    atomic_json(run / "metrics.json", {"value": 1.0})
    write_manifest(run, {"run_id": "unit-run", "cuda": True})
    assert validate_run(run, {"run_id": "unit-run"}) == (True, "valid")
    atomic_json(run / "metrics.json", {"value": 2.0})
    valid, reason = validate_run(run, {"run_id": "unit-run"})
    assert not valid and "artifact mismatch" in reason


def test_cuda_enforcement_rejects_cpu() -> None:
    with pytest.raises((RuntimeError, ValueError)):
        require_cuda("cpu")
