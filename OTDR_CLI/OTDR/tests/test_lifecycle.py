from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from OTDR_CLI.OTDR.src.lifecycle_data import (
    FEATURE_REGIMES,
    deterministic_support_indices,
    fit_lifecycle_fold,
    fit_lifecycle_scaler,
    lifecycle_split_manifest,
    split_known_calibration,
    transform_lifecycle,
    validate_feature_contract,
)
from OTDR_CLI.OTDR.src.lifecycle_baselines import NearestNeighborReference, balanced_context_indices
from OTDR_CLI.OTDR.src.lifecycle_domain import (
    apply_stress,
    canonicalize_batch,
    propose_event,
    synthetic_domain_view,
)
from OTDR_CLI.OTDR.src.lifecycle_enrollment import (
    EnrollmentSession,
    fit_distance_temperature,
    projection_adapter_predict,
    sequential_orders,
    support_prototype,
    teen_calibrate,
)
from OTDR_CLI.OTDR.src.lifecycle_scod import (
    EmpiricalCDFNormalizer,
    PrototypeBank,
    assemble_components,
    classifier_novelty_components,
    DistanceReference,
    evaluate_joint_operating_point,
    evaluate_grouped_operating_point,
    fit_joint_threshold,
    fit_joint_threshold_grouped,
    fuse_scores,
)
from OTDR_CLI.OTDR.src.lifecycle_sweep import (
    generate_cfe_candidates,
    generate_kpsc_candidates,
    generate_representation_candidates,
)
from OTDR_CLI.OTDR.src.lifecycle_physics import CoherentOTDRCounterfactuals, event_grammar_residual
from OTDR_CLI.OTDR.src.model_functions.event_openworld import load_event_recipes
from OTDR_CLI.OTDR.src.lifecycle_external import external_batch
from OTDR_CLI.OTDR.src.lifecycle_metrics import gate_diagnostics
from OTDR_CLI.OTDR.src import lifecycle_analysis
from OTDR_CLI.OTDR.src import lifecycle_experiment
from OTDR_CLI.OTDR.src.lifecycle_synthesis import (
    nested_two_key_mapping,
    symmetric_color_limit,
)
from OTDR_CLI.OTDR.src.lifecycle_tabpfn import (
    _balanced_query_indices,
    _ranked_indices,
)
from OTDR_CLI.OTDR.src.tabpfn_full_study import (
    FROZEN_PROTOCOL_SHA256,
    PROTOCOL_PATH,
    _aligned_probability,
    _distance_probability,
    metric_row,
    reconstruct_row,
)
from OTDR_CLI.OTDR.src.tabpfn_full_analysis import (
    _per_fault_recall,
    _prototype_efficiency,
)
from OTDR_CLI.OTDR.src.study_state import file_sha256
from OTDR_CLI.OTDR.src.model_functions.lifecycle import FeatureAssistedOTDR, LifecycleModelConfig, coral_loss
from OTDR_CLI.OTDR.src.model_functions.zero_shot import require_cuda


ROOT = Path(__file__).resolve().parents[1]


def test_nested_two_key_mapping_is_json_native() -> None:
    import json

    payload = nested_two_key_mapping({
        ("projection", 1): {"harmonic_mean": 0.42},
        ("projection", 5): {"harmonic_mean": 0.66},
    })
    assert payload == {
        "projection": {
            "1": {"harmonic_mean": 0.42},
            "5": {"harmonic_mean": 0.66},
        }
    }
    json.dumps(payload)


def lifecycle_frame(groups_per_class: int = 30) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(8)
    for class_id in range(8):
        for group in range(groups_per_class):
            trace = class_id * 0.25 + np.sin(np.linspace(0, np.pi, 30)) + group * 0.001
            row = {
                "Class": class_id,
                "Position": group % 30,
                "SNR": 10 + class_id + group / 100,
                "loss": class_id * 0.5 + rng.normal(0, 0.01),
                "Reflectance": -50 + class_id + rng.normal(0, 0.01),
            }
            row.update({f"P{i + 1}": value for i, value in enumerate(trace)})
            rows.append(row)
    return pd.DataFrame(rows)


def test_lifecycle_feature_contract_rejects_targets_and_declares_three_regimes() -> None:
    frame = lifecycle_frame(3)
    assert set(FEATURE_REGIMES) == {"full", "trace_only", "summary_only"}
    for regime in FEATURE_REGIMES:
        validate_feature_contract(frame, regime=regime)
    with pytest.raises(ValueError, match="Targets cannot"):
        validate_feature_contract(frame, regime="full", requested_inputs=[*FEATURE_REGIMES["full"], "Class"])
    with pytest.raises(ValueError, match="Targets cannot"):
        validate_feature_contract(frame, regime="full", requested_inputs=["Position"])


def test_grouping_ignores_scalar_differences_and_all_lifecycle_partitions_are_isolated() -> None:
    frame = lifecycle_frame(30)
    duplicate = frame.iloc[[0]].copy()
    duplicate["loss"] += 99
    duplicate["Reflectance"] -= 99
    folded = fit_lifecycle_fold(pd.concat((frame, duplicate), ignore_index=True), holdout=(2, 6), seed=42)
    groups = {name: set(part["_input_group"]) for name, part in folded.split.partitions().items()}
    for left, left_name in enumerate(groups):
        for right_name in list(groups)[left + 1:]:
            assert groups[left_name].isdisjoint(groups[right_name])
    assert sum(len(part) for part in folded.split.partitions().values()) == len(frame)
    manifest = lifecycle_split_manifest(
        folded.split, data_path=ROOT / "src" / "data" / "OTDR_DATA.csv",
        regime="full",
    )
    assert all(
        values["rows"] >= values["groups"]
        for values in manifest["partitions"].values()
    )
    assert sum(
        values["duplicate_rows_beyond_first"]
        for values in manifest["partitions"].values()
    ) == 0


def test_scaler_handles_missing_operational_summary_with_explicit_indicators() -> None:
    frame = lifecycle_frame(4)
    frame.loc[0, "loss"] = np.nan
    frame.loc[1, "Reflectance"] = np.nan
    scaler = fit_lifecycle_scaler(frame, regime="full")
    batch = transform_lifecycle(frame, scaler)
    assert torch.isfinite(batch.context).all()
    assert batch.context_missing[0, 1] == 1
    assert batch.context_missing[1, 2] == 1
    trace_only = transform_lifecycle(frame, fit_lifecycle_scaler(frame, regime="trace_only"))
    assert torch.all(trace_only.context[:, 1:] == 0)
    assert torch.all(trace_only.context_missing[:, 1:] == 1)


def test_known_selector_calibration_split_and_support_draws_are_group_safe_and_deterministic() -> None:
    fold = fit_lifecycle_fold(lifecycle_frame(80), holdout=(1, 7), seed=7)
    left, right = split_known_calibration(fold.split.validation, seed=7)
    assert set(left["_input_group"]).isdisjoint(set(right["_input_group"]))
    first = deterministic_support_indices(
        fold.split.reference_pool, class_ids=(1, 7), shots=3, seed=7, draw=4
    )
    second = deterministic_support_indices(
        fold.split.reference_pool, class_ids=(1, 7), shots=3, seed=7, draw=4
    )
    assert np.array_equal(first, second)
    assert len(set(fold.split.reference_pool.loc[first, "_input_group"])) == 6


def test_feature_assisted_model_forward_and_gates_have_expected_shapes() -> None:
    model = FeatureAssistedOTDR(LifecycleModelConfig(width=24, context_width=12, embedding_dim=16, blocks=2))
    output = model(torch.randn(9, 30), torch.randn(9, 3), torch.zeros(9, 3))
    assert output["logits"].shape == (9, 8)
    assert output["position"].shape == (9,)
    assert output["embedding"].shape == (9, 16)
    assert output["gate"].shape == (9, 24)
    assert torch.allclose(output["embedding"].norm(dim=1), torch.ones(9), atol=1e-5)
    loss = output["logits"].sum() + output["position"].sum()
    loss.backward()
    assert model.morphology.backbone.tcn[0].net[0].weight.grad is not None
    canonical = FeatureAssistedOTDR(
        LifecycleModelConfig(width=24, context_width=12, embedding_dim=16, blocks=2, canonicalize=True)
    )
    canonical_output = canonical(torch.randn(4, 30), torch.randn(4, 3), torch.zeros(4, 3))
    canonical_output["logits"].sum().backward()
    assert canonical.morphology.proposal_projection.weight.grad is not None
    for pooling in ("mean", "attention", "self_attention"):
        pooled = FeatureAssistedOTDR(
            LifecycleModelConfig(
                width=24, context_width=12, embedding_dim=16, blocks=2,
                pooling=pooling,
            )
        )
        assert pooled(
            torch.randn(3, 30), torch.randn(3, 3), torch.zeros(3, 3)
        )["logits"].shape == (3, 8)


def test_multi_prototype_and_distance_components_are_finite_and_sensitive() -> None:
    rng = np.random.default_rng(4)
    base = np.vstack((rng.normal(-2, 0.1, (20, 4)), rng.normal(2, 0.1, (20, 4))))
    labels = np.repeat([0, 1], 20)
    single = PrototypeBank.fit(base, labels, prototypes_per_class=1)
    multi = PrototypeBank.fit(base, labels, prototypes_per_class=2)
    query = np.asarray([[-2, -2, -2, -2], [12, -12, 12, -12]], dtype=float)
    assert multi.novelty(query)[1] > multi.novelty(query)[0]
    reference = DistanceReference.fit(base, labels)
    logits = np.asarray([[5, 0], [0, 0]], dtype=float)
    names, components = assemble_components(
        logits=logits, embeddings=query, distance_reference=reference, prototype_bank=single
    )
    assert components.shape == (2, len(names))
    assert np.isfinite(components).all()


def test_empirical_normalization_fusions_and_joint_constraint_calibration() -> None:
    rng = np.random.default_rng(9)
    reference = rng.normal(size=(300, 4))
    normalizer = EmpiricalCDFNormalizer.fit(reference, ["a", "b", "c", "d"])
    transformed = normalizer.transform(reference)
    assert np.all((transformed > 0) & (transformed < 1))
    for method in ("confidence", "best_single", "weighted", "sirc", "meta_p", "robust_regret"):
        score = fuse_scores(transformed, method=method)
        assert score.shape == (300,) and np.isfinite(score).all()
    calibration_labels = np.r_[np.zeros(400, dtype=int), np.ones(600, dtype=int)]
    calibration_score = np.r_[np.linspace(0, 1, 400), np.linspace(0, 0.8, 600)]
    threshold = fit_joint_threshold(calibration_score, calibration_labels)
    assert threshold.calibration_normal_far <= 0.0125
    assert threshold.calibration_known_fault_acceptance >= 0.95
    result = evaluate_joint_operating_point(
        np.r_[calibration_score, np.ones(100)],
        np.r_[calibration_labels, np.full(50, 6), np.full(50, 7)],
        np.r_[calibration_labels, np.full(50, 6), np.full(50, 7)],
        holdout=(6, 7),
        calibration=threshold,
    )
    assert result["unknown_recall"] > 0
    assert len(result["normal_far_clopper_pearson_95"]) == 2


def test_group_weighted_calibration_does_not_count_duplicate_rows_as_independent() -> None:
    score = np.r_[np.full(100, 0.9), [0.1, 0.2, 0.3, 0.4], np.linspace(0.1, 0.7, 20)]
    labels = np.r_[np.zeros(104, dtype=int), np.ones(20, dtype=int)]
    groups = np.asarray(
        ["duplicated-normal"] * 100
        + [f"normal-{index}" for index in range(4)]
        + [f"fault-{index}" for index in range(20)]
    )
    threshold = fit_joint_threshold_grouped(
        score, labels, groups,
        normal_far_cap=0.25, known_acceptance_floor=0.5,
    )
    result = evaluate_grouped_operating_point(
        np.r_[score, [1.0, 1.0]],
        np.r_[labels, [6, 7]],
        np.r_[labels, [6, 7]],
        np.r_[groups, ["unknown-6", "unknown-7"]],
        holdout=(6, 7),
        calibration=threshold,
        bootstrap_iterations=100,
    )
    assert result["normal_independent_groups"] == 5
    assert result["group_weighted_normal_far"] != pytest.approx(
        float((score[:104] > threshold.threshold).mean())
    )


def test_enrollment_is_immutable_training_free_and_sequential_orders_are_both_tested() -> None:
    rng = np.random.default_rng(11)
    base = np.vstack((rng.normal(-1, 0.1, (20, 6)), rng.normal(1, 0.1, (20, 6))))
    labels = np.repeat([0, 2], 20)
    session = EnrollmentSession.from_base(base, labels, metric="cosine")
    support = rng.normal(0, 0.1, (3, 6))
    enrolled = session.enroll(7, support, method="median", support_group_ids=("a", "b", "c"))
    assert session.class_ids == (0, 2)
    assert enrolled.class_ids == (0, 2, 7)
    assert enrolled.enrollment_history[-1]["query_adaptation"] is False
    assert sequential_orders((3, 7)) == ((3, 7), (7, 3))
    one = support_prototype(support[:1], method="mean")
    assert np.allclose(one, support_prototype(support[:1], method="median"))
    calibrated = teen_calibrate(one, np.stack(session.prototypes), alpha=0.2)
    assert calibrated.shape == one.shape and np.isclose(np.linalg.norm(calibrated), 1)
    temperature = fit_distance_temperature(session, base, labels)
    probability = session.predict_proba(base, temperature=temperature)
    assert probability.shape == (40, 8)
    assert np.allclose(probability.sum(1), 1)


def test_domain_preprocessing_is_label_independent_deterministic_and_stresses_are_bounded() -> None:
    trace = np.zeros(30)
    trace[18:] = -2
    proposal = propose_event(trace)
    assert 14 <= proposal.center <= 22
    first, context = canonicalize_batch(np.stack((trace, trace)))
    second, _ = canonicalize_batch(np.stack((trace, trace)))
    assert np.array_equal(first, second)
    c = np.zeros((2, 3), dtype=float)
    stressed, scalar, missing = apply_stress(first, c, kind="missing_loss", severity=1, seed=3)
    assert stressed.shape == (2, 30)
    assert np.isnan(scalar[:, 1]).all() and missing[:, 1].all()
    torch_trace = torch.zeros(4, 30)
    torch_context = torch.zeros(4, 3)
    torch_missing = torch.zeros(4, 3)
    transformed = synthetic_domain_view(
        torch_trace, torch_context, torch_missing,
        generator=torch.Generator().manual_seed(3),
    )
    repeated = synthetic_domain_view(
        torch_trace, torch_context, torch_missing,
        generator=torch.Generator().manual_seed(3),
    )
    assert all(torch.equal(left, right) for left, right in zip(transformed, repeated))
    assert transformed[0].shape == torch_trace.shape


def test_coral_is_zero_for_identical_features() -> None:
    values = torch.randn(20, 8)
    assert float(coral_loss(values, values)) == pytest.approx(0.0, abs=1e-8)


def test_cuda_contract_rejects_cpu_even_on_cuda_host() -> None:
    with pytest.raises(ValueError, match="CUDA"):
        require_cuda("cpu")
    with pytest.raises(ValueError, match="CUDA"):
        projection_adapter_predict(
            np.zeros((2, 3)),
            np.asarray([0, 0]),
            np.ones((1, 3)),
            np.asarray([1]),
            np.zeros((1, 3)),
            device="cpu",
        )


def test_each_primary_family_has_at_least_24_unique_predeclared_candidates() -> None:
    representations = generate_representation_candidates()
    kpsc = generate_kpsc_candidates()
    cfe = generate_cfe_candidates()
    assert len(representations) >= 24
    assert len({item.candidate_id for item in representations}) == len(representations)
    assert len(kpsc) >= 24 and len({str(sorted(item.items())) for item in kpsc}) == len(kpsc)
    assert len(cfe) >= 24 and len({str(sorted(item.items())) for item in cfe}) == len(cfe)


def test_balanced_1nn_context_and_support_enrollment() -> None:
    base = np.asarray([[-2.0, 0], [-1.8, 0], [2.0, 0], [1.8, 0]])
    labels = np.asarray([0, 0, 2, 2])
    indices = balanced_context_indices(labels, total=4, seed=3)
    assert set(labels[indices]) == {0, 2}
    reference = NearestNeighborReference.fit(base, labels, metric="euclidean", max_reference=4)
    query = np.asarray([[0.0, 2.0], [-2.0, 0.0]])
    distance, prediction = reference.nearest(query)
    enrolled = reference.combine_support(
        query, distance, prediction,
        np.asarray([[0.0, 2.1]]), np.asarray([7]),
    )
    assert enrolled.tolist() == [7, 0]


def test_physics_counterfactuals_are_seeded_diverse_and_jointly_coherent() -> None:
    renderer = CoherentOTDRCounterfactuals()
    trace = torch.zeros(60, 30)
    context = torch.zeros(60, 3)
    missing = torch.zeros(60, 3)
    one = renderer.render(
        trace, context, missing,
        generator=torch.Generator().manual_seed(8), diverse=True,
    )
    two = renderer.render(
        trace, context, missing,
        generator=torch.Generator().manual_seed(8), diverse=True,
    )
    assert all(torch.equal(left, right) for left, right in zip(one, two))
    rendered_trace, rendered_context, _, clusters = one
    assert set(clusters.tolist()) == set(range(6))
    reflective = torch.isin(clusters, torch.tensor([0, 4, 5]))
    attenuating = torch.isin(clusters, torch.tensor([1, 2, 3, 4, 5]))
    assert torch.all(rendered_context[reflective, 2] > 0)
    assert torch.all(rendered_context[attenuating, 1] > 0)
    assert rendered_trace.abs().sum() > 0
    recipes = load_event_recipes(
        ROOT / "experiments" / "otdr_event_openworld_study" / "configs" / "event_recipes.json"
    )
    residual = event_grammar_residual(
        trace, context, recipes["means"], recipes["stds"],
        known_class_ids=(0, 1, 2, 3, 4, 5),
    )
    assert residual.shape == (60,) and np.isfinite(residual).all()


def test_external_adapter_keeps_taxonomy_isolated_and_marks_missing_summaries() -> None:
    features = np.zeros((3, 31), dtype=np.float32)
    metadata = [{"file": "a.sor", "center": index} for index in range(3)]
    batch = external_batch(features, metadata)
    assert torch.all(batch.labels == 0)  # placeholder only; archive labels stay outside local 0..7
    assert torch.all(batch.context_missing[:, 1:] == 1)
    assert len(set(batch.group_ids)) == 3
    scaler = fit_lifecycle_scaler(lifecycle_frame(3), regime="full")
    shaped = np.c_[
        np.zeros(3),
        np.tile(np.linspace(-3, 3, 30), (3, 1)),
    ].astype(np.float32)
    aligned = external_batch(
        shaped,
        metadata,
        lifecycle_scaler=scaler,
        source_snr_mean=12.0,
        source_snr_scale=3.0,
        trace_mode="source_range_aligned",
    )
    assert torch.isfinite(aligned.trace).all()
    assert torch.isfinite(aligned.context).all()
    assert torch.all(aligned.context_missing[:, 1:] == 1)


def test_analysis_discovers_only_the_requested_feature_regime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stage = tmp_path / "full_benchmark"
    for name, regime in (("full-run", "full"), ("trace-run", "trace_only")):
        run = stage / name
        run.mkdir(parents=True)
        (run / "manifest.json").write_text("{}", encoding="utf-8")
        (run / "config.json").write_text(
            f'{{"regime": "{regime}"}}', encoding="utf-8"
        )
    monkeypatch.setattr(lifecycle_analysis, "validate_run", lambda _: (True, "ok"))
    discovered = lifecycle_analysis.discover_runs(tmp_path, regime="trace_only")
    assert [path.name for path in discovered] == ["trace-run"]


def test_sequential_analysis_replaces_empty_heterogeneous_baseline_column() -> None:
    sequential = pd.DataFrame([{
        "run_id": "run",
        "shots": 1,
        "draw": 0,
        "base_accuracy_before": np.nan,
        "base_accuracy": 0.8,
    }])
    finalist = pd.DataFrame([{
        "run_id": "run",
        "shots": 1,
        "draw": 0,
        "base_accuracy_before": 0.9,
    }])
    attached = lifecycle_analysis.attach_sequential_baseline(
        sequential, finalist
    )
    assert attached["base_accuracy_before"].tolist() == [pytest.approx(0.9)]
    assert "base_accuracy_before_x" not in attached


def test_stress_heatmap_color_limit_preserves_both_tails() -> None:
    assert symmetric_color_limit(np.asarray([[-0.2, 0.8], [np.nan, 0.1]])) == pytest.approx(0.8)
    assert symmetric_color_limit(np.asarray([[np.nan]])) == pytest.approx(0.01)


def test_threshold_reconstruction_bounds_float32_tie_ambiguity() -> None:
    labels = np.asarray([0, 1, 4, 4, 6, 6])
    predicted = np.asarray([0, 1, 4, 1, 6, 1])
    scores = np.asarray(
        [0.1, 0.2, 0.5, 0.5, 0.5, 0.6], dtype=np.float32
    )
    intervals, tie_count = lifecycle_analysis._threshold_ambiguity_intervals(
        labels,
        predicted,
        scores,
        holdout=(4, 6),
        threshold=0.5,
    )
    assert tie_count == 3
    assert intervals["unknown_recall"] == pytest.approx((0.25, 1.0))
    assert intervals["worst_fault_recall"] == pytest.approx((0.0, 1.0))
    assert intervals["known_fault_acceptance"] == pytest.approx((1.0, 1.0))


def test_experiment_component_builder_returns_scod_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rng = np.random.default_rng(14)
    reference_embedding = rng.normal(size=(24, 4))
    reference_labels = np.repeat([0, 1, 2], 8)
    distance_reference = DistanceReference.fit(
        reference_embedding, reference_labels
    )
    prototype_bank = PrototypeBank.fit(
        reference_embedding,
        reference_labels,
        prototypes_per_class=1,
        metric="cosine",
        seed=14,
    )
    monkeypatch.setattr(
        lifecycle_experiment,
        "event_grammar_residual",
        lambda *args, **kwargs: np.zeros(5),
    )
    names, components = lifecycle_experiment._components(
        {
            "logits": torch.from_numpy(rng.normal(size=(5, 8))).float(),
            "embedding": torch.from_numpy(rng.normal(size=(5, 4))).float(),
        },
        batch=SimpleNamespace(
            trace=torch.zeros(5, 30),
            context=torch.zeros(5, 3),
        ),
        reference=distance_reference,
        prototypes=prototype_bank,
        config=lifecycle_experiment.SCODConfig(prototypes_per_class=1),
        recipe_means=torch.zeros(8, 1),
        recipe_stds=torch.ones(8, 1),
        known_class_ids=(0, 1, 2),
    )
    assert len(names) == components.shape[1]
    assert components.shape == (5, 11)
    assert np.isfinite(components).all()


def test_provenance_filters_generated_logs_before_filesystem_access(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_check_output(command, **kwargs):
        if command[1:3] == ["rev-parse", "HEAD"]:
            return b"abc123\n"
        if command[1:3] == ["diff", "--binary"]:
            return b""
        if command[1:4] == ["ls-files", "--others", "--exclude-standard"]:
            return (
                b"OTDR_CLI/OTDR/src/wandb/run-x/logs/debug-core.log\n"
            )
        raise AssertionError(command)

    original_is_file = Path.is_file

    def guarded_is_file(path: Path) -> bool:
        if "wandb" in path.as_posix():
            raise AssertionError("Generated W&B paths must be filtered first.")
        return original_is_file(path)

    monkeypatch.setattr(
        lifecycle_experiment.subprocess,
        "check_output",
        fake_check_output,
    )
    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    metadata = lifecycle_experiment._git_metadata(tmp_path)
    assert metadata["revision"] == "abc123"
    assert metadata["content_hashed_untracked_files"] == 0


def test_gate_diagnostics_are_descriptive_and_class_stratified() -> None:
    result = gate_diagnostics(
        np.asarray([[0.2, 0.4], [0.6, 0.8]]),
        np.asarray([0, 1]),
    )
    assert result is not None
    assert result["mean"] == pytest.approx(0.5)
    assert result["per_class_mean"] == {"0": pytest.approx(0.3), "1": pytest.approx(0.7)}
    assert gate_diagnostics(
        np.asarray([[np.nan, np.nan]]), np.asarray([0])
    ) is None


def test_tabpfn_context_and_query_subsamples_are_group_distinct() -> None:
    labels = np.asarray([0, 0, 0, 0, 1, 1, 1])
    groups = ("dup", "dup", "normal-a", "normal-b", "fault-a", "fault-a", "fault-b")
    selected = _ranked_indices(labels, groups, 0, 3, "test")
    assert len({groups[index] for index in selected}) == 3
    query = _balanced_query_indices(labels, groups, per_class=3)
    assert len(query) == 5
    assert len({groups[index] for index in query}) == 5
    with pytest.raises(ValueError, match="unique groups"):
        _ranked_indices(labels, groups, 1, 3, "test")


def test_confirmatory_tabpfn_protocol_hash_is_frozen() -> None:
    assert file_sha256(PROTOCOL_PATH) == FROZEN_PROTOCOL_SHA256


def test_confirmatory_probability_metrics_reconstruct_independently() -> None:
    labels = np.tile(np.arange(8), 3)
    probability = np.full((len(labels), 8), 0.01)
    probability[np.arange(len(labels)), labels] = 0.83
    probability[np.arange(len(labels)), (labels + 1) % 8] = 0.10
    probability /= probability.sum(1, keepdims=True)
    row = metric_row(
        labels=labels,
        probability=probability,
        base_class_ids=(0, 1, 2, 3, 4, 5),
        enrolled_class_ids=(6, 7),
        method="test",
        shots=1,
        draw=0,
        elapsed_seconds=0.0,
        probability_source="test",
        extra={
            "base_class_ids": [0, 1, 2, 3, 4, 5],
            "enrolled_class_ids": [6, 7],
        },
    )
    reconstructed = reconstruct_row(row)
    for name, value in reconstructed.items():
        assert value == pytest.approx(row[name], abs=1e-12)


def test_confirmatory_probability_alignment_and_distance_classes() -> None:
    aligned = _aligned_probability(
        np.asarray([[0.7, 0.3]]), classes=(2, 7)
    )
    assert aligned.shape == (1, 8)
    assert aligned.argmax(1).item() == 2
    probability = _distance_probability(
        np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]]),
        np.asarray([[0.01, 0.7]]),
        base_ids=(0, 1, 2, 3, 4, 5),
        enrolled_ids=(6, 7),
        temperature=0.1,
    )
    assert probability.argmax(1).item() == 6
    assert probability.sum() == pytest.approx(1.0)


def test_confirmatory_exact_fault_recall_distinguishes_enrolled_role(
    tmp_path: Path,
) -> None:
    unit = tmp_path / "pair_01_02" / "seed_42"
    unit.mkdir(parents=True)
    base_row = {
        "method": "tabpfn_v2",
        "shots": 5,
        "draw": 0,
        "base_class_ids": [0, 3, 4, 5, 6, 7],
        "enrolled_class_ids": [1, 2],
        "per_class_recall": {
            "0": 0.99,
            "1": 0.04,
            "2": 0.49,
            "3": 0.98,
            "4": 0.97,
            "5": 0.96,
            "6": 0.95,
            "7": 0.94,
        },
    }
    (unit / "metrics.json").write_text(
        __import__("json").dumps(
            {
                "run_id": "test-unit",
                "pair": [1, 2],
                "seed": 42,
                "rows": [base_row],
            }
        ),
        encoding="utf-8",
    )
    draw, pair_seed, summary = _per_fault_recall([unit])
    assert len(draw) == 7
    assert len(pair_seed) == 7
    enrolled = summary[summary["role"] == "enrolled"].set_index("fault")
    assert enrolled.loc[1, "near_zero_pair_seed_units"] == 1
    assert enrolled.loc[1, "weak_pair_seed_units"] == 1
    assert enrolled.loc[2, "near_zero_pair_seed_units"] == 0
    assert enrolled.loc[2, "weak_pair_seed_units"] == 1
    assert summary[summary["role"] == "base"]["fault"].tolist() == [3, 4, 5, 6, 7]


def test_confirmatory_prototype_efficiency_uses_saved_storage_and_latency(
    tmp_path: Path,
) -> None:
    unit = tmp_path / "pair_01_02" / "seed_42"
    unit.mkdir(parents=True)
    (unit / "metrics.json").write_text(
        __import__("json").dumps(
            {
                "run_id": "test-unit",
                "rows": [
                    {
                        "method": "cfe_finalist",
                        "shots": 5,
                        "draw": 0,
                        "storage_bytes": 2304,
                        "elapsed_seconds": 0.03,
                        "query_examples": 800,
                    },
                    {
                        "method": "tabpfn_v2",
                        "shots": 5,
                        "draw": 0,
                        "elapsed_seconds": 1.2,
                        "query_examples": 800,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    result = _prototype_efficiency([unit])
    assert result["method"].tolist() == ["cfe_finalist"]
    assert result.loc[0, "prototype_storage_bytes_mean"] == 2304
    assert result.loc[0, "enroll_and_predict_seconds_mean"] == pytest.approx(0.03)
