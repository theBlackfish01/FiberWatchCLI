from __future__ import annotations

"""Independent reconstruction, hierarchical analysis, plots, and report material."""

from collections import defaultdict
import json
from pathlib import Path
from typing import Any, Callable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from .study_state import atomic_json, validate_run
from .event_openworld_metrics import oscr_auc
from .lifecycle_scod import JointThreshold, evaluate_grouped_operating_point


RECONSTRUCTION_RATE_TOLERANCE = 5e-4
REQUIRED_ENRICHMENT_VERSION = 2


def _rejection_metrics(
    labels: np.ndarray,
    predicted: np.ndarray,
    rejected: np.ndarray,
    holdout: tuple[int, ...],
) -> dict[str, float]:
    normal = labels == 0
    unknown = np.isin(labels, holdout)
    known_fault = (~unknown) & (~normal)
    accepted_known = (~unknown) & (~rejected)
    return {
        "normal_far": float(rejected[normal].mean()),
        "known_fault_acceptance": float((~rejected[known_fault]).mean()),
        "unknown_recall": float(rejected[unknown].mean()),
        "worst_fault_recall": float(
            min(rejected[labels == class_id].mean() for class_id in holdout)
        ),
        "accepted_known_accuracy": float(
            (predicted[accepted_known] == labels[accepted_known]).mean()
        ),
    }


def _threshold_ambiguity_intervals(
    labels: np.ndarray,
    predicted: np.ndarray,
    score: np.ndarray,
    *,
    holdout: tuple[int, ...],
    threshold: float,
) -> tuple[dict[str, tuple[float, float]], int]:
    """Bound metrics when pre-serialization scores collapse onto one float32 tie."""
    persisted_threshold = np.asarray(threshold, dtype=score.dtype).item()
    tied = score == persisted_threshold
    reject_no_ties = score > persisted_threshold
    reject_all_ties = score >= persisted_threshold
    lower_metrics = _rejection_metrics(
        labels, predicted, reject_no_ties, holdout
    )
    upper_metrics = _rejection_metrics(
        labels, predicted, reject_all_ties, holdout
    )
    intervals = {
        key: (
            min(lower_metrics[key], upper_metrics[key]),
            max(lower_metrics[key], upper_metrics[key]),
        )
        for key in lower_metrics
        if key != "accepted_known_accuracy"
    }

    unknown = np.isin(labels, holdout)
    fixed_accepted = (~unknown) & (score < persisted_threshold)
    tied_known = (~unknown) & tied
    fixed_correct = int(
        (predicted[fixed_accepted] == labels[fixed_accepted]).sum()
    )
    fixed_count = int(fixed_accepted.sum())
    tied_correct = int(
        (predicted[tied_known] == labels[tied_known]).sum()
    )
    tied_incorrect = int(tied_known.sum()) - tied_correct
    minimum_accuracy = (
        fixed_correct / (fixed_count + tied_incorrect)
        if fixed_count + tied_incorrect
        else float("nan")
    )
    maximum_accuracy = (
        (fixed_correct + tied_correct) / (fixed_count + tied_correct)
        if fixed_count + tied_correct
        else float("nan")
    )
    intervals["accepted_known_accuracy"] = (
        min(minimum_accuracy, maximum_accuracy),
        max(minimum_accuracy, maximum_accuracy),
    )
    return intervals, int(tied.sum())


def discover_runs(
    study_root: Path,
    stage: str = "full_benchmark",
    *,
    regime: str | None = None,
) -> list[Path]:
    """Return validated runs, optionally restricted to one feature regime.

    The full, trace-only, and summary-only matrices intentionally share a stage
    directory so that the idempotent run hashes remain the single source of
    truth.  Analysis must nevertheless keep them statistically separate.
    """
    root = study_root / stage
    runs = []
    for path in root.glob("*/manifest.json"):
        run_dir = path.parent
        if not validate_run(run_dir)[0]:
            continue
        if regime is not None:
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            if config.get("regime") != regime:
                continue
        runs.append(run_dir)
    return sorted(runs)


def reconstruct_run(run_dir: Path) -> dict[str, Any]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    with np.load(run_dir / "predictions.npz") as payload:
        labels = payload["labels"].astype(int)
        predicted = payload["predicted"].astype(int)
        # Preserve the persisted dtype: NumPy's scalar comparison casts the saved
        # threshold consistently with the float32 score used for the run artifact.
        score = payload["kpsc_score"]
        holdout = tuple(config["holdout"])
        threshold = float(metrics["kpsc"]["threshold"])
        rejected = score > threshold
        unknown = np.isin(labels, holdout)
        reconstructed = _rejection_metrics(
            labels, predicted, rejected, holdout
        )
        differences = {
            key: abs(float(metrics["kpsc"][key]) - value)
            for key, value in reconstructed.items()
        }
        ambiguity_intervals, threshold_tie_count = (
            _threshold_ambiguity_intervals(
                labels,
                predicted,
                score,
                holdout=holdout,
                threshold=threshold,
            )
        )
        interval_violations = {
            key: max(
                ambiguity_intervals[key][0] - float(metrics["kpsc"][key]),
                0.0,
                float(metrics["kpsc"][key]) - ambiguity_intervals[key][1],
            )
            for key in reconstructed
        }
        cfe_checks = []
        for row in metrics["cfe"]:
            if row.get("method") != "finalist":
                continue
            name = f"cfe_prediction_shot{row['shots']}_draw{row['draw']}"
            prediction = payload[name].astype(int)
            base = ~unknown
            novel = unknown
            base_accuracy = float((prediction[base] == labels[base]).mean())
            novel_accuracy = float((prediction[novel] == labels[novel]).mean())
            harmonic = 0.0 if base_accuracy + novel_accuracy == 0 else (
                2 * base_accuracy * novel_accuracy / (base_accuracy + novel_accuracy)
            )
            cfe_checks.append({
                "shots": row["shots"], "draw": row["draw"],
                "maximum_absolute_difference": max(
                    abs(base_accuracy - float(row["base_accuracy"])),
                    abs(novel_accuracy - float(row["enrolled_accuracy"])),
                    abs(harmonic - float(row["harmonic_mean"])),
                ),
            })
    return {
        "run_id": metrics["run_id"],
        "holdout": list(holdout),
        "seed": config["seed"],
        "regime": config["regime"],
        "kpsc_reconstructed": reconstructed,
        "kpsc_maximum_absolute_difference": max(differences.values()),
        "kpsc_maximum_interval_violation": max(interval_violations.values()),
        "threshold_tie_count": threshold_tie_count,
        "threshold_ambiguity_intervals": {
            key: list(value) for key, value in ambiguity_intervals.items()
        },
        "cfe_maximum_absolute_difference": max(value["maximum_absolute_difference"] for value in cfe_checks),
        "score_storage_dtype": str(score.dtype),
        "rate_tolerance": RECONSTRUCTION_RATE_TOLERANCE,
        "precision_note": (
            "KPSC scores were evaluated before float32 artifact serialization; "
            "saved-metric validity is checked against the exact interval induced "
            "by observations tied at the persisted threshold."
        ),
        "passed": max(interval_violations.values()) <= 1e-12 and max(
            value["maximum_absolute_difference"] for value in cfe_checks
        ) < 1e-6,
    }


def _saved_score_diagnostics(run_dir: Path) -> dict[str, Any]:
    """Reconstruct ranking/selective diagnostics from immutable predictions."""
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    holdout = tuple(config["holdout"])
    with np.load(run_dir / "predictions.npz") as payload:
        labels = payload["labels"].astype(int)
        predicted = payload["predicted"].astype(int)
        score = payload["kpsc_score"].astype(float)
        group_ids = payload["group_ids"].astype(str)
    unknown = np.isin(labels, holdout)
    known = ~unknown
    correct_known = known & (predicted == labels)
    fpr, tpr, _ = roc_curve(unknown.astype(int), score)
    reached = np.flatnonzero(tpr >= 0.95)
    # OpenAUC is the pairwise probability that a correctly classified known
    # sample receives greater acceptance confidence than an unknown sample,
    # normalized by all known/unknown pairs.
    if correct_known.any():
        pair_labels = np.r_[
            np.ones(correct_known.sum(), dtype=int),
            np.zeros(unknown.sum(), dtype=int),
        ]
        pair_confidence = np.r_[-score[correct_known], -score[unknown]]
        openauc = float(
            roc_auc_score(pair_labels, pair_confidence)
            * correct_known.sum()
            / max(known.sum(), 1)
        )
    else:
        openauc = 0.0
    order = np.argsort(score)
    error = unknown | (known & (predicted != labels))
    cumulative_error = np.cumsum(error[order])
    accepted = np.arange(1, len(labels) + 1)
    risk = cumulative_error / accepted
    coverage = accepted / len(labels)
    grouped = evaluate_grouped_operating_point(
        score,
        labels,
        predicted,
        group_ids,
        holdout=holdout,
        calibration=JointThreshold(**metrics["kpsc"]["calibration"]),
        bootstrap_iterations=1000,
        seed=int(config["seed"]) + 100 * holdout[0] + holdout[1],
    )
    return {
        "openauc": openauc,
        "oscr_reconstructed": oscr_auc(labels, predicted, unknown, -score),
        "fpr_at_95_tpr": float(fpr[reached[0]]) if len(reached) else 1.0,
        "selective_aurc": float(np.trapezoid(risk, coverage)),
        "risk_at_80_coverage": float(risk[np.searchsorted(coverage, 0.8)]),
        **grouped,
    }


def _metric_rows(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    kpsc_rows, cfe_rows = [], []
    for run_dir in run_dirs:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        config = json.loads(
            (run_dir / "config.json").read_text(encoding="utf-8")
        )
        base = {
            "run_id": metrics["run_id"],
            "pair": "-".join(str(value) for value in metrics["holdout"]),
            "fault_a": metrics["holdout"][0],
            "fault_b": metrics["holdout"][1],
            "seed": metrics["seed"],
            "regime": metrics["regime"],
            "localization_training_weight": float(
                config["training"]["localization_weight"]
            ),
        }
        kpsc_rows.append({
            **base,
            **{key: metrics["kpsc"][key] for key in (
                "normal_far", "known_fault_acceptance", "unknown_recall",
                "worst_fault_recall", "accepted_known_accuracy",
                "overall_known_accuracy_rejection_failure",
                "normal_far_count", "normal_count", "constraints_met",
                "auroc", "aupr", "pauroc_0_01", "oscr",
            )},
            "known_balanced_accuracy": metrics["known_closed_set"]["balanced_accuracy"],
            "known_accuracy": metrics["known_closed_set"]["accuracy"],
            "known_macro_f1": metrics["known_closed_set"]["macro_f1"],
            "known_nll": metrics["known_closed_set"]["nll"],
            "known_brier": metrics["known_closed_set"]["brier"],
            "known_ece": metrics["known_closed_set"]["ece_15"],
            "known_localization_mae": metrics["known_closed_set"].get(
                "localization_mae"
            ),
            "known_localization_rmse": metrics["known_closed_set"].get(
                "localization_rmse"
            ),
            "known_per_class": metrics["known_closed_set"]["per_class"],
            "threshold": metrics["kpsc"]["threshold"],
            "calibration_mode": metrics["kpsc"]["calibration"]["mode"],
            "calibration_normal_groups": metrics["kpsc"]["calibration"][
                "normal_groups"
            ],
            "calibration_known_fault_groups": metrics["kpsc"][
                "calibration"
            ]["known_fault_groups"],
            "training_seconds": metrics["training"]["duration_seconds"],
            "total_seconds": metrics["duration_seconds"],
            "peak_cuda_memory_bytes": metrics["training"]["peak_reserved_bytes"],
            "parameter_count": metrics["training"]["parameter_count"],
            "checkpoint_size_bytes": (run_dir / "checkpoint.pt").stat().st_size,
            "inference_ms_per_trace": metrics.get("inference", {}).get(
                "milliseconds_per_trace_including_transfer"
            ),
            **_saved_score_diagnostics(run_dir),
        })
        for row in metrics["cfe"]:
            if row.get("method") == "finalist_sequential":
                cfe_rows.append({**base, **row})
            else:
                cfe_rows.append({**base, **row})
    return pd.DataFrame(kpsc_rows), pd.DataFrame(cfe_rows)


def hierarchical_bootstrap(
    frame: pd.DataFrame,
    value: str,
    *,
    iterations: int = 5000,
    seed: int = 20260725,
) -> dict[str, float]:
    """Pair is top cluster; seed is resampled within pair."""
    pair_values = sorted(frame["pair"].unique())
    nested = [
        part.groupby("seed")[value].mean().to_numpy(dtype=float)
        for _, part in frame.groupby("pair", sort=True)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(iterations)
    for iteration in range(iterations):
        chosen_pairs = rng.integers(0, len(nested), size=len(nested))
        total = 0.0
        count = 0
        for pair_index in chosen_pairs:
            values = nested[pair_index]
            selected = values[rng.integers(0, len(values), size=len(values))]
            total += float(selected.sum())
            count += len(selected)
        samples[iteration] = total / count
    return {
        "mean": float(frame[value].mean()),
        "ci_low": float(np.quantile(samples, 0.025)),
        "ci_high": float(np.quantile(samples, 0.975)),
        "bootstrap_iterations": iterations,
        "top_cluster": "heldout_pair",
        "nested_cluster": "seed",
    }


def paired_sign_flip(
    left: pd.DataFrame,
    right: pd.DataFrame,
    *,
    value: str,
    iterations: int = 20000,
    seed: int = 20260725,
) -> dict[str, float]:
    keys = ["pair", "seed"]
    merged = left.groupby(keys)[value].mean().rename("left").to_frame().join(
        right.groupby(keys)[value].mean().rename("right"), how="inner"
    )
    difference = (merged["left"] - merged["right"]).to_numpy()
    rng = np.random.default_rng(seed)
    observed = float(difference.mean())
    null = np.empty(iterations)
    for index in range(iterations):
        null[index] = np.mean(difference * rng.choice((-1, 1), len(difference)))
    return {
        "matched_pair_seed_units": len(difference),
        "mean_difference": observed,
        "median_difference": float(np.median(difference)),
        "two_sided_permutation_p": float((1 + np.sum(np.abs(null) >= abs(observed))) / (iterations + 1)),
    }


def fault_aware_leave_one_out(
    frame: pd.DataFrame,
    value: str,
) -> dict[str, Any]:
    """Sensitivity to removing every pair that contains one repeated fault."""
    if "fault_a" not in frame or "fault_b" not in frame:
        split = frame["pair"].str.split("-", expand=True).astype(int)
        fault_a, fault_b = split[0], split[1]
    else:
        fault_a, fault_b = frame["fault_a"], frame["fault_b"]
    estimates = {}
    for fault in range(1, 8):
        retained = frame[(fault_a != fault) & (fault_b != fault)]
        estimates[str(fault)] = float(retained[value].mean())
    values = list(estimates.values())
    return {
        "full_mean": float(frame[value].mean()),
        "leave_one_fault_out": estimates,
        "minimum": min(values),
        "maximum": max(values),
        "maximum_absolute_shift": max(
            abs(value_ - float(frame[value].mean())) for value_ in values
        ),
        "purpose": (
            "Sensitivity to repeated-class dependence; not a replacement for "
            "pair/seed hierarchical uncertainty."
        ),
    }


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def attach_sequential_baseline(
    sequential: pd.DataFrame,
    finalist: pd.DataFrame,
) -> pd.DataFrame:
    """Attach pre-enrollment base accuracy to heterogeneous sequential rows."""
    before = finalist[[
        "run_id", "shots", "draw", "base_accuracy_before"
    ]].drop_duplicates()
    clean = sequential.drop(
        columns=["base_accuracy_before"], errors="ignore"
    )
    return clean.merge(
        before,
        on=["run_id", "shots", "draw"],
        how="left",
        validate="many_to_one",
    )


def _frozen_baseline_comparisons(
    finalist: pd.DataFrame,
    kpsc: pd.DataFrame,
    study_root: Path,
    *,
    regime: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    path = (
        study_root.parent
        / "otdr_three_approach_study"
        / "tables"
        / "per_run_results.csv"
    )
    if not path.exists():
        return [], {"available": False, "path": str(path)}
    frozen = pd.read_csv(path)
    one_shot = (
        finalist[finalist["shots"] == 1]
        .groupby(["pair", "seed"], as_index=False)["harmonic_mean"]
        .mean()
    )
    comparisons = []
    labels = {
        "a": "frozen_episodic_metric",
        "b": "frozen_physics_semantic",
        "c": "frozen_ssl_conformal",
    }
    for approach in ("a", "b", "c"):
        prior = frozen[frozen["approach"] == approach].rename(
            columns={"fold": "pair", "post_h": "harmonic_mean"}
        )
        comparison = paired_sign_flip(
            one_shot,
            prior[["pair", "seed", "harmonic_mean"]],
            value="harmonic_mean",
        )
        comparisons.append({
            "baseline": labels[approach],
            "new_regime": regime,
            "frozen_regime": "trace_only",
            "protocol_note": (
                "Matched on shared held-out pair/seed units. The frozen study has "
                "three seeds; full-regime comparisons additionally change features."
            ),
            **comparison,
        })
    adjusted = holm_adjust([
        row["two_sided_permutation_p"] for row in comparisons
    ])
    for row, value in zip(comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = value
    semantic = frozen[frozen["approach"] == "b"]
    frozen_episodic = frozen[frozen["approach"] == "a"].rename(
        columns={
            "fold": "pair",
            "pre_unknown_recall": "unknown_recall",
        }
    )
    kpsc_unknown_comparison = paired_sign_flip(
        kpsc,
        frozen_episodic[["pair", "seed", "unknown_recall"]],
        value="unknown_recall",
    )
    context = {
        "available": True,
        "path": str(path),
        "frozen_runs_per_approach": int(
            frozen.groupby("approach").size().min()
        ),
        "frozen_semantic_strict_balanced_mean": float(
            semantic["strict_balanced"].mean()
        ),
        "frozen_semantic_gzsl_h_mean": float(semantic["gzsl_h"].mean()),
        "immutability": "read-only comparator; no frozen artifact was modified",
        "kpsc_vs_frozen_episodic_unknown_recall": {
            "new_regime": regime,
            "frozen_regime": "trace_only",
            "protocol_note": (
                "Matched shared pair/seed units; a full-regime comparison "
                "also changes the available feature contract."
            ),
            **kpsc_unknown_comparison,
        },
    }
    return comparisons, context


def _posthoc_operating_rows(
    study_root: Path,
    *,
    regime: str,
    source_run_ids: set[str],
) -> pd.DataFrame:
    rows = []
    root = study_root / "posthoc_calibration" / regime
    for path in root.glob("*/metrics.json") if root.exists() else ():
        if not validate_run(
            path.parent,
            expected={"enrichment_version": REQUIRED_ENRICHMENT_VERSION},
        )[0]:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_run_id") not in source_run_ids:
            continue
        pair = "-".join(str(value) for value in payload["holdout"])
        for far, values in payload["operating_points"].items():
            rows.append({
                "source_run_id": payload["source_run_id"],
                "pair": pair,
                "seed": payload["seed"],
                "far": far,
                "weighting": "row",
                "normal_far": values["normal_far"],
                "known_fault_acceptance": values["known_fault_acceptance"],
                "unknown_recall": values["unknown_recall"],
                "worst_fault_recall": values["worst_fault_recall"],
                "constraints_met": values["constraints_met"],
            })
        for far, values in payload["group_equal_weight_operating_points"].items():
            rows.append({
                "source_run_id": payload["source_run_id"],
                "pair": pair,
                "seed": payload["seed"],
                "far": far,
                "weighting": "group_equal",
                "normal_far": values["group_weighted_normal_far"],
                "known_fault_acceptance": values[
                    "group_weighted_known_fault_acceptance"
                ],
                "unknown_recall": values["group_weighted_unknown_recall"],
                "worst_fault_recall": values[
                    "group_weighted_worst_fault_recall"
                ],
                "constraints_met": values["group_constraints_met"],
            })
    return pd.DataFrame(rows)


def _posthoc_cfe_rows(
    study_root: Path,
    *,
    regime: str,
    source_run_ids: set[str],
) -> pd.DataFrame:
    rows = []
    root = study_root / "posthoc_calibration" / regime
    for path in root.glob("*/metrics.json") if root.exists() else ():
        if not validate_run(
            path.parent,
            expected={"enrichment_version": REQUIRED_ENRICHMENT_VERSION},
        )[0]:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("source_run_id") not in source_run_ids:
            continue
        calibration = payload.get("cfe_probability_calibration")
        if not calibration:
            continue
        pair = "-".join(str(value) for value in payload["holdout"])
        for row in calibration["rows"]:
            rows.append({
                "source_run_id": payload["source_run_id"],
                "pair": pair,
                "seed": payload["seed"],
                "distance_temperature": calibration["distance_temperature"],
                **row,
            })
    return pd.DataFrame(rows)


def _save_plots(
    kpsc: pd.DataFrame,
    cfe: pd.DataFrame,
    study_root: Path,
    *,
    regime: str,
    run_dirs: list[Path],
) -> list[str]:
    plot_root = study_root / "plots" / regime
    plot_root.mkdir(parents=True, exist_ok=True)
    created = []
    plt.figure(figsize=(7, 5))
    feasible = kpsc["constraints_met"].astype(bool)
    plt.scatter(
        kpsc.loc[~feasible, "known_fault_acceptance"],
        kpsc.loc[~feasible, "unknown_recall"], alpha=0.55, label="infeasible"
    )
    plt.scatter(
        kpsc.loc[feasible, "known_fault_acceptance"],
        kpsc.loc[feasible, "unknown_recall"], alpha=0.75, label="jointly feasible"
    )
    plt.axvline(0.95, color="black", linestyle="--", linewidth=1)
    plt.xlabel("Known-fault acceptance")
    plt.ylabel("Held-out-fault recall")
    plt.title("KPSC preservation–novelty trade-off")
    plt.legend()
    plt.tight_layout()
    path = plot_root / "kpsc_preservation_tradeoff.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    fpr_grid = np.linspace(0, 1, 501)
    interpolated = []
    known_scores: list[np.ndarray] = []
    unknown_scores: list[np.ndarray] = []
    fault_rows: list[dict[str, Any]] = []
    risk_curves = []
    for run_dir in run_dirs:
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        holdout = tuple(config["holdout"])
        with np.load(run_dir / "predictions.npz") as payload:
            labels = payload["labels"].astype(int)
            predicted = payload["predicted"].astype(int)
            score = payload["kpsc_score"].astype(float)
        unknown = np.isin(labels, holdout)
        fpr, tpr, _ = roc_curve(unknown.astype(int), score)
        interpolated.append(np.interp(fpr_grid, fpr, tpr))
        rng = np.random.default_rng(int(config["seed"]) + 100 * holdout[0] + holdout[1])
        for mask, target in ((~unknown, known_scores), (unknown, unknown_scores)):
            values = score[mask]
            target.append(values[rng.choice(len(values), min(2000, len(values)), replace=False)])
        error = unknown | ((~unknown) & (predicted != labels))
        order = np.argsort(score)
        coverage = np.arange(1, len(score) + 1) / len(score)
        risk = np.cumsum(error[order]) / np.arange(1, len(score) + 1)
        risk_curves.append(np.interp(fpr_grid, coverage, risk))
        pair = "-".join(str(value) for value in holdout)
        for fault, recall in metrics["kpsc"]["per_fault_recall"].items():
            fault_rows.append({"pair": pair, "fault": int(fault), "recall": float(recall)})

    mean_tpr = np.mean(interpolated, axis=0)
    low_tpr, high_tpr = np.quantile(interpolated, (0.1, 0.9), axis=0)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_grid, mean_tpr, label="pair/seed mean")
    plt.fill_between(fpr_grid, low_tpr, high_tpr, alpha=0.2, label="10th-90th percentile")
    plt.plot([0, 1], [0, 1], color="grey", linestyle=":")
    plt.xlim(0, 0.2)
    plt.ylim(0, 1)
    plt.xlabel("Known-sample false-positive rate")
    plt.ylabel("Held-out-fault true-positive rate")
    plt.title(f"KPSC ROC ({regime.replace('_', ' ')})")
    plt.legend()
    plt.tight_layout()
    path = plot_root / "kpsc_roc.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    plt.figure(figsize=(7, 5))
    plt.hist(np.concatenate(known_scores), bins=60, density=True, alpha=0.55, label="known")
    plt.hist(np.concatenate(unknown_scores), bins=60, density=True, alpha=0.55, label="held-out faults")
    plt.xlabel("KPSC rejection score")
    plt.ylabel("Density")
    plt.title(f"KPSC score distributions ({regime.replace('_', ' ')})")
    plt.legend()
    plt.tight_layout()
    path = plot_root / "kpsc_score_histogram.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    fault_frame = pd.DataFrame(fault_rows)
    heatmap = fault_frame.groupby(["fault", "pair"])["recall"].mean().unstack()
    plt.figure(figsize=(11, 4.5))
    image = plt.imshow(heatmap.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="viridis")
    plt.yticks(range(len(heatmap.index)), heatmap.index)
    plt.xticks(range(len(heatmap.columns)), heatmap.columns, rotation=70, ha="right", fontsize=7)
    plt.xlabel("Held-out pair")
    plt.ylabel("Fault class")
    plt.title(f"Per-fault rejection recall ({regime.replace('_', ' ')})")
    plt.colorbar(image, label="Recall")
    plt.tight_layout()
    path = plot_root / "kpsc_per_fault_heatmap.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    mean_risk = np.mean(risk_curves, axis=0)
    low_risk, high_risk = np.quantile(risk_curves, (0.1, 0.9), axis=0)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr_grid, mean_risk)
    plt.fill_between(fpr_grid, low_risk, high_risk, alpha=0.2)
    plt.xlabel("Accepted coverage")
    plt.ylabel("Selective error risk")
    plt.title(f"KPSC risk-coverage ({regime.replace('_', ' ')})")
    plt.tight_layout()
    path = plot_root / "kpsc_risk_coverage.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    finalist = cfe[cfe["method"] == "finalist"].copy()
    shot = finalist.groupby("shots")["harmonic_mean"].agg(["mean", "std"])
    plt.figure(figsize=(7, 5))
    plt.errorbar(shot.index, shot["mean"], yerr=shot["std"], marker="o", capsize=4)
    plt.xticks(sorted(shot.index))
    plt.ylim(0, 1)
    plt.xlabel("Shots per enrolled class")
    plt.ylabel("Seen/enrolled harmonic mean")
    plt.title("CFE enrollment curve across pair/seed/support draws")
    plt.tight_layout()
    path = plot_root / "cfe_shot_curve.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)

    methods = cfe[
        cfe["method"].isin((
            "finalist", "uncalibrated_mean", "raw_cosine_1nn",
            "raw_euclidean_1nn", "encoder_cosine_1nn",
        ))
    ]
    table = methods.groupby(["method", "shots"])["harmonic_mean"].mean().unstack(0)
    table.plot(marker="o", figsize=(8, 5))
    plt.ylim(0, 1)
    plt.xlabel("Shots")
    plt.ylabel("Harmonic mean")
    plt.title("CFE finalist and required baselines")
    plt.tight_layout()
    path = plot_root / "cfe_baseline_comparison.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(path.name)
    return created


def analyze_lifecycle(
    study_root: Path,
    *,
    regime: str = "full",
    require_complete: bool = True,
    expected_runs: int | None = None,
    bootstrap_iterations: int = 5000,
) -> dict[str, Any]:
    if bootstrap_iterations < 1:
        raise ValueError("bootstrap_iterations must be positive.")
    def bootstrap(frame: pd.DataFrame, value: str) -> dict[str, float]:
        return hierarchical_bootstrap(
            frame,
            value,
            iterations=bootstrap_iterations,
        )

    runs = discover_runs(study_root, regime=regime)
    expected = 21 * 5 if expected_runs is None else expected_runs
    if require_complete and len(runs) != expected:
        raise RuntimeError(
            f"{regime!r} benchmark has {len(runs)}/{expected} validated runs."
        )
    reconstruction = [reconstruct_run(path) for path in runs]
    if any(not row["passed"] for row in reconstruction):
        raise AssertionError("At least one saved headline metric failed reconstruction.")
    kpsc, cfe = _metric_rows(runs)
    table_root = study_root / "tables" / regime
    table_root.mkdir(parents=True, exist_ok=True)
    kpsc.to_csv(table_root / "kpsc_per_run.csv", index=False)
    cfe.to_csv(table_root / "cfe_per_draw.csv", index=False)
    known_class_rows = []
    for _, row in kpsc.iterrows():
        for class_id, class_metrics in row["known_per_class"].items():
            if int(class_metrics["support"]) == 0:
                continue
            known_class_rows.append({
                "pair": row["pair"],
                "seed": int(row["seed"]),
                "class_id": int(class_id),
                **class_metrics,
            })
    known_per_class = pd.DataFrame(known_class_rows)
    known_per_class.to_csv(
        table_root / "known_closed_set_per_class.csv", index=False
    )
    kpsc_fault_rows = []
    for run_dir in runs:
        metrics = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        pair = "-".join(str(value) for value in metrics["holdout"])
        for class_id, recall in metrics["kpsc"]["per_fault_recall"].items():
            kpsc_fault_rows.append({
                "pair": pair,
                "seed": int(metrics["seed"]),
                "class_id": int(class_id),
                "unknown_recall": float(recall),
            })
    kpsc_per_fault = pd.DataFrame(kpsc_fault_rows)
    kpsc_per_fault.to_csv(
        table_root / "kpsc_per_fault_recall.csv", index=False
    )
    gate_rows, gate_channel_rows, gate_class_rows = [], [], []
    for run_dir in runs:
        metrics = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8")
        )
        diagnostic = metrics.get("fusion_gate")
        if diagnostic is None:
            enriched_path = (
                study_root
                / "posthoc_calibration"
                / regime
                / run_dir.name
                / "metrics.json"
            )
            if enriched_path.exists():
                if validate_run(
                    enriched_path.parent,
                    expected={
                        "enrichment_version": REQUIRED_ENRICHMENT_VERSION
                    },
                )[0]:
                    diagnostic = json.loads(
                        enriched_path.read_text(encoding="utf-8")
                    ).get("fusion_gate")
        if not diagnostic:
            continue
        base = {
            "run_id": run_dir.name,
            "pair": "-".join(str(value) for value in metrics["holdout"]),
            "seed": int(metrics["seed"]),
        }
        gate_rows.append({
            **base,
            **{
                key: diagnostic[key]
                for key in ("mean", "std", "minimum", "maximum")
            },
        })
        gate_channel_rows.extend({
            **base,
            "channel": channel,
            "mean_gate": float(value),
        } for channel, value in enumerate(diagnostic["per_channel_mean"]))
        gate_class_rows.extend({
            **base,
            "class_id": int(class_id),
            "mean_gate": float(value),
        } for class_id, value in diagnostic["per_class_mean"].items())
    gate_frame = pd.DataFrame(gate_rows)
    gate_channels = pd.DataFrame(gate_channel_rows)
    gate_classes = pd.DataFrame(gate_class_rows)
    if len(gate_frame):
        gate_frame.to_csv(
            table_root / "fusion_gate_per_run.csv", index=False
        )
        gate_channels.to_csv(
            table_root / "fusion_gate_per_channel.csv", index=False
        )
        gate_classes.to_csv(
            table_root / "fusion_gate_per_class.csv", index=False
        )
    atomic_json(table_root / "metric_reconstruction.json", {
        "runs": len(reconstruction),
        "all_passed": True,
        "maximum_kpsc_rate_difference": max(
            row["kpsc_maximum_absolute_difference"] for row in reconstruction
        ),
        "maximum_cfe_difference": max(
            row["cfe_maximum_absolute_difference"] for row in reconstruction
        ),
        "kpsc_rate_tolerance": RECONSTRUCTION_RATE_TOLERANCE,
        "details": reconstruction,
    })
    posthoc = _posthoc_operating_rows(
        study_root,
        regime=regime,
        source_run_ids={path.name for path in runs},
    )
    posthoc_summary: dict[str, Any] = {
        "enriched_source_runs": int(posthoc["source_run_id"].nunique())
        if len(posthoc) else 0,
        "required_source_runs": len(runs),
        "complete": bool(
            len(posthoc)
            and posthoc["source_run_id"].nunique() == len(runs)
        ),
        "rows": len(posthoc),
        "summaries": {},
    }
    if len(posthoc):
        posthoc.to_csv(table_root / "kpsc_operating_points.csv", index=False)
        for (weighting, far), part in posthoc.groupby(["weighting", "far"]):
            posthoc_summary["summaries"][f"{weighting}_{far}"] = {
                value: bootstrap(part, value)
                for value in (
                    "normal_far",
                    "known_fault_acceptance",
                    "unknown_recall",
                    "worst_fault_recall",
                )
            } | {
                "joint_feasibility_rate": float(part["constraints_met"].mean()),
                "runs": len(part),
            }
    posthoc_cfe = _posthoc_cfe_rows(
        study_root,
        regime=regime,
        source_run_ids={path.name for path in runs},
    )
    cfe_probability_summary: dict[str, Any] = {
        "runs": int(posthoc_cfe["source_run_id"].nunique())
        if len(posthoc_cfe) else 0,
        "complete": bool(
            len(posthoc_cfe)
            and posthoc_cfe["source_run_id"].nunique() == len(runs)
        ),
        "shots": {},
    }
    if len(posthoc_cfe):
        posthoc_cfe.to_csv(
            table_root / "cfe_probability_calibration.csv", index=False
        )
        cfe_probability_units = posthoc_cfe.groupby(
            ["pair", "seed", "shots"], as_index=False
        ).agg({
            "nll": "mean",
            "brier": "mean",
            "ece_15": "mean",
            "accuracy": "mean",
            "normal_far_after_enrollment": "mean",
            "enrollment_latency_ms": "mean",
            "prototype_storage_bytes": "mean",
        })
        for shots, part in cfe_probability_units.groupby("shots"):
            cfe_probability_summary["shots"][str(int(shots))] = {
                value: bootstrap(part, value)
                for value in (
                    "nll",
                    "brier",
                    "ece_15",
                    "accuracy",
                    "normal_far_after_enrollment",
                    "enrollment_latency_ms",
                    "prototype_storage_bytes",
                )
            }

    kpsc_summary = {
        value: bootstrap(kpsc, value)
        for value in (
            "normal_far", "known_fault_acceptance", "unknown_recall",
            "worst_fault_recall", "accepted_known_accuracy", "auroc", "aupr",
            "overall_known_accuracy_rejection_failure",
            "pauroc_0_01", "oscr", "openauc", "fpr_at_95_tpr",
            "selective_aurc", "risk_at_80_coverage",
            "known_accuracy", "known_balanced_accuracy", "known_macro_f1",
            "known_nll", "known_brier", "known_ece",
            "known_localization_mae", "known_localization_rmse",
            "group_weighted_normal_far",
            "group_weighted_known_fault_acceptance",
            "group_weighted_unknown_recall",
            "group_weighted_worst_fault_recall",
        )
    }
    kpsc_summary["joint_feasibility_rate"] = float(kpsc["constraints_met"].mean())
    kpsc_summary["group_weighted_feasibility_rate_at_frozen_threshold"] = float(
        kpsc["group_constraints_met"].mean()
    )
    kpsc_summary["feasible_runs"] = int(kpsc["constraints_met"].sum())
    kpsc_summary["runs"] = len(kpsc)
    kpsc_summary["calibration_sample_size"] = {
        "normal_groups_minimum": int(
            kpsc["calibration_normal_groups"].min()
        ),
        "normal_groups_mean": float(
            kpsc["calibration_normal_groups"].mean()
        ),
        "known_fault_groups_minimum": int(
            kpsc["calibration_known_fault_groups"].min()
        ),
        "known_fault_groups_mean": float(
            kpsc["calibration_known_fault_groups"].mean()
        ),
        "mode": sorted(kpsc["calibration_mode"].unique()),
        "interpretation": (
            "Exact input groups are deduplicated before splitting; these "
            "counts are group-distinct calibration observations."
        ),
    }
    kpsc_summary["localization_head"] = {
        "trained": bool(
            (kpsc["localization_training_weight"] > 0).any()
        ),
        "training_weight": float(
            kpsc["localization_training_weight"].max()
        ),
        "headline_eligible": bool(
            (kpsc["localization_training_weight"] > 0).all()
        ),
        "interpretation": (
            "When weight is zero, MAE/RMSE are an untrained-head control and "
            "must not be claimed as learned localization performance."
        ),
    }
    kpsc_summary["known_per_class_recall"] = {
        str(int(class_id)): bootstrap(part, "recall")
        for class_id, part in known_per_class.groupby("class_id")
    }
    kpsc_summary["heldout_per_fault_recall"] = {
        str(int(class_id)): bootstrap(part, "unknown_recall")
        for class_id, part in kpsc_per_fault.groupby("class_id")
    }
    kpsc_summary["fusion_gate_diagnostic"] = {
        "available_runs": len(gate_frame),
        "complete": len(gate_frame) == len(runs),
        "overall_mean": (
            bootstrap(gate_frame, "mean") if len(gate_frame) else None
        ),
        "per_class_mean": {
            str(int(class_id)): bootstrap(part, "mean_gate")
            for class_id, part in gate_classes.groupby("class_id")
        } if len(gate_classes) else {},
        "interpretation": (
            "Descriptive gate activations only; they are neither normalized "
            "branch shares nor causal feature importance."
        ),
    }
    kpsc_summary["fault_aware_sensitivity"] = {
        value: fault_aware_leave_one_out(kpsc, value)
        for value in (
            "unknown_recall",
            "worst_fault_recall",
            "normal_far",
            "known_fault_acceptance",
        )
    }
    finalist = cfe[cfe["method"] == "finalist"].copy()
    cfe_class_rows = []
    for _, row in finalist.iterrows():
        holdout = {int(row["fault_a"]), int(row["fault_b"])}
        for class_id, recall in row["per_class_recall"].items():
            cfe_class_rows.append({
                "pair": row["pair"],
                "seed": int(row["seed"]),
                "shots": int(row["shots"]),
                "draw": int(row["draw"]),
                "class_id": int(class_id),
                "role": (
                    "enrolled" if int(class_id) in holdout else "base"
                ),
                "recall": float(recall),
            })
    cfe_per_class = pd.DataFrame(cfe_class_rows)
    cfe_per_class.to_csv(
        table_root / "cfe_finalist_per_class_draw.csv", index=False
    )
    cfe_per_class_units = cfe_per_class.groupby(
        ["pair", "seed", "shots", "class_id", "role"],
        as_index=False,
    )["recall"].mean()
    # Average support draws before hierarchical resampling.
    cfe_units = finalist.groupby(["pair", "seed", "shots"], as_index=False).agg({
        "accuracy": "mean", "macro_f1": "mean",
        "harmonic_mean": "mean", "base_accuracy": "mean",
        "enrolled_accuracy": "mean", "worst_enrolled_recall": "mean",
        "normal_far_after_enrollment": "mean",
        "forgetting": "mean", "backward_transfer": "mean",
        "retention_ratio": "mean",
    })
    cfe_summary = {}
    for shots, part in cfe_units.groupby("shots"):
        cfe_summary[str(int(shots))] = {
            value: bootstrap(part, value)
            for value in (
                "accuracy", "macro_f1", "harmonic_mean",
                "base_accuracy", "enrolled_accuracy",
                "worst_enrolled_recall", "normal_far_after_enrollment",
                "forgetting", "backward_transfer", "retention_ratio",
            )
        } | {
            "fault_aware_harmonic_sensitivity": fault_aware_leave_one_out(
                part, "harmonic_mean"
            ),
            "per_class_recall": {
                f"{role}_{int(class_id)}": bootstrap(class_part, "recall")
                for (class_id, role), class_part in cfe_per_class_units[
                    cfe_per_class_units["shots"] == shots
                ].groupby(["class_id", "role"])
            },
        }
    sequential = cfe[cfe["method"] == "finalist_sequential"].copy()
    sequential_summary: dict[str, Any] = {}
    if len(sequential):
        sequential["order_label"] = sequential["order"].map(
            lambda value: "-".join(str(item) for item in value)
        )
        sequential = attach_sequential_baseline(sequential, finalist)
        sequential["forgetting"] = np.maximum(
            0,
            sequential["base_accuracy_before"] - sequential["base_accuracy"],
        )
        sequential["backward_transfer"] = (
            sequential["base_accuracy"] - sequential["base_accuracy_before"]
        )
        sequential["retention_ratio"] = (
            sequential["base_accuracy"] / sequential["base_accuracy_before"]
        )
        sequential_units = sequential.groupby(
            ["pair", "seed", "shots", "session", "order_label"],
            as_index=False,
        ).agg({
            "base_accuracy": "mean",
            "enrolled_accuracy": "mean",
            "harmonic_mean": "mean",
            "normal_far_after_enrollment": "mean",
            "forgetting": "mean",
            "backward_transfer": "mean",
            "retention_ratio": "mean",
        })
        sequential_units.to_csv(
            table_root / "cfe_sequential_units.csv", index=False
        )
        for (shots, session), part in sequential_units.groupby(
            ["shots", "session"]
        ):
            sequential_summary[f"shot_{int(shots)}_session_{int(session)}"] = {
                value: bootstrap(part, value)
                for value in (
                    "base_accuracy",
                    "enrolled_accuracy",
                    "harmonic_mean",
                    "normal_far_after_enrollment",
                    "forgetting",
                    "backward_transfer",
                    "retention_ratio",
                )
            } | {
                "both_class_orders_included": sorted(
                    part["order_label"].unique()
                )
            }
        incremental_units = sequential_units.groupby(
            ["pair", "seed", "shots"], as_index=False
        ).agg({
            "enrolled_accuracy": "mean",
            "harmonic_mean": "mean",
            "base_accuracy": "mean",
        })
        sequential_summary["average_incremental_accuracy"] = {
            str(int(shots)): {
                "enrolled_accuracy_across_sessions": bootstrap(
                    part, "enrolled_accuracy"
                ),
                "harmonic_mean_across_sessions": bootstrap(
                    part, "harmonic_mean"
                ),
                "base_accuracy_across_sessions": bootstrap(
                    part, "base_accuracy"
                ),
            }
            for shots, part in incremental_units.groupby("shots")
        }
    baseline_comparisons = []
    for method in (
        "uncalibrated_mean", "raw_cosine_1nn", "raw_euclidean_1nn",
        "raw_mahalanobis_1nn", "encoder_cosine_1nn",
    ):
        baseline = cfe[cfe["method"] == method]
        for shots in (1, 3, 5):
            left = finalist[finalist["shots"] == shots]
            right = baseline[baseline["shots"] == shots]
            comparison = paired_sign_flip(left, right, value="harmonic_mean")
            baseline_comparisons.append({"method": method, "shots": shots, **comparison})
    adjusted = holm_adjust([row["two_sided_permutation_p"] for row in baseline_comparisons])
    for row, value in zip(baseline_comparisons, adjusted, strict=True):
        row["holm_adjusted_p"] = value
    frozen_comparisons, frozen_context = _frozen_baseline_comparisons(
        finalist, kpsc, study_root, regime=regime
    )
    posthoc_latency_rows = []
    for run_dir in runs:
        path = (
            study_root
            / "posthoc_calibration"
            / regime
            / run_dir.name
            / "metrics.json"
        )
        if not path.exists():
            continue
        if not validate_run(
            path.parent,
            expected={"enrichment_version": REQUIRED_ENRICHMENT_VERSION},
        )[0]:
            continue
        benchmark = json.loads(
            path.read_text(encoding="utf-8")
        ).get("inference_benchmark")
        if benchmark and benchmark.get("outer"):
            posthoc_latency_rows.append({
                "run_id": run_dir.name,
                **benchmark["outer"],
            })
    posthoc_latency = pd.DataFrame(posthoc_latency_rows)
    if len(posthoc_latency):
        posthoc_latency.to_csv(
            table_root / "posthoc_outer_inference_latency.csv",
            index=False,
        )
    recorded_latency = (
        float(kpsc["inference_ms_per_trace"].dropna().mean())
        if kpsc["inference_ms_per_trace"].notna().any()
        else None
    )
    reconstructed_latency = (
        float(
            posthoc_latency[
                "milliseconds_per_trace_including_transfer"
            ].mean()
        )
        if len(posthoc_latency)
        else None
    )
    efficiency = {
        "parameter_count": float(kpsc["parameter_count"].mean()),
        "cuda_training_seconds_per_fold": float(kpsc["training_seconds"].mean()),
        "total_seconds_per_fold": float(kpsc["total_seconds"].mean()),
        "peak_cuda_memory_bytes": float(kpsc["peak_cuda_memory_bytes"].max()),
        "checkpoint_size_bytes": float(kpsc["checkpoint_size_bytes"].mean()),
        "inference_ms_per_trace_including_transfer": (
            recorded_latency
            if recorded_latency is not None
            else reconstructed_latency
        ),
        "inference_latency_source": (
            "source_run"
            if recorded_latency is not None
            else (
                "cuda_posthoc_checkpoint_reconstruction"
                if reconstructed_latency is not None
                else None
            )
        ),
        "finalist_prototype_storage_bytes": float(
            finalist["storage_bytes"].mean()
        ),
        "finalist_enrollment_latency_ms": (
            float(finalist["enrollment_latency_ms"].dropna().mean())
            if (
                "enrollment_latency_ms" in finalist
                and finalist["enrollment_latency_ms"].notna().any()
            )
            else (
                float(posthoc_cfe["enrollment_latency_ms"].mean())
                if len(posthoc_cfe)
                else None
            )
        ),
    }
    cfe_one = cfe_summary.get("1", {})
    cfe_five = cfe_summary.get("5", {})
    one_enrolled_class_recall = [
        values["mean"]
        for key, values in cfe_one.get("per_class_recall", {}).items()
        if key.startswith("enrolled_")
    ]
    success_criteria = {
        "kpsc": {
            "mean_normal_far_le_0_0125": bool(
                kpsc_summary["normal_far"]["mean"] <= 0.0125
            ),
            "mean_known_fault_acceptance_ge_0_95": bool(
                kpsc_summary["known_fault_acceptance"]["mean"] >= 0.95
            ),
            "mean_unknown_recall_ge_0_25": bool(
                kpsc_summary["unknown_recall"]["mean"] >= 0.25
            ),
            "aggregate_numeric_targets_jointly_met": bool(
                kpsc_summary["normal_far"]["mean"] <= 0.0125
                and kpsc_summary["known_fault_acceptance"]["mean"] >= 0.95
                and kpsc_summary["unknown_recall"]["mean"] >= 0.25
            ),
            "per_fold_joint_feasibility_rate": kpsc_summary[
                "joint_feasibility_rate"
            ],
            "matched_frozen_episodic_unknown_recall": frozen_context.get(
                "kpsc_vs_frozen_episodic_unknown_recall"
            ),
            "interpretation": (
                "Aggregate targets and fold-level feasibility are reported "
                "together; an aggregate pass does not erase heterogeneous "
                "pair/seed failures."
            ),
        },
        "cfe": {
            "one_shot_h_ge_0_55": bool(
                cfe_one
                and cfe_one["harmonic_mean"]["mean"] >= 0.55
            ),
            "five_shot_h_ge_0_65": bool(
                cfe_five
                and cfe_five["harmonic_mean"]["mean"] >= 0.65
            ),
            "one_shot_retention_ratio_ge_0_95": bool(
                cfe_one
                and cfe_one["retention_ratio"]["mean"] >= 0.95
            ),
            "one_shot_forgetting_le_0_03": bool(
                cfe_one and cfe_one["forgetting"]["mean"] <= 0.03
            ),
            "minimum_one_shot_enrolled_class_mean_recall": (
                min(one_enrolled_class_recall)
                if one_enrolled_class_recall
                else None
            ),
            "interpretation": (
                "The vague predeclared near-zero-recall phrase is not assigned "
                "a post-hoc cutoff; the exact minimum class mean is reported."
            ),
        },
    }
    plots = _save_plots(
        kpsc, cfe, study_root, regime=regime, run_dirs=runs
    ) if len(runs) else []
    summary = {
        "schema_version": 1,
        "regime": regime,
        "validated_runs": len(runs),
        "expected_runs": expected,
        "complete": len(runs) == expected,
        "kpsc": kpsc_summary,
        "cfe": cfe_summary,
        "cfe_sequential": sequential_summary,
        "cfe_probability_calibration": cfe_probability_summary,
        "cfe_paired_comparisons": baseline_comparisons,
        "frozen_enrollment_comparisons": frozen_comparisons,
        "frozen_semantic_context": frozen_context,
        "efficiency": efficiency,
        "predeclared_success_criteria": success_criteria,
        "posthoc_multi_far_and_group_calibration": posthoc_summary,
        "plots": plots,
        "statistical_unit_warning": (
            "Support draws were averaged inside matched pair/seed units; faults repeat across pairs, "
            "so pair-bootstrap intervals do not eliminate repeated-class dependence."
        ),
    }
    atomic_json(table_root / "headline_summary.json", summary)
    return summary
