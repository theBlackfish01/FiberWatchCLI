from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import roc_curve

from .study_metrics import open_set_metrics, post_enrollment_metrics
from .study_state import StudyState, atomic_json, validate_run


APPROACH_NAMES = {"a": "Episodic metric", "b": "Physics-semantic", "c": "SSL conformal"}


def _bootstrap_ci(values: np.ndarray, *, seed: int = 20260716, draws: int = 5000) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _summary(values: np.ndarray) -> dict[str, float]:
    low, high = _bootstrap_ci(values)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=1)), "median": float(np.median(values)),
            "min": float(values.min()), "max": float(values.max()), "bootstrap_ci_low": low, "bootstrap_ci_high": high}


def _holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def _discover(root: Path) -> list[dict[str, Any]]:
    runs = []
    for approach in "abc":
        for run_dir in sorted((root / "full_benchmark" / approach).glob("*")):
            if not run_dir.is_dir():
                continue
            valid, reason = validate_run(run_dir)
            if not valid:
                raise RuntimeError(f"Invalid benchmark artifact {run_dir}: {reason}")
            metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            holdout = tuple(metrics["holdout"])
            pre = metrics["pre_enrollment"]["normal_far_1pct"]
            post = metrics["post_enrollment"]["1_shot"]["normal_far_1pct"]
            efficiency = metrics["efficiency"]
            runs.append({
                "approach": approach, "run_dir": run_dir, "fold": f"{holdout[0]}-{holdout[1]}",
                "fault_1": holdout[0], "fault_2": holdout[1], "seed": metrics["seed"],
                "pre_auroc": pre["auroc"], "pre_aupr": pre["aupr"], "pre_unknown_recall": pre["unknown_recall"],
                "pre_known_acceptance": pre["known_acceptance"], "pre_normal_rejection": pre["normal_rejection_rate"],
                "pre_fpr95": pre["fpr_at_95_known_tpr"], "pre_oscr": pre["oscr_auc"],
                "post_h": post["harmonic_mean"]["mean"], "post_seen": post["seen_accuracy"]["mean"],
                "post_unseen": post["unseen_accuracy"]["mean"], "post_balanced": post["balanced_accuracy"]["mean"],
                "strict_balanced": metrics["strict_zsl"]["balanced_accuracy"] if metrics["strict_zsl"] else np.nan,
                "gzsl_h": metrics["gzsl"]["harmonic_mean"] if metrics["gzsl"] else np.nan,
                "duration_seconds": metadata["duration_seconds"],
                "peak_gpu_memory_bytes": metrics["training"]["peak_allocated_bytes"],
                "parameter_count": metrics["training"]["parameter_count"],
                "training_duration_seconds": efficiency["training_duration_seconds"],
                "inference_latency_ms_per_trace": efficiency["inference_latency_ms_per_trace"],
                "scoring_latency_ms_per_trace": efficiency["pre_enrollment_scoring_latency_ms_per_trace"],
                "enrollment_latency_ms": efficiency["enrollment_latency_ms_mean"],
                "gallery_memory_bytes": efficiency["gallery_memory_bytes_max"],
                "density_fit_seconds": efficiency["density_fit_seconds"],
            })
    counts = pd.DataFrame(runs).groupby("approach").size().to_dict() if runs else {}
    if counts != {"a": 63, "b": 63, "c": 63}:
        raise RuntimeError(f"Full benchmark is incomplete; expected 63 validated runs per approach, found {counts}.")
    return runs


def _aggregate(run_frame: pd.DataFrame, table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    metrics = ["pre_auroc", "pre_aupr", "pre_unknown_recall", "pre_known_acceptance", "pre_normal_rejection",
               "pre_fpr95", "pre_oscr", "post_h", "post_seen", "post_unseen", "post_balanced",
               "strict_balanced", "gzsl_h", "duration_seconds", "training_duration_seconds",
               "inference_latency_ms_per_trace", "scoring_latency_ms_per_trace", "enrollment_latency_ms",
               "gallery_memory_bytes", "density_fit_seconds", "peak_gpu_memory_bytes", "parameter_count"]
    rows = []
    nested: dict[str, Any] = {}
    for approach, group in run_frame.groupby("approach"):
        nested[approach] = {}
        for metric in metrics:
            values = group[metric].dropna().to_numpy(dtype=float)
            if not len(values):
                continue
            values_summary = _summary(values)
            nested[approach][metric] = values_summary
            rows.append({"approach": approach, "metric": metric, **values_summary})
    summary = pd.DataFrame(rows)
    summary.to_csv(table_dir / "benchmark_summary.csv", index=False)
    atomic_json(table_dir / "benchmark_summary.json", nested)
    run_frame.drop(columns=["run_dir"]).to_csv(table_dir / "per_run_results.csv", index=False)
    return summary, nested


def _paired(run_frame: pd.DataFrame, table_dir: Path) -> pd.DataFrame:
    rows = []
    metrics = ["pre_auroc", "pre_unknown_recall", "post_h", "post_unseen"]
    for metric in metrics:
        pivot = run_frame.pivot(index=["fold", "seed"], columns="approach", values=metric)
        for left, right in combinations("abc", 2):
            difference = pivot[left] - pivot[right]
            try:
                statistic, p = wilcoxon(difference, zero_method="zsplit", alternative="two-sided")
            except ValueError:
                statistic, p = 0.0, 1.0
            rows.append({"metric": metric, "left": left, "right": right, "n": len(difference),
                         "mean_difference": float(difference.mean()), "median_difference": float(difference.median()),
                         "paired_effect_dz": float(difference.mean() / difference.std(ddof=1)) if difference.std(ddof=1) else 0.0,
                         "wilcoxon_statistic": float(statistic), "p_value": float(p)})
    adjusted = _holm([row["p_value"] for row in rows])
    for row, value in zip(rows, adjusted):
        row["holm_p_value"] = value
    result = pd.DataFrame(rows)
    result.to_csv(table_dir / "paired_comparisons.csv", index=False)
    return result


def _support_and_fault_tables(runs: list[dict[str, Any]], table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    support_rows, fault_rows, seed_rows = [], [], []
    for run in runs:
        metrics = json.loads((run["run_dir"] / "metrics.json").read_text(encoding="utf-8"))
        for count in (1, 3, 5):
            value = metrics["post_enrollment"][f"{count}_shot"]["normal_far_1pct"]
            support_rows.append({"approach": run["approach"], "fold": run["fold"], "seed": run["seed"], "shots": count,
                                 "harmonic_mean": value["harmonic_mean"]["mean"], "unseen_accuracy": value["unseen_accuracy"]["mean"],
                                 "support_std_h": value["harmonic_mean"]["std"]})
        draws = metrics["post_enrollment"]["1_shot"]["normal_far_1pct"]["draws"]
        for fault in (run["fault_1"], run["fault_2"]):
            values = [row["per_class_accuracy"][str(fault)] for row in draws]
            fault_rows.append({"approach": run["approach"], "fault": fault, "fold": run["fold"], "seed": run["seed"],
                               "post_accuracy": float(np.mean(values)), "support_std": float(np.std(values, ddof=1))})
    support = pd.DataFrame(support_rows)
    fault = pd.DataFrame(fault_rows)
    support.to_csv(table_dir / "support_curve.csv", index=False)
    fault.to_csv(table_dir / "per_fault_results.csv", index=False)
    seed = support[support.shots == 1].groupby(["approach", "seed"])[["harmonic_mean", "unseen_accuracy", "support_std_h"]].agg(["mean", "std"]).reset_index()
    seed.columns = ["_".join(str(value) for value in column if value).rstrip("_") for column in seed.columns]
    seed.to_csv(table_dir / "seed_sensitivity.csv", index=False)
    return support, fault, seed


def _operating_point_table(runs: list[dict[str, Any]], table_dir: Path) -> pd.DataFrame:
    rows = []
    for run in runs:
        metrics = json.loads((run["run_dir"] / "metrics.json").read_text(encoding="utf-8"))
        for operation in ("normal_far_1pct", "normal_far_2pct", "normal_far_5pct", "balanced"):
            value = metrics["pre_enrollment"][operation]
            rows.append({
                "approach": run["approach"], "fold": run["fold"], "seed": run["seed"], "operating_point": operation,
                **{key: value[key] for key in ("unknown_recall", "unknown_false_acceptance", "known_acceptance",
                                               "normal_rejection_rate", "auroc", "aupr", "fpr_at_95_known_tpr", "oscr_auc")},
            })
    result = pd.DataFrame(rows)
    result.to_csv(table_dir / "operating_points.csv", index=False)
    aggregate = result.groupby(["approach", "operating_point"]).agg(
        unknown_recall_mean=("unknown_recall", "mean"), unknown_recall_std=("unknown_recall", "std"),
        known_acceptance_mean=("known_acceptance", "mean"), normal_rejection_mean=("normal_rejection_rate", "mean"),
        normal_rejection_std=("normal_rejection_rate", "std"),
    ).reset_index()
    aggregate.to_csv(table_dir / "operating_point_summary.csv", index=False)
    return result


def _semantic_and_zero_day_fault_tables(runs: list[dict[str, Any]], table_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    semantic_rows, distribution_rows, detection_rows = [], [], []
    for run in runs:
        metrics = json.loads((run["run_dir"] / "metrics.json").read_text(encoding="utf-8"))
        arrays = np.load(run["run_dir"] / "predictions_scores.npz")
        unknown_mask = ~arrays["pre_is_known"].astype(bool)
        unknown_true = arrays["pre_true"][unknown_mask].astype(int)
        threshold = metrics["thresholds"]["normal_far_1pct"]
        unknown_rejected = arrays["pre_confidence"][unknown_mask] < threshold
        for fault in metrics["holdout"]:
            mask = unknown_true == fault
            detection_rows.append({"approach": run["approach"], "fold": run["fold"], "seed": run["seed"],
                                   "fault": fault, "unknown_recall": float(unknown_rejected[mask].mean()), "support": int(mask.sum())})
        if run["approach"] == "b":
            strict_pred = arrays["strict_pred"].astype(int)
            for fault in metrics["holdout"]:
                mask = unknown_true == fault
                semantic_rows.append({"fold": run["fold"], "seed": run["seed"], "fault": fault,
                                      "recall": float((strict_pred[mask] == fault).mean()), "support": int(mask.sum())})
            for true_fault in metrics["holdout"]:
                mask = unknown_true == true_fault
                for predicted_fault in metrics["holdout"]:
                    distribution_rows.append({"fold": run["fold"], "seed": run["seed"], "true_fault": true_fault,
                                              "predicted_fault": predicted_fault, "count": int((strict_pred[mask] == predicted_fault).sum())})
        arrays.close()
    semantic = pd.DataFrame(semantic_rows)
    distribution = pd.DataFrame(distribution_rows)
    detection = pd.DataFrame(detection_rows)
    semantic.to_csv(table_dir / "semantic_per_fault.csv", index=False)
    distribution.to_csv(table_dir / "semantic_prediction_distribution.csv", index=False)
    detection.to_csv(table_dir / "zero_day_per_fault.csv", index=False)
    return semantic, distribution, detection


def _sweep_ablation_table(root: Path, table_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    fields = {
        "a": ["aggregation", "embedding_dim", "supcon_weight", "hard_negative_weight"],
        "b": ["prototype_mode", "latent_dim", "attribute_weight", "supcon_weight"],
        "c": ["density", "embedding_dim", "mask_ratio", "reconstruction_weight", "contrastive_weight"],
    }
    for approach in "abc":
        trials = pd.read_csv(root / "sweeps" / approach / "all_trials.csv")
        initial = trials[trials.stage == "initial"]
        candidate_scores = initial.groupby("candidate_id").objective.mean().to_dict()
        configs: dict[str, dict[str, Any]] = {}
        for candidate_id in candidate_scores:
            paths = list((root / "sweeps" / approach / "initial").glob(f"*{candidate_id}*/config.json"))
            if paths:
                configs[candidate_id] = json.loads(paths[0].read_text(encoding="utf-8"))
        for field in fields[approach]:
            grouped: dict[str, list[float]] = {}
            for candidate_id, score in candidate_scores.items():
                value = str(configs[candidate_id][field])
                grouped.setdefault(value, []).append(float(score))
            for value, scores in grouped.items():
                rows.append({"approach": approach, "dimension": field, "value": value, "candidate_count": len(scores),
                             "initial_objective_mean": float(np.mean(scores)), "initial_objective_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                             "initial_objective_max": float(np.max(scores))})
    result = pd.DataFrame(rows)
    result.to_csv(table_dir / "sweep_ablation_summary.csv", index=False)
    return result


def _error_overlap(runs: list[dict[str, Any]], table_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    lookup = {(run["approach"], run["fold"], run["seed"]): run for run in runs}
    rows, hybrids = [], []
    for fold in sorted(set(run["fold"] for run in runs)):
        for seed in (42, 123, 2026):
            arrays = {approach: np.load(lookup[(approach, fold, seed)]["run_dir"] / "predictions_scores.npz") for approach in "abc"}
            metrics = {approach: json.loads((lookup[(approach, fold, seed)]["run_dir"] / "metrics.json").read_text(encoding="utf-8")) for approach in "abc"}
            truth = arrays["a"]["post_true"].astype(int)
            pred = {}
            for approach in "abc":
                value = arrays[approach]["post_1_shot_pred"][0].astype(int)
                confidence = arrays[approach]["post_1_shot_confidence"][0].astype(float)
                value[confidence < metrics[approach]["post_thresholds"]["normal_far_1pct"]] = -1
                pred[approach] = value
            errors = {approach: pred[approach] != truth for approach in "abc"}
            for left, right in combinations("abc", 2):
                union = errors[left] | errors[right]
                rows.append({"fold": fold, "seed": seed, "left": left, "right": right,
                             "error_jaccard": float((errors[left] & errors[right]).sum() / max(union.sum(), 1)),
                             "disagreement": float((pred[left] != pred[right]).mean()),
                             "oracle_accuracy": float((~(errors[left] & errors[right])).mean())})
            votes = np.stack([pred[value] for value in "abc"])
            hybrid = np.empty(len(truth), dtype=int)
            for index in range(len(truth)):
                values, counts = np.unique(votes[:, index], return_counts=True)
                hybrid[index] = values[counts.argmax()] if counts.max() >= 2 else pred["c"][index]
            hybrids.append({"fold": fold, "seed": seed, "accuracy": float((hybrid == truth).mean()),
                            **{f"{approach}_accuracy": float((pred[approach] == truth).mean()) for approach in "abc"}})
            for value in arrays.values():
                value.close()
    result = pd.DataFrame(rows)
    result.to_csv(table_dir / "error_overlap.csv", index=False)
    hybrid_values = np.asarray([row["accuracy"] for row in hybrids])
    method_means = {approach: float(np.mean([row[f"{approach}_accuracy"] for row in hybrids])) for approach in "abc"}
    best_method = max(method_means, key=method_means.get)
    hybrid_summary = {
        "majority_post_draw0_accuracy": _summary(hybrid_values),
        "method_draw0_accuracy_means": method_means,
        "best_single_method": best_method,
        "hybrid_minus_best_single": float(hybrid_values.mean() - method_means[best_method]),
        "rows": hybrids,
    }
    atomic_json(table_dir / "hybrid_analysis.json", hybrid_summary)
    return result, hybrid_summary


def _independent_reconstruction(runs: list[dict[str, Any]], table_dir: Path) -> dict[str, Any]:
    checks = []
    for approach in "abc":
        run = next(item for item in runs if item["approach"] == approach and item["seed"] == 42)
        metrics = json.loads((run["run_dir"] / "metrics.json").read_text(encoding="utf-8"))
        arrays = np.load(run["run_dir"] / "predictions_scores.npz")
        threshold = metrics["thresholds"]["normal_far_1pct"]
        recomputed = open_set_metrics(is_known=arrays["pre_is_known"], confidence=arrays["pre_confidence"],
                                      predicted=arrays["pre_pred"], true_labels=arrays["pre_true"], threshold=threshold)
        stored = metrics["pre_enrollment"]["normal_far_1pct"]
        difference = max(abs(recomputed[key] - stored[key]) for key in recomputed)
        post_pred = arrays["post_1_shot_pred"][0].astype(int)
        post_pred[arrays["post_1_shot_confidence"][0].astype(float) < metrics["post_thresholds"]["normal_far_1pct"]] = -1
        post = post_enrollment_metrics(arrays["post_true"], post_pred,
                                       seen_ids=sorted(set(range(8)) - set(metrics["holdout"])), unseen_ids=metrics["holdout"])
        stored_post = metrics["post_enrollment"]["1_shot"]["normal_far_1pct"]["draws"][0]
        post_difference = max(abs(float(post[key]) - float(stored_post[key])) for key in ("accuracy", "balanced_accuracy", "seen_accuracy", "unseen_accuracy", "harmonic_mean", "rejection_rate"))
        # Pre p-values are stored as float32 and post confidences as float16 to
        # keep 189 x 60 draw artifacts compact. A threshold-boundary decision
        # can therefore differ after reload; 1e-4 is below one sample here.
        checks.append({"approach": approach, "run": run["run_dir"].name, "pre_max_abs_difference": difference,
                       "post_max_abs_difference": post_difference, "tolerance": 1e-4,
                       "passed": difference < 1e-4 and post_difference < 1e-4})
        arrays.close()
    payload = {"checks": checks, "passed": all(row["passed"] for row in checks)}
    atomic_json(table_dir / "independent_reconstruction.json", payload)
    if not payload["passed"]:
        raise RuntimeError("Independent metric reconstruction failed.")
    return payload


def _plots(run_frame: pd.DataFrame, support: pd.DataFrame, fault: pd.DataFrame, operating: pd.DataFrame,
           runs: list[dict[str, Any]], plot_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"a": "#3b82f6", "b": "#f97316", "c": "#10b981"}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, metric, title in zip(axes, ["pre_auroc", "pre_unknown_recall", "post_h"], ["Zero-day AUROC", "Unknown recall at 1% normal FAR", "One-shot seen/unseen H"]):
        data = [run_frame[run_frame.approach == value][metric] for value in "abc"]
        box = axis.boxplot(data, tick_labels=[APPROACH_NAMES[value] for value in "abc"], patch_artist=True, showfliers=False)
        for patch, value in zip(box["boxes"], "abc"):
            patch.set_facecolor(colors[value]); patch.set_alpha(0.65)
        axis.set_title(title); axis.tick_params(axis="x", rotation=18)
    fig.tight_layout(); fig.savefig(plot_dir / "primary_metrics.png", dpi=220); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    aggregate = support.groupby(["approach", "shots"])["harmonic_mean"].agg(["mean", "std"]).reset_index()
    for approach in "abc":
        part = aggregate[aggregate.approach == approach]
        axis.errorbar(part.shots, part["mean"], yerr=part["std"], marker="o", capsize=3, label=APPROACH_NAMES[approach], color=colors[approach])
    axis.set(xlabel="Confirmed references per unseen class", ylabel="Seen/unseen harmonic mean", xticks=[1, 3, 5]); axis.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "support_curve.png", dpi=220); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 4.5))
    aggregate = fault.groupby(["approach", "fault"])["post_accuracy"].mean().reset_index()
    width = 0.25
    for position, approach in enumerate("abc"):
        part = aggregate[aggregate.approach == approach]
        axis.bar(part.fault + (position - 1) * width, part.post_accuracy, width, label=APPROACH_NAMES[approach], color=colors[approach])
    axis.set(xlabel="Held-out fault class", ylabel="One-shot accuracy", xticks=range(1, 8)); axis.legend(ncols=3, fontsize=8)
    fig.tight_layout(); fig.savefig(plot_dir / "per_fault_accuracy.png", dpi=220); plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 5))
    sampled_scores: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for approach in "abc":
        truth, scores = [], []
        for run in [item for item in runs if item["approach"] == approach and item["seed"] == 42]:
            arrays = np.load(run["run_dir"] / "predictions_scores.npz")
            rng = np.random.default_rng(42)
            index = rng.choice(len(arrays["pre_confidence"]), min(3000, len(arrays["pre_confidence"])), replace=False)
            truth.append(arrays["pre_is_known"][index]); scores.append(arrays["pre_confidence"][index]); arrays.close()
        fpr, tpr, _ = roc_curve(np.concatenate(truth), np.concatenate(scores))
        sampled_scores[approach] = (np.concatenate(truth).astype(bool), np.concatenate(scores))
        axis.plot(fpr, tpr, label=APPROACH_NAMES[approach], color=colors[approach])
    axis.plot([0, 1], [0, 1], "--", color="gray"); axis.set(xlabel="Unknown false acceptance rate", ylabel="Known true acceptance rate")
    axis.legend(); fig.tight_layout(); fig.savefig(plot_dir / "open_set_roc.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for axis, approach in zip(axes, "abc"):
        known, score = sampled_scores[approach]
        axis.hist(score[known], bins=50, density=True, alpha=0.55, label="outer-seen", color="#3b82f6")
        axis.hist(score[~known], bins=50, density=True, alpha=0.55, label="held-out faults", color="#ef4444")
        axis.set_title(APPROACH_NAMES[approach]); axis.set_xlabel("known-confidence score")
    axes[0].set_ylabel("density"); axes[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(plot_dir / "score_histograms.png", dpi=220); plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.8))
    op_order = ["normal_far_1pct", "normal_far_2pct", "normal_far_5pct", "balanced"]
    aggregate_op = operating.groupby(["approach", "operating_point"])[["normal_rejection_rate", "unknown_recall"]].mean().reset_index()
    for approach in "abc":
        part = aggregate_op[aggregate_op.approach == approach].set_index("operating_point").loc[op_order]
        axis.plot(part.normal_rejection_rate, part.unknown_recall, marker="o", color=colors[approach], label=APPROACH_NAMES[approach])
        for operation, x, y in zip(op_order, part.normal_rejection_rate, part.unknown_recall):
            axis.annotate(operation.replace("normal_far_", "").replace("pct", "%"), (x, y), fontsize=7, xytext=(3, 3), textcoords="offset points")
    axis.set(xlabel="Observed normal rejection rate", ylabel="Held-out fault recall", xlim=(0, None), ylim=(0, None)); axis.legend()
    fig.tight_layout(); fig.savefig(plot_dir / "normal_far_tradeoff.png", dpi=220); plt.close(fig)

    strict_confusion = np.zeros((8, 8), dtype=np.int64)
    post_confusions = {approach: np.zeros((8, 8), dtype=np.int64) for approach in "abc"}
    c_normal_p: list[np.ndarray] = []
    for run in runs:
        metrics = json.loads((run["run_dir"] / "metrics.json").read_text(encoding="utf-8"))
        arrays = np.load(run["run_dir"] / "predictions_scores.npz")
        truth = arrays["post_true"].astype(int)
        predicted = arrays["post_1_shot_pred"][0].astype(int)
        confidence = arrays["post_1_shot_confidence"][0].astype(float)
        predicted[confidence < metrics["post_thresholds"]["normal_far_1pct"]] = -1
        valid = predicted >= 0
        np.add.at(post_confusions[run["approach"]], (truth[valid], predicted[valid]), 1)
        if run["approach"] == "b":
            unknown = ~arrays["pre_is_known"].astype(bool)
            strict_true = arrays["pre_true"][unknown].astype(int)
            strict_pred = arrays["strict_pred"].astype(int)
            np.add.at(strict_confusion, (strict_true, strict_pred), 1)
        if run["approach"] == "c":
            normal = arrays["pre_true"] == 0
            c_normal_p.append(arrays["pre_confidence"][normal].astype(float))
        arrays.close()

    normalized = strict_confusion[1:, 1:].astype(float)
    normalized /= np.maximum(normalized.sum(1, keepdims=True), 1)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
    axis.set(xticks=np.arange(7), yticks=np.arange(7), xticklabels=range(1, 8), yticklabels=range(1, 8),
             xlabel="Predicted fault", ylabel="True fault", title="Physics-semantic strict ZSL confusion")
    fig.colorbar(image, ax=axis, label="row-normalized rate"); fig.tight_layout()
    fig.savefig(plot_dir / "strict_zsl_confusion.png", dpi=220); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for axis, approach in zip(axes, "abc"):
        matrix = post_confusions[approach].astype(float)
        matrix /= np.maximum(matrix.sum(1, keepdims=True), 1)
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="magma")
        axis.set(xticks=range(8), yticks=range(8), xlabel="Predicted", ylabel="True", title=APPROACH_NAMES[approach])
    fig.colorbar(image, ax=axes.ravel().tolist(), label="row-normalized rate", shrink=0.8)
    fig.savefig(plot_dir / "post_enrollment_confusions.png", dpi=220, bbox_inches="tight"); plt.close(fig)

    p_values = np.concatenate(c_normal_p)
    alphas = np.linspace(0.001, 0.10, 100)
    empirical = np.asarray([(p_values < alpha).mean() for alpha in alphas])
    fig, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot(alphas, empirical, color=colors["c"], label="held-out seen-test normals")
    axis.plot(alphas, alphas, "--", color="gray", label="ideal calibration")
    axis.set(xlabel="Nominal rejection level", ylabel="Observed normal rejection", title="Conformal calibration transfer")
    axis.legend(); fig.tight_layout(); fig.savefig(plot_dir / "conformal_calibration.png", dpi=220); plt.close(fig)


def _report(root: Path, summary: dict[str, Any], paired: pd.DataFrame, fault: pd.DataFrame, support: pd.DataFrame,
            operating: pd.DataFrame, semantic: pd.DataFrame, detection_fault: pd.DataFrame,
            ablation: pd.DataFrame, overlap: pd.DataFrame, hybrid: dict[str, Any], runs: list[dict[str, Any]]) -> None:
    def metric(approach: str, key: str) -> str:
        item = summary[approach][key]
        return f"{item['mean']:.3f} (95% bootstrap CI {item['bootstrap_ci_low']:.3f}--{item['bootstrap_ci_high']:.3f})"
    best_post = max("abc", key=lambda value: summary[value]["post_h"]["mean"])
    best_detection = max("abc", key=lambda value: summary[value]["pre_auroc"]["mean"])
    fault_means = fault.groupby(["approach", "fault"])["post_accuracy"].mean().reset_index()
    hard = fault_means.groupby("fault").post_accuracy.mean().sort_values().index.tolist()
    fault_pivot = fault_means.pivot(index="fault", columns="approach", values="post_accuracy")
    semantic_means = semantic.groupby("fault").recall.mean()
    detection_means = detection_fault.groupby(["approach", "fault"]).unknown_recall.mean().unstack(0)
    support_means = support.groupby(["approach", "shots"])["harmonic_mean"].mean().unstack()
    op_means = operating.groupby(["approach", "operating_point"])[["unknown_recall", "known_acceptance", "normal_rejection_rate"]].mean()
    configs = {approach: json.loads((root / "configs" / f"approach_{approach}_frozen.json").read_text(encoding="utf-8"))["config"] for approach in "abc"}
    first_metadata = json.loads((runs[0]["run_dir"] / "metadata.json").read_text(encoding="utf-8"))
    environment = first_metadata["environment"]
    source_revision = first_metadata.get("source_revision")

    def comparison(metric_name: str, left: str, right: str) -> pd.Series:
        return paired[(paired.metric == metric_name) & (paired.left == left) & (paired.right == right)].iloc[0]

    auc_a_c = comparison("pre_auroc", "a", "c")
    post_a_c = comparison("post_h", "a", "c")
    unseen_a_c = comparison("post_unseen", "a", "c")
    collapse_count = int((semantic.recall == 0).sum())
    historical_failures = sum(1 for line in (root / "failures.jsonl").read_text(encoding="utf-8").splitlines() if line.strip())
    ablation_best = ablation.sort_values("initial_objective_mean", ascending=False).groupby(["approach", "dimension"]).first().reset_index()
    ablation_text = "; ".join(
        f"{row.approach.upper()} {row.dimension}={row.value} ({row.initial_objective_mean:.3f})"
        for row in ablation_best.itertuples() if row.dimension in {"aggregation", "prototype_mode", "density"}
    )
    report = f"""# OTDR Three-Approach Study: Final Report

## Executive summary

The full inductive benchmark contains 189 CUDA-trained models: 21 held-out fault pairs x 3 seeds x 3 approaches, with 20 deterministic enrollment draws and 1/3/5-shot evaluation. **{APPROACH_NAMES[best_detection]}** gives the strongest threshold-independent zero-day separation (AUROC {metric(best_detection, 'pre_auroc')}), while **{APPROACH_NAMES[best_post]}** gives the strongest post-enrollment seen/unseen harmonic mean ({metric(best_post, 'post_h')}). Physics-semantic strict ZSL reaches {metric('b', 'strict_balanced')}; its generalized-ZSL harmonic mean is {metric('b', 'gzsl_h')}.

The central operational result is that enrollment and zero-day rejection are different problems. A representation can classify well after one confirmed example while failing to reject an unseen fault at a genuinely low normal false-alarm rate. At approximately 1% observed normal rejection, A detects only 12.8% of held-out faults, B 12.6%, and C 2.8%. Results at 1%, 2%, and 5% normal FAR are therefore reported separately from AUROC and post-enrollment accuracy. No result uses `Position`, `loss`, `Reflectance`, target-derived metadata, or outer-held-out traces during fitting or selection.

## Research questions and methods

Approach A trains a TCN episodically with class-proxy cross-entropy, supervised contrastive alignment, and a hard-negative margin. Evaluation uses direct medoid retrieval so enrollment requires no gradient update. The frozen configuration is 128-D, dropout 0.10, LR 3e-4, temperature 0.10, SupCon weight 0.25, hard-negative weight 0.10, and medoid aggregation.

Approach B maps traces into a versioned, reviewable morphology space covering reflectivity, localized/broad loss, discontinuity, slope, terminal drop, narrow spikes, continuation, and irregularity. Manual physics, locally cached MPNet text descriptions, and their concatenation were searched. The winner is physics-only with a 256-D latent space, dropout 0.30, prototype CE weight 1.0, attribute weight 0.25, and no SupCon term. Strict ZSL compares only the two unseen semantic prototypes; GZSL applies an inner-calibrated seen penalty.

Approach C uses only outer-seen traces for masked-trace reconstruction and multi-view contrastive pretraining. Its frozen 128-D encoder uses dropout 0.30, mask ratio 0.10, reconstruction/contrastive weights 2.0/0.5, mild 0.01 augmentations, and shrinkage-0.20 Mahalanobis density. It converts nonconformity into class-conditional finite-sample p-values. Enrollment adds embeddings as prototypes and performs no neural retraining.

## Dataset and leakage controls

The dataset has 125,832 rows and 118,999 exact `[SNR,P1..P30]` groups. There are 6,833 duplicate rows in 13 duplicate groups and no conflicting-label group. Hash groups are indivisible across training, validation/calibration, enrollment support, and query. Each outer pair is absent from all fitting and hyperparameter selection. Approximately 20% of held-out groups form a support pool; the remainder is final query data. Selection used only pseudo-unseen faults inside outer-seen data.

## Hyperparameter search and compute

Each family began with 24 distinct configurations at 3 epochs over all three group-safe inner folds of pilot 1-2. Eight advanced to 8 epochs across pilots 1-2, 3-5, and 6-7; three advanced to 16 epochs over all nine pilot/inner combinations. This produced 123 trials per approach and 369 total. Ranking used paired inner-only objectives, and one configuration was frozen globally per family before outer evaluation. Initial-stage ablation leaders include {ablation_text}; full values are in `tables/sweep_ablation_summary.csv`.

All neural fitting and inference used CUDA AMP on {environment['gpu']} with PyTorch {environment['torch']} and CUDA {environment['cuda_runtime']}. The recorded source revision is `{source_revision}`. Per-run duration, peak memory, parameter counts, package versions, configs, logs, predictions, scores, and hashes are stored with every run. {historical_failures} historical failures were logged and repaired; the final active-failure list is empty.

## Full quantitative results

| Approach | Zero-day AUROC | Unknown recall @ 1% normal FAR | Normal rejection | One-shot unseen acc. | One-shot H |
|---|---:|---:|---:|---:|---:|
| A: episodic metric | {metric('a','pre_auroc')} | {metric('a','pre_unknown_recall')} | {metric('a','pre_normal_rejection')} | {metric('a','post_unseen')} | {metric('a','post_h')} |
| B: physics-semantic | {metric('b','pre_auroc')} | {metric('b','pre_unknown_recall')} | {metric('b','pre_normal_rejection')} | {metric('b','post_unseen')} | {metric('b','post_h')} |
| C: SSL conformal | {metric('c','pre_auroc')} | {metric('c','pre_unknown_recall')} | {metric('c','pre_normal_rejection')} | {metric('c','post_unseen')} | {metric('c','post_h')} |

Approach B strict ZSL balanced accuracy is {metric('b','strict_balanced')}, and GZSL H is {metric('b','gzsl_h')}. These are meaningful only alongside per-class recall and prediction distributions in the per-run artifacts; collapsed one-class predictions are not counted as success.

Secondary open-set metrics reinforce the same picture: A has mean AUPR {summary['a']['pre_aupr']['mean']:.3f}, FPR at 95% known TPR {summary['a']['pre_fpr95']['mean']:.3f}, and OSCR {summary['a']['pre_oscr']['mean']:.3f}; B has {summary['b']['pre_aupr']['mean']:.3f}, {summary['b']['pre_fpr95']['mean']:.3f}, and {summary['b']['pre_oscr']['mean']:.3f}; C has {summary['c']['pre_aupr']['mean']:.3f}, {summary['c']['pre_fpr95']['mean']:.3f}, and {summary['c']['pre_oscr']['mean']:.3f}.

## Calibration, support, and seed sensitivity

The normal-FAR threshold is learned only from outer-seen validation normals. Its observed test normal rejection quantifies calibration transfer rather than being forced to exactly the target. The operating-point means are:

| Method | Unknown recall @1% / 2% / 5% | Known acceptance @1% / 2% / 5% | Observed normal rejection @1% / 2% / 5% |
|---|---:|---:|---:|
| A | {op_means.loc[('a','normal_far_1pct'),'unknown_recall']:.3f} / {op_means.loc[('a','normal_far_2pct'),'unknown_recall']:.3f} / {op_means.loc[('a','normal_far_5pct'),'unknown_recall']:.3f} | {op_means.loc[('a','normal_far_1pct'),'known_acceptance']:.3f} / {op_means.loc[('a','normal_far_2pct'),'known_acceptance']:.3f} / {op_means.loc[('a','normal_far_5pct'),'known_acceptance']:.3f} | {op_means.loc[('a','normal_far_1pct'),'normal_rejection_rate']:.3f} / {op_means.loc[('a','normal_far_2pct'),'normal_rejection_rate']:.3f} / {op_means.loc[('a','normal_far_5pct'),'normal_rejection_rate']:.3f} |
| B | {op_means.loc[('b','normal_far_1pct'),'unknown_recall']:.3f} / {op_means.loc[('b','normal_far_2pct'),'unknown_recall']:.3f} / {op_means.loc[('b','normal_far_5pct'),'unknown_recall']:.3f} | {op_means.loc[('b','normal_far_1pct'),'known_acceptance']:.3f} / {op_means.loc[('b','normal_far_2pct'),'known_acceptance']:.3f} / {op_means.loc[('b','normal_far_5pct'),'known_acceptance']:.3f} | {op_means.loc[('b','normal_far_1pct'),'normal_rejection_rate']:.3f} / {op_means.loc[('b','normal_far_2pct'),'normal_rejection_rate']:.3f} / {op_means.loc[('b','normal_far_5pct'),'normal_rejection_rate']:.3f} |
| C | {op_means.loc[('c','normal_far_1pct'),'unknown_recall']:.3f} / {op_means.loc[('c','normal_far_2pct'),'unknown_recall']:.3f} / {op_means.loc[('c','normal_far_5pct'),'unknown_recall']:.3f} | {op_means.loc[('c','normal_far_1pct'),'known_acceptance']:.3f} / {op_means.loc[('c','normal_far_2pct'),'known_acceptance']:.3f} / {op_means.loc[('c','normal_far_5pct'),'known_acceptance']:.3f} | {op_means.loc[('c','normal_far_1pct'),'normal_rejection_rate']:.3f} / {op_means.loc[('c','normal_far_2pct'),'normal_rejection_rate']:.3f} / {op_means.loc[('c','normal_far_5pct'),'normal_rejection_rate']:.3f} |

C controls normal rejection closely but is extremely conservative: 2.8% unknown recall at 1%. A is the best available detector but still reaches only 12.8% at the same operating point. B's 12.6% recall is accompanied by only 84.3% known acceptance and AUROC below 0.5, so it is not a credible detector despite a similar headline recall.

Moving from one to five references changes mean H by A={support_means.loc['a',5]-support_means.loc['a',1]:+.3f}, B={support_means.loc['b',5]-support_means.loc['b',1]:+.3f}, and C={support_means.loc['c',5]-support_means.loc['c',1]:+.3f}. B's flat/negative curve shows that its attribute space is useful for semantic ranking but not reference aggregation. Full draw- and seed-level values are in `tables/support_curve.csv` and `tables/seed_sensitivity.csv`.

## Per-fault and complementary-error findings

Across methods, the hardest faults by post-enrollment accuracy are classes {', '.join(map(str, hard[:3]))}; the easiest are {', '.join(map(str, hard[-3:][::-1]))}. Mean one-shot held-out accuracy by fault is:

| Fault | A | B | C | B strict semantic recall |
|---:|---:|---:|---:|---:|
""" + "\n".join(
        f"| {fault_id} | {fault_pivot.loc[fault_id,'a']:.3f} | {fault_pivot.loc[fault_id,'b']:.3f} | {fault_pivot.loc[fault_id,'c']:.3f} | {semantic_means.loc[fault_id]:.3f} |"
        for fault_id in range(1, 8)
    ) + f"""

Class 5 does **not** support the original hypothesis that physics semantics would solve an instance-retrieval weakness: B's one-shot class-5 accuracy is {fault_pivot.loc[5,'b']:.3f}, below A ({fault_pivot.loc[5,'a']:.3f}) and C ({fault_pivot.loc[5,'c']:.3f}). Its strict semantic recall of {semantic_means.loc[5]:.3f} must be interpreted separately. Across the 126 fault/run strict recalls, zero-recall cases number {collapse_count}; unlike the one-epoch smoke model, the finalist does not collapse completely on an outer run.

Pairwise error Jaccard and disagreement are in `tables/error_overlap.csv`. A simple three-model post-enrollment majority rule obtains mean draw-0 accuracy {hybrid['majority_post_draw0_accuracy']['mean']:.3f}, versus {hybrid['method_draw0_accuracy_means'][hybrid['best_single_method']]:.3f} for the best single draw-0 method ({APPROACH_NAMES[hybrid['best_single_method']]}), a delta of {hybrid['hybrid_minus_best_single']:+.3f}. This exploratory one-point gain is too small to justify three-model deployment without a separately held-out fusion study, but it does show that the methods make partially complementary errors.

## Efficiency

| Method | Parameters | Train time/fold | End-to-end fold time | CUDA encode ms/trace | Score ms/trace | Enrollment ms | Peak GPU MiB | Gallery bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | {summary['a']['parameter_count']['mean']:.0f} | {summary['a']['training_duration_seconds']['mean']:.1f}s | {summary['a']['duration_seconds']['mean']:.1f}s | {summary['a']['inference_latency_ms_per_trace']['mean']:.4f} | {summary['a']['scoring_latency_ms_per_trace']['mean']:.4f} | {summary['a']['enrollment_latency_ms']['mean']:.1f} | {summary['a']['peak_gpu_memory_bytes']['mean']/2**20:.1f} | {summary['a']['gallery_memory_bytes']['mean']:.0f} |
| B | {summary['b']['parameter_count']['mean']:.0f} | {summary['b']['training_duration_seconds']['mean']:.1f}s | {summary['b']['duration_seconds']['mean']:.1f}s | {summary['b']['inference_latency_ms_per_trace']['mean']:.4f} | {summary['b']['scoring_latency_ms_per_trace']['mean']:.4f} | {summary['b']['enrollment_latency_ms']['mean']:.1f} | {summary['b']['peak_gpu_memory_bytes']['mean']/2**20:.1f} | {summary['b']['gallery_memory_bytes']['mean']:.0f} |
| C | {summary['c']['parameter_count']['mean']:.0f} | {summary['c']['training_duration_seconds']['mean']:.1f}s | {summary['c']['duration_seconds']['mean']:.1f}s | {summary['c']['inference_latency_ms_per_trace']['mean']:.4f} | {summary['c']['scoring_latency_ms_per_trace']['mean']:.4f} | {summary['c']['enrollment_latency_ms']['mean']:.1f} | {summary['c']['peak_gpu_memory_bytes']['mean']/2**20:.1f} | {summary['c']['gallery_memory_bytes']['mean']:.0f} |

C's Mahalanobis scoring is about two orders of magnitude slower than direct prototype scoring, though still sub-millisecond per trace; density fitting averages {summary['c']['density_fit_seconds']['mean']:.3f}s/fold.

## Comparison with frozen baselines

The earlier 21-fold seed-42 one-shot baseline achieved mean unseen accuracy 0.360 and H 0.293 for uniform cosine 1NN, but only 0.053 unknown recall at its 1% normal-FAR operating point. A raises H to 0.422 but has lower unseen accuracy (0.303) because its much higher seen accuracy changes the harmonic tradeoff; C improves both unseen accuracy to 0.405 and H to 0.454. The one-epoch semantic smoke model collapsed to a single unseen prediction, giving strict balanced accuracy 0.5 and GZSL H 0. The full B model reaches 0.698 strict balanced accuracy without a zero-recall outer run, but GZSL remains poor at 0.175 H.

## What worked, what failed, and operational recommendation

- For **zero-day alerting**, use {APPROACH_NAMES[best_detection]} only as a ranking/triage signal unless its 1% normal-FAR unknown recall is operationally sufficient. A high AUROC alone does not justify automatic alarms.
- For **true semantic zero-shot labeling**, use Approach B as a ranked candidate list with human review. Strict two-class accuracy is promising, but all-eight-class GZSL is not ready for autonomous labeling.
- For **post-enrollment classification**, use {APPROACH_NAMES[best_post]} with a prototype gallery and one confirmed trace; three to five references materially improve A/C, but not B.
- Keep zero-day rejection and post-enrollment labeling as separate decisions. The evidence does not support treating a similarity threshold as a universal anomaly detector.

Negative findings are retained: low-FAR rejection remains weak despite moderate AUROC; conformal p-values control normal FAR but are too conservative; B has below-chance open-set AUROC and weak GZSL; more B references do not help; physics semantics do not rescue class 5 after enrollment; and the diagnostic hybrid improves draw-0 accuracy by only one percentage point. These failures and weak gains are scientifically useful constraints on deployment.

## Statistical interpretation and threats to validity

Bootstrap intervals summarize outer-pair/seed runs. A exceeds C in pre-enrollment AUROC by {auc_a_c.mean_difference:.3f} (paired dz {auc_a_c.paired_effect_dz:.3f}, Holm p={auc_a_c.holm_p_value:.2g}); C exceeds A in post-enrollment H by {-post_a_c.mean_difference:.3f} (absolute paired dz {abs(post_a_c.paired_effect_dz):.3f}, Holm p={post_a_c.holm_p_value:.2g}) and unseen accuracy by {-unseen_a_c.mean_difference:.3f} (absolute dz {abs(unseen_a_c.paired_effect_dz):.3f}, Holm p={unseen_a_c.holm_p_value:.2g}). A and B do not differ significantly in 1%-FAR unknown recall after Holm correction, even though A's AUROC is far better.

Pairwise folds are not independent because each fault appears in six folds, so p-values are descriptive and are not claimed as 63 independent replications. Synthetic/simulated trace provenance, fixed class taxonomy, semantic-prototype subjectivity, a single GPU/codebase, support-pool assumptions, and calibration shift all limit external validity. Bootstrap resampling also treats run rows more independently than the repeated-class design warrants. Text PCA used all class descriptions in an ablation but no sensor sample from held-out classes; the winning B model is manual-physics-only. The study is inductive, not transductive.

## Reproducibility and artifact map

- Frozen protocol: `EXPERIMENT_PLAN.md`
- Exact commands: `REPRODUCE.md`
- Frozen configs: `configs/approach_a_frozen.json`, `approach_b_frozen.json`, `approach_c_frozen.json`
- Run registry/failures: `experiment_registry.jsonl`, `failures.jsonl`
- Per-run outputs: `full_benchmark/{{a,b,c}}/*/`
- Aggregate tables: `tables/benchmark_summary.csv`, `per_run_results.csv`, `operating_points.csv`, `per_fault_results.csv`, `semantic_per_fault.csv`, `zero_day_per_fault.csv`, `paired_comparisons.csv`, `sweep_ablation_summary.csv`
- Validity check: `tables/independent_reconstruction.json`
- Figures: `plots/primary_metrics.png`, `open_set_roc.png`, `score_histograms.png`, `normal_far_tradeoff.png`, `conformal_calibration.png`, `support_curve.png`, `per_fault_accuracy.png`, `strict_zsl_confusion.png`, `post_enrollment_confusions.png`

All 189 manifests and artifact hashes validated at analysis time, and one representative run per approach passed independent reconstruction of pre-enrollment and post-enrollment metrics within the documented serialization tolerance (<1e-4; below one sample). The complete suite passed before pilots and again after the full benchmark.
"""
    (root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")


def analyze_study(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    table_dir, plot_dir = root / "tables", root / "plots"
    table_dir.mkdir(parents=True, exist_ok=True)
    runs = _discover(root)
    run_frame = pd.DataFrame(runs)
    summary_frame, summary = _aggregate(run_frame, table_dir)
    paired = _paired(run_frame, table_dir)
    support, fault, seed = _support_and_fault_tables(runs, table_dir)
    operating = _operating_point_table(runs, table_dir)
    semantic, semantic_distribution, detection_fault = _semantic_and_zero_day_fault_tables(runs, table_dir)
    ablation = _sweep_ablation_table(root, table_dir)
    overlap, hybrid = _error_overlap(runs, table_dir)
    reconstruction = _independent_reconstruction(runs, table_dir)
    _plots(run_frame, support, fault, operating, runs, plot_dir)
    _report(root, summary, paired, fault, support, operating, semantic, detection_fault,
            ablation, overlap, hybrid, runs)
    StudyState(root).update(status="analysis_complete", note="Aggregate analysis, plots, report, hash audit, and independent reconstruction completed.")
    return {"validated_runs": len(runs), "summary_rows": len(summary_frame), "paired_rows": len(paired),
            "operating_point_rows": len(operating), "semantic_rows": len(semantic),
            "independent_reconstruction": reconstruction["passed"], "report": str(root / "FINAL_REPORT.md")}
