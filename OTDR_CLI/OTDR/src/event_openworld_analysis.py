from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve

from .event_openworld_metrics import NormalOnlyCalibrator, semantic_metrics
from .study_metrics import post_enrollment_metrics
from .study_state import atomic_json, validate_run


def collect_results(study_root: Path) -> dict[str, pd.DataFrame]:
    runs, inductive, sgme, raw, faults, semantic_faults, open_baselines = [], [], [], [], [], [], []
    validation_failures = []
    source_path = study_root / "configs" / "source_manifest.json"
    if not source_path.exists():
        raise FileNotFoundError("Frozen source manifest missing; run the finalist pilot first.")
    runtime_source_sha256 = json.loads(source_path.read_text(encoding="utf-8"))["runtime_source_sha256"]
    for metrics_path in sorted((study_root / "full_benchmark").rglob("metrics.json")):
        run_dir = metrics_path.parent
        valid, reason = validate_run(run_dir, {
            "runtime_source_sha256": runtime_source_sha256,
            "cuda_verified": True,
        })
        if not valid:
            validation_failures.append({"run": str(run_dir), "reason": reason})
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        holdout = tuple(metrics["holdout"])
        op = metrics["zero_day"]["operating_points"]
        row = {
            "run_id": run_dir.name, "approach": metrics["approach"], "fold": f"{holdout[0]}-{holdout[1]}",
            "fault_1": holdout[0], "fault_2": holdout[1], "seed": metrics["seed"],
            "auroc": metrics["zero_day"]["auroc"], "aupr": metrics["zero_day"]["aupr"],
            "pauroc_0_01": metrics["zero_day"]["pauroc_0_01"], "pauroc_0_05": metrics["zero_day"]["pauroc_0_05"],
            "oscr": metrics["zero_day"]["oscr"],
            "fpr_at_95_unknown_tpr": metrics["zero_day"]["fpr_at_95_unknown_tpr"],
            "unknown_false_acceptance_at_95_known_acceptance": metrics["zero_day"]["unknown_false_acceptance_at_95_known_acceptance"],
            "strict_balanced": metrics["semantic"]["strict_zsl_balanced_accuracy"],
            "gzsl_seen": metrics["semantic"]["gzsl_seen_accuracy"],
            "gzsl_unseen": metrics["semantic"]["gzsl_unseen_accuracy"],
            "gzsl_h": metrics["semantic"]["gzsl_harmonic_mean"],
            "class_collapse": metrics["semantic"]["gzsl_class_collapse_count"],
            **{f"{name}_{field}": values[field] for name, values in op.items()
               for field in ("observed_normal_far", "unknown_recall", "known_acceptance", "worst_fault_recall")},
            **metrics["efficiency"],
        }
        if "posthoc_localization" in metrics:
            row["posthoc_localization_mae_bins"] = metrics["posthoc_localization"]["mae_bins"]
            row["posthoc_localization_correlation"] = metrics["posthoc_localization"]["pearson_correlation"]
        runs.append(row)
        for fault, recall in metrics["semantic"]["strict_per_class_recall"].items():
            semantic_faults.append({
                "run_id": run_dir.name,
                "approach": metrics["approach"],
                "fold": row["fold"],
                "seed": metrics["seed"],
                "fault": int(fault),
                "strict_recall": recall,
                "strict_prediction_count": metrics["semantic"]["strict_prediction_distribution"][fault],
            })
        for far_name, values in op.items():
            for fault, recall in values["per_fault_recall"].items():
                faults.append({"run_id": run_dir.name, "approach": metrics["approach"], "fold": row["fold"],
                               "seed": metrics["seed"], "far": far_name, "fault": int(fault), "recall": recall})
        for value in metrics["inductive_enrollment"]:
            inductive.append({"run_id": run_dir.name, "approach": metrics["approach"], "fold": row["fold"],
                              "seed": metrics["seed"], **value})
        for value in metrics["sgme_enrollment"]:
            flattened = {key: item for key, item in value.items() if key != "graph"}
            flattened["accepted"] = value["graph"]["accepted_count"]
            sgme.append({"run_id": run_dir.name, "approach": "sgme", "fold": row["fold"], "seed": metrics["seed"], **flattened})
        if metrics["approach"] == "ec":  # identical raw inputs are stored in both neural runs; collect once
            for value in metrics["raw_baselines"]:
                for baseline in ("raw_cosine_1nn", "raw_euclidean_1nn"):
                    raw.append({"run_id": run_dir.name, "approach": baseline, "fold": row["fold"], "seed": metrics["seed"],
                                "draw": value["draw"], "shots": value["shots"], **value[baseline]})

        def add_open_baseline(name: str, values: dict[str, Any], semantic_values: dict[str, Any] | None = None) -> None:
            op_values = values["operating_points"]["far_0.010"]
            open_baselines.append({
                "run_id": run_dir.name, "approach": name, "fold": row["fold"], "seed": metrics["seed"],
                "auroc": values["auroc"], "aupr": values["aupr"], "pauroc_0_01": values["pauroc_0_01"],
                "unknown_recall": op_values["unknown_recall"], "normal_far": op_values["observed_normal_far"],
                "known_acceptance": op_values["known_acceptance"], "worst_fault_recall": op_values["worst_fault_recall"],
                "strict_balanced": (semantic_values or {}).get("strict_zsl_balanced_accuracy", np.nan),
            })

        baseline_values = metrics["open_set_baselines"]
        for baseline in ("energy", "recipe_distance", "openmax_evt"):
            add_open_baseline(f"{metrics['approach']}_{baseline}", baseline_values[baseline])
        add_open_baseline(
            f"{metrics['approach']}_deterministic_physics",
            baseline_values["deterministic_physics"]["zero_day"],
            baseline_values["deterministic_physics"]["semantic"],
        )
        if metrics["approach"] == "ec":
            closed = baseline_values["strongest_closed_set_encoder"]
            add_open_baseline("closed_encoder_energy", closed["energy"])
            add_open_baseline("closed_encoder_openmax_evt", closed["openmax_evt"])
    if validation_failures:
        atomic_json(study_root / "tables" / "analysis_validation_failures.json", validation_failures)
        raise RuntimeError(f"{len(validation_failures)} run artifacts failed hash validation.")
    if len(runs) != 210:
        raise RuntimeError(f"Full analysis requires exactly 210 neural runs; found {len(runs)}.")
    frames = {"runs": pd.DataFrame(runs), "inductive": pd.DataFrame(inductive), "sgme": pd.DataFrame(sgme),
              "raw": pd.DataFrame(raw), "faults": pd.DataFrame(faults),
              "semantic_faults": pd.DataFrame(semantic_faults),
              "open_baselines": pd.DataFrame(open_baselines)}
    for name, frame in frames.items():
        frame.to_csv(study_root / "tables" / f"{name}.csv", index=False)
    return frames


def hierarchical_bootstrap(frame: pd.DataFrame, value: str, *, draws: int = 2000, seed: int = 20260720) -> dict[str, float]:
    values = frame[["fold", "seed", value] + (["draw"] if "draw" in frame else [])].dropna()
    if values.empty:
        return {"mean": np.nan, "median": np.nan, "std": np.nan, "ci_low": np.nan, "ci_high": np.nan, "minimum": np.nan}
    pair_summary = values.groupby("fold")[value].mean()
    rng = np.random.default_rng(seed)
    pairs = values["fold"].unique()
    nested = {
        pair: {seed_value: seed_part[value].to_numpy(dtype=float)
               for seed_value, seed_part in pair_part.groupby("seed")}
        for pair, pair_part in values.groupby("fold")
    }
    estimates = []
    for _ in range(draws):
        selected_pairs = rng.choice(pairs, size=len(pairs), replace=True)
        pair_values = []
        for pair in selected_pairs:
            seeds = np.asarray(list(nested[pair]))
            selected_seeds = rng.choice(seeds, size=len(seeds), replace=True)
            seed_values = []
            for selected_seed in selected_seeds:
                seed_array = nested[pair][selected_seed]
                if len(seed_array) > 1:
                    sampled = rng.choice(seed_array, size=len(seed_array), replace=True)
                    seed_values.append(float(sampled.mean()))
                else:
                    seed_values.append(float(seed_array.mean()))
            pair_values.append(float(np.mean(seed_values)))
        estimates.append(float(np.mean(pair_values)))
    return {
        "mean": float(pair_summary.mean()), "median": float(pair_summary.median()),
        "std": float(pair_summary.std(ddof=1)), "ci_low": float(np.quantile(estimates, 0.025)),
        "ci_high": float(np.quantile(estimates, 0.975)), "minimum": float(pair_summary.min()),
    }


def aggregate_tables(frames: dict[str, pd.DataFrame], study_root: Path) -> pd.DataFrame:
    rows = []
    run_metrics = [
        "far_0.010_unknown_recall", "far_0.010_observed_normal_far", "far_0.010_known_acceptance",
        "far_0.010_worst_fault_recall", "far_0.020_unknown_recall", "far_0.050_unknown_recall",
        "pauroc_0_01", "pauroc_0_05", "auroc", "aupr", "oscr", "fpr_at_95_unknown_tpr",
        "unknown_false_acceptance_at_95_known_acceptance", "strict_balanced", "gzsl_seen", "gzsl_unseen",
        "gzsl_h", "class_collapse",
    ]
    for approach, part in frames["runs"].groupby("approach"):
        for metric in run_metrics:
            rows.append({"task": "outer", "approach": approach, "setting": "all_seeds", "metric": metric,
                         **hierarchical_bootstrap(part, metric)})
            subset = part[part["seed"].isin([42, 123, 2026])]
            rows.append({"task": "outer", "approach": approach, "setting": "previous_seed_subset", "metric": metric,
                         **hierarchical_bootstrap(subset, metric)})
    for source, task in (("inductive", "inductive"), ("sgme", "semi_supervised"), ("raw", "raw_1nn")):
        frame = frames[source]
        if frame.empty:
            continue
        settings = ["shots"] + (["buffer_per_class"] if "buffer_per_class" in frame else [])
        for key, part in frame.groupby(["approach", *settings]):
            key = key if isinstance(key, tuple) else (key,)
            base_setting = ",".join(f"{name}={value}" for name, value in zip(settings, key[1:], strict=True))
            for cohort, cohort_part in (
                ("all_seeds", part),
                ("previous_seed_subset", part[part["seed"].isin([42, 123, 2026])]),
            ):
                metrics = ["seen_accuracy", "unseen_accuracy", "harmonic_mean", "balanced_accuracy", "rejection_rate"]
                if source == "sgme":
                    metrics.extend(["coverage", "selective_risk"])
                for metric in metrics:
                    if metric not in cohort_part:
                        continue
                    rows.append({"task": task, "approach": key[0],
                                 "setting": f"{cohort},{base_setting}", "metric": metric,
                                 **hierarchical_bootstrap(cohort_part, metric)})
    for approach, part in frames["open_baselines"].groupby("approach"):
        for cohort, cohort_part in (
            ("all_seeds", part),
            ("previous_seed_subset", part[part["seed"].isin([42, 123, 2026])]),
        ):
            for metric in ("unknown_recall", "normal_far", "known_acceptance", "worst_fault_recall",
                           "pauroc_0_01", "auroc", "aupr", "strict_balanced"):
                rows.append({"task": "open_set_baseline", "approach": approach, "setting": cohort, "metric": metric,
                             **hierarchical_bootstrap(cohort_part, metric)})
    aggregate = pd.DataFrame(rows)
    aggregate.to_csv(study_root / "tables" / "aggregate_summary.csv", index=False)
    runs = frames["runs"]
    runs.groupby(["approach", "seed"], as_index=False).agg(
        unknown_recall_mean=("far_0.010_unknown_recall", "mean"),
        unknown_recall_std=("far_0.010_unknown_recall", "std"),
        strict_zsl_mean=("strict_balanced", "mean"), gzsl_h_mean=("gzsl_h", "mean"),
    ).to_csv(study_root / "tables" / "seed_sensitivity.csv", index=False)
    if not frames["inductive"].empty:
        frames["inductive"].groupby(["approach", "shots", "draw"], as_index=False).agg(
            harmonic_mean=("harmonic_mean", "mean"), unseen_accuracy=("unseen_accuracy", "mean")
        ).to_csv(study_root / "tables" / "support_draw_sensitivity.csv", index=False)
    runs[["approach", "fold", "seed", "training_seconds", "inference_seconds", "inference_ms_per_trace",
          "enrollment_seconds", "graph_seconds", "graph_update_ms_mean", "parameters", "checkpoint_bytes",
          "prediction_bytes", "enrollment_prediction_bytes", "enrollment_group_manifest_bytes",
          "peak_allocated_bytes"]].to_csv(study_root / "tables" / "efficiency.csv", index=False)
    calibration_columns = ["approach", "fold", "seed"] + [column for column in runs if "observed_normal_far" in column or "known_acceptance" in column]
    runs[calibration_columns].to_csv(study_root / "tables" / "calibration.csv", index=False)
    per_fault_rows = []
    for (approach, far, fault), part in frames["faults"].groupby(["approach", "far", "fault"]):
        per_fault_rows.append({"approach": approach, "far": far, "fault": fault,
                               **hierarchical_bootstrap(part, "recall")})
    pd.DataFrame(per_fault_rows).to_csv(study_root / "tables" / "per_fault_summary.csv", index=False)
    semantic_rows = []
    for (approach, fault), part in frames["semantic_faults"].groupby(["approach", "fault"]):
        semantic_rows.append({"approach": approach, "fault": fault,
                              **hierarchical_bootstrap(part, "strict_recall")})
    pd.DataFrame(semantic_rows).to_csv(study_root / "tables" / "strict_semantic_per_fault_summary.csv", index=False)
    return aggregate


def _rate_difference_in_samples(difference: float, denominator: int) -> float:
    if denominator <= 0:
        raise ValueError("Rate reconstruction denominator must be positive.")
    return float(difference) * denominator


def reconstruct_saved_metrics(study_root: Path) -> dict[str, Any]:
    rows, failures = [], []
    for predictions_path in sorted((study_root / "full_benchmark").rglob("predictions.npz")):
        metrics = json.loads((predictions_path.parent / "metrics.json").read_text(encoding="utf-8"))
        arrays = np.load(predictions_path)
        labels, score, logits = arrays["labels"].astype(int), arrays["novelty_score"], arrays["logits"].astype(float)
        holdout = tuple(metrics["holdout"])
        unknown = np.isin(labels, holdout)
        reconstructed = {
            "auroc": float(roc_auc_score(unknown, score)),
            "aupr": float(average_precision_score(unknown, score)),
            "strict": semantic_metrics(logits, labels, holdout)["strict_zsl_balanced_accuracy"],
        }
        expected = {"auroc": metrics["zero_day"]["auroc"], "aupr": metrics["zero_day"]["aupr"],
                    "strict": metrics["semantic"]["strict_zsl_balanced_accuracy"]}
        differences = {key: abs(reconstructed[key] - expected[key]) for key in expected}
        validation_normal = arrays["validation_labels"] == 0
        calibrator = NormalOnlyCalibrator(metrics["zero_day"]["calibration"]["mode"]).fit(
            arrays["validation_novelty_score"][validation_normal], arrays["validation_snr"][validation_normal]
        )
        operating_differences = {}
        operating_sample_differences = {}
        rate_denominators = {
            "observed_normal_far": int((labels == 0).sum()),
            "unknown_recall": int(unknown.sum()),
            "known_acceptance": int((~unknown).sum()),
        }
        for far in (0.01, 0.02, 0.05):
            threshold = calibrator.threshold(arrays["snr"], far)
            rejected = score > threshold
            name = f"far_{far:.3f}"
            actual = metrics["zero_day"]["operating_points"][name]
            reconstructed_op = {
                "observed_normal_far": float(rejected[labels == 0].mean()),
                "unknown_recall": float(rejected[unknown].mean()),
                "known_acceptance": float((~rejected[~unknown]).mean()),
            }
            operating_differences[name] = {key: abs(reconstructed_op[key] - actual[key]) for key in reconstructed_op}
            operating_sample_differences[name] = {
                key: _rate_difference_in_samples(operating_differences[name][key], rate_denominators[key])
                for key in reconstructed_op
            }
        max_operating_sample_difference = max(
            value for row in operating_sample_differences.values() for value in row.values()
        )
        enrollment = np.load(predictions_path.parent / "enrollment_predictions.npz")
        enrollment_labels = enrollment["labels"].astype(int)
        seen_ids = sorted(set(range(8)) - set(holdout))
        inductive_index = json.loads((predictions_path.parent / "inductive_predictions_index.json").read_text(encoding="utf-8"))
        graph_index = json.loads((predictions_path.parent / "enrollment_predictions_index.json").read_text(encoding="utf-8"))
        expected_inductive = {(row["draw"], row["shots"]): row for row in metrics["inductive_enrollment"]}
        expected_graph = {(row["draw"], row["shots"], row["buffer_per_class"]): row for row in metrics["sgme_enrollment"]}
        enrollment_max_difference = 0.0
        for index, setting in enumerate(inductive_index):
            actual = post_enrollment_metrics(
                enrollment_labels, enrollment["inductive_predicted"][index].astype(int),
                seen_ids=seen_ids, unseen_ids=holdout,
            )
            expected_row = expected_inductive[(setting["draw"], setting["shots"])]
            enrollment_max_difference = max(enrollment_max_difference, *(
                abs(float(actual[key]) - float(expected_row[key]))
                for key in ("accuracy", "balanced_accuracy", "seen_accuracy", "unseen_accuracy", "harmonic_mean", "rejection_rate")
            ))
        for index, setting in enumerate(graph_index):
            actual = post_enrollment_metrics(
                enrollment_labels, enrollment["sgme_predicted"][index].astype(int),
                seen_ids=seen_ids, unseen_ids=holdout,
            )
            expected_row = expected_graph[(setting["draw"], setting["shots"], setting["buffer_per_class"])]
            enrollment_max_difference = max(enrollment_max_difference, *(
                abs(float(actual[key]) - float(expected_row[key]))
                for key in ("accuracy", "balanced_accuracy", "seen_accuracy", "unseen_accuracy", "harmonic_mean", "rejection_rate")
            ))
        ok = (differences["auroc"] < 1e-6 and differences["aupr"] < 1e-6 and
              differences["strict"] < 1e-3 and max_operating_sample_difference <= 1.0000001 and
              enrollment_max_difference < 1e-8)
        row = {"run_id": predictions_path.parent.name, "ok": ok, "differences": differences}
        row["operating_differences"] = operating_differences
        row["operating_sample_differences"] = operating_sample_differences
        row["operating_max_sample_difference"] = max_operating_sample_difference
        row["enrollment_max_difference"] = enrollment_max_difference
        arrays.close()
        enrollment.close()
        rows.append(row)
        if not ok:
            failures.append(row)
    result = {
        "runs": len(rows), "passed": len(rows) - len(failures), "failures": failures,
        "runs_using_one_boundary_sample_tolerance": sum(
            row["operating_max_sample_difference"] > 1e-7 for row in rows
        ),
        "note": (
            "Strict tolerance is looser because logits are intentionally stored as float16. "
            "Operating rates may differ by at most one boundary sample because novelty scores are stored as float32."
        ),
    }
    atomic_json(study_root / "tables" / "independent_reconstruction.json", result)
    if failures:
        raise RuntimeError(f"Independent reconstruction failed for {len(failures)} runs.")
    return result


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values, dtype=float)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def paired_previous_comparisons(frames: dict[str, pd.DataFrame], study_root: Path) -> pd.DataFrame:
    previous_path = study_root.parent / "otdr_three_approach_study" / "tables" / "per_run_results.csv"
    previous = pd.read_csv(previous_path)
    new = frames["runs"]
    comparisons = []
    specs = [
        ("unknown_recall", "far_0.010_unknown_recall", "pre_unknown_recall", ["a", "c"]),
        ("strict_zsl", "strict_balanced", "strict_balanced", ["b"]),
        ("gzsl_h", "gzsl_h", "gzsl_h", ["b"]),
    ]
    for new_approach in new["approach"].unique():
        for metric, new_col, old_col, old_approaches in specs:
            left = new[(new["approach"] == new_approach) & new["seed"].isin([42, 123, 2026])]
            for old_approach in old_approaches:
                right = previous[(previous["approach"] == old_approach) & previous[old_col].notna()]
                merged = left[["fold", "seed", new_col]].rename(columns={new_col: "new_value"}).merge(
                    right[["fold", "seed", old_col]].rename(columns={old_col: "previous_value"}),
                    on=["fold", "seed"],
                )
                pair = merged.groupby("fold")[["new_value", "previous_value"]].mean().dropna()
                if len(pair) < 5:
                    continue
                difference = pair["new_value"] - pair["previous_value"]
                rng = np.random.default_rng(20260720 + len(comparisons))
                bootstrap_difference = [float(rng.choice(difference.to_numpy(), size=len(difference), replace=True).mean())
                                        for _ in range(5000)]
                try:
                    p_value = float(wilcoxon(difference).pvalue)
                except ValueError:
                    p_value = 1.0
                comparisons.append({"new": new_approach, "previous": old_approach, "metric": metric,
                                    "pairs": len(pair), "new_mean": float(pair["new_value"].mean()),
                                    "previous_mean": float(pair["previous_value"].mean()), "mean_difference": float(difference.mean()),
                                    "difference_ci_low": float(np.quantile(bootstrap_difference, 0.025)),
                                    "difference_ci_high": float(np.quantile(bootstrap_difference, 0.975)),
                                    "median_difference": float(difference.median()), "standardized_effect": float(difference.mean() / (difference.std(ddof=1) or 1)),
                                    "p_value": p_value})
    enrollment_sources = []
    inductive = frames["inductive"]
    if not inductive.empty:
        enrollment_sources.extend((approach, "inductive_1shot_h", part[part["shots"] == 1])
                                  for approach, part in inductive.groupby("approach"))
    sgme = frames["sgme"]
    if not sgme.empty:
        enrollment_sources.append(("sgme", "semi_supervised_1shot_buffer512_h",
                                   sgme[(sgme["shots"] == 1) & (sgme["buffer_per_class"] == 512)]))
    for new_approach, metric, source in enrollment_sources:
        left = source[source["seed"].isin([42, 123, 2026])].groupby(["fold", "seed"], as_index=False)["harmonic_mean"].mean()
        for old_approach in ("a", "b", "c"):
            right = previous[(previous["approach"] == old_approach) & previous["post_h"].notna()]
            merged = left.merge(right[["fold", "seed", "post_h"]], on=["fold", "seed"])
            pair = merged.groupby("fold")[["harmonic_mean", "post_h"]].mean().dropna()
            if len(pair) < 5:
                continue
            difference = pair["harmonic_mean"] - pair["post_h"]
            rng = np.random.default_rng(20260720 + len(comparisons))
            bootstrap_difference = [float(rng.choice(difference.to_numpy(), size=len(difference), replace=True).mean())
                                    for _ in range(5000)]
            try:
                p_value = float(wilcoxon(difference).pvalue)
            except ValueError:
                p_value = 1.0
            comparisons.append({"new": new_approach, "previous": old_approach, "metric": metric,
                                "pairs": len(pair), "new_mean": float(pair["harmonic_mean"].mean()),
                                "previous_mean": float(pair["post_h"].mean()), "mean_difference": float(difference.mean()),
                                "difference_ci_low": float(np.quantile(bootstrap_difference, 0.025)),
                                "difference_ci_high": float(np.quantile(bootstrap_difference, 0.975)),
                                "median_difference": float(difference.median()),
                                "standardized_effect": float(difference.mean() / (difference.std(ddof=1) or 1)),
                                "p_value": p_value})
    result = pd.DataFrame(comparisons)
    if not result.empty:
        result["p_holm"] = holm_adjust(result["p_value"].to_numpy())
    result.to_csv(study_root / "tables" / "paired_previous_comparisons.csv", index=False)
    return result


def collect_ablation_summary(study_root: Path) -> pd.DataFrame:
    rows = []
    pattern = re.compile(r"^(ec|pc2)-(.+)-(\d_\d)-[0-9a-f]+$")
    for path in (study_root / "stress" / "ablations").rglob("metrics.json"):
        match = pattern.match(path.parent.name)
        if not match:
            continue
        metrics = json.loads(path.read_text(encoding="utf-8"))
        inductive = [row["harmonic_mean"] for row in metrics["inductive_enrollment"] if row["shots"] == 1]
        graph = [row["harmonic_mean"] for row in metrics["sgme_enrollment"] if row["buffer_per_class"] == 128]
        rows.append({"approach": match.group(1), "ablation": match.group(2), "fold": match.group(3).replace("_", "-"),
                     "unknown_recall": metrics["zero_day"]["operating_points"]["far_0.010"]["unknown_recall"],
                     "normal_far": metrics["zero_day"]["operating_points"]["far_0.010"]["observed_normal_far"],
                     "strict_zsl": metrics["semantic"]["strict_zsl_balanced_accuracy"],
                     "inductive_1shot_h": float(np.mean(inductive)) if inductive else np.nan,
                     "sgme_1shot_buffer128_h": float(np.mean(graph)) if graph else np.nan})
    reference_pairs = {"1-2", "3-5", "6-7"}
    for path in (study_root / "full_benchmark").rglob("metrics.json"):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        fold = "-".join(str(value) for value in metrics["holdout"])
        if metrics["seed"] != 42 or fold not in reference_pairs:
            continue
        inductive = [row["harmonic_mean"] for row in metrics["inductive_enrollment"] if row["shots"] == 1]
        graph = [row["harmonic_mean"] for row in metrics["sgme_enrollment"]
                 if row["shots"] == 1 and row["buffer_per_class"] == 128]
        rows.append({
            "approach": metrics["approach"], "ablation": "finalist_reference", "fold": fold,
            "unknown_recall": metrics["zero_day"]["operating_points"]["far_0.010"]["unknown_recall"],
            "normal_far": metrics["zero_day"]["operating_points"]["far_0.010"]["observed_normal_far"],
            "strict_zsl": metrics["semantic"]["strict_zsl_balanced_accuracy"],
            "inductive_1shot_h": float(np.mean(inductive)) if inductive else np.nan,
            "sgme_1shot_buffer128_h": float(np.mean(graph)) if graph else np.nan,
        })
    raw = pd.DataFrame(rows)
    if raw.empty:
        return raw
    raw.to_csv(study_root / "tables" / "ablation_runs.csv", index=False)
    summary = raw.groupby(["approach", "ablation"], as_index=False).mean(numeric_only=True)
    summary.to_csv(study_root / "tables" / "ablation_summary.csv", index=False)
    return summary


def _save(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def make_core_plots(frames: dict[str, pd.DataFrame], study_root: Path) -> list[str]:
    plots = study_root / "plots"
    plots.mkdir(exist_ok=True)
    created = []
    runs, inductive, sgme, faults = frames["runs"], frames["inductive"], frames["sgme"], frames["faults"]
    fig, ax = plt.subplots(figsize=(7, 4))
    for approach, part in runs.groupby("approach"):
        x, y = [], []
        for far in (0.01, 0.02, 0.05):
            x.append(part[f"far_{far:.3f}_observed_normal_far"].mean())
            y.append(part[f"far_{far:.3f}_unknown_recall"].mean())
        ax.plot(x, y, marker="o", label=approach.upper())
    ax.set(xlabel="Observed normal FAR", ylabel="Unknown recall", title="Normal-FAR / unknown-recall tradeoff")
    ax.legend(); path = plots / "far_unknown_tradeoff.png"; _save(fig, path); created.append(path.name)

    fig, ax = plt.subplots(figsize=(8, 4))
    summary = faults[faults["far"] == "far_0.010"].groupby(["approach", "fault"])["recall"].mean().unstack(0)
    summary.plot(kind="bar", ax=ax)
    ax.set(ylabel="Recall", title="Per-fault recall near 1% normal FAR", ylim=(0, 1)); path = plots / "per_fault_recall.png"; _save(fig, path); created.append(path.name)

    fig, ax = plt.subplots(figsize=(7, 4))
    for approach, part in inductive.groupby("approach"):
        curve = part.groupby("shots")["harmonic_mean"].mean()
        ax.plot(curve.index, curve.values, marker="o", label=approach.upper())
    ax.set(xlabel="Labeled shots per held-out class", ylabel="Harmonic mean", title="Inductive enrollment")
    ax.legend(); path = plots / "shot_curves.png"; _save(fig, path); created.append(path.name)

    if not sgme.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        for shot, part in sgme.groupby("shots"):
            curve = part.groupby("buffer_per_class")["harmonic_mean"].mean()
            ax.plot(curve.index, curve.values, marker="o", label=f"{shot}-shot")
        ax.set(xlabel="Unlabeled buffer per held-out class", ylabel="Harmonic mean", title="SGME buffer curves")
        ax.legend(); path = plots / "buffer_curves.png"; _save(fig, path); created.append(path.name)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    runs.boxplot(column="far_0.010_unknown_recall", by=["approach", "seed"], ax=axes[0], rot=45)
    axes[0].set_title("Seed sensitivity"); axes[0].set_ylabel("Unknown recall")
    if not inductive.empty:
        subset = inductive[inductive["shots"] == 1]
        subset.boxplot(column="harmonic_mean", by="draw", ax=axes[1], rot=90)
    axes[1].set_title("Support-draw sensitivity (1-shot)"); axes[1].set_ylabel("H")
    fig.suptitle(""); path = plots / "seed_support_sensitivity.png"; _save(fig, path); created.append(path.name)

    first = next((study_root / "full_benchmark").rglob("predictions.npz"), None)
    if first is not None:
        arrays = np.load(first)
        metrics = json.loads((first.parent / "metrics.json").read_text(encoding="utf-8"))
        holdout = tuple(metrics["holdout"]); unknown = np.isin(arrays["labels"], holdout)
        fpr, tpr, _ = roc_curve(unknown, arrays["novelty_score"])
        fig, ax = plt.subplots(figsize=(6, 4)); ax.plot(fpr, tpr); ax.set_xlim(0, 0.05); ax.set_ylim(0, 1)
        ax.set(xlabel="Normal/known false-positive rate", ylabel="Unknown recall", title="Low-FAR ROC zoom")
        path = plots / "low_far_roc_zoom.png"; _save(fig, path); created.append(path.name)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(arrays["novelty_score"][~unknown], bins=80, density=True, alpha=.55, label="known")
        ax.hist(arrays["novelty_score"][unknown], bins=80, density=True, alpha=.55, label="held-out")
        ax.set(title="Novelty score distributions", xlabel="score", ylabel="density"); ax.legend()
        path = plots / "score_histograms.png"; _save(fig, path); created.append(path.name)
        unseen = unknown
        strict_local = arrays["logits"][unseen][:, list(holdout)].argmax(1)
        predicted = np.asarray(holdout)[strict_local]
        cm = confusion_matrix(arrays["labels"][unseen], predicted, labels=list(holdout))
        fig, ax = plt.subplots(figsize=(5, 4)); image = ax.imshow(cm, cmap="Blues")
        ax.set(xticks=[0, 1], yticks=[0, 1], xticklabels=holdout, yticklabels=holdout,
               xlabel="Predicted", ylabel="True", title="Strict-ZSL confusion (representative run)")
        fig.colorbar(image, ax=ax); path = plots / "strict_zsl_confusion.png"; _save(fig, path); created.append(path.name)
        embedding = arrays["example_embedding"].astype(float); emb_labels = arrays["example_embedding_labels"]
        boundary_embedding = arrays["boundary_embedding"].astype(float)
        all_embedding = np.vstack([embedding, boundary_embedding])
        all_coordinates = PCA(n_components=2).fit_transform(all_embedding)
        coordinates, boundary_coordinates = all_coordinates[:len(embedding)], all_coordinates[len(embedding):]
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter = ax.scatter(coordinates[:, 0], coordinates[:, 1], c=emb_labels, cmap="tab10", s=7, alpha=.6)
        ax.scatter(boundary_coordinates[:, 0], boundary_coordinates[:, 1], c="black", marker="x", s=12, alpha=.45, label="synthetic boundary")
        ax.set(title="Frozen embedding with synthetic boundary outliers"); ax.legend(); fig.colorbar(scatter, ax=ax, label="class")
        path = plots / "embedding_projection.png"; _save(fig, path); created.append(path.name)
        factors = arrays["factor_mean"].astype(float)
        fig, ax = plt.subplots(figsize=(9, 4)); ax.boxplot([factors[:, index] for index in range(factors.shape[1])], showfliers=False)
        ax.set(xlabel="Factor index", ylabel="Predicted value", title="Learned event-factor distributions")
        path = plots / "factor_distributions.png"; _save(fig, path); created.append(path.name)
        features = arrays["example_features"]
        centers = arrays["example_event_center"]
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
        for index, ax in enumerate(axes.flat):
            ax.plot(features[index, 1:], color="#2a6fbb"); ax.axvline(float(centers[index]), color="#d1495b", linestyle="--")
            ax.set_title(f"label {int(arrays['example_labels'][index])}")
        fig.suptitle("Event alignment examples (red: soft center)")
        path = plots / "event_alignment_examples.png"; _save(fig, path); created.append(path.name)
        fig, axes = plt.subplots(2, 4, figsize=(12, 6), sharex=True)
        for index in range(4):
            axes[0, index].plot(features[index, 1:], color="#2a6fbb")
            axes[0, index].set_title(f"real class {int(arrays['example_labels'][index])}")
            axes[1, index].plot(arrays["boundary_features"][index, 1:], color="#d1495b")
            axes[1, index].set_title("synthetic boundary")
        fig.suptitle("Real versus frozen-renderer traces")
        path = plots / "real_vs_synthetic_traces.png"; _save(fig, path); created.append(path.name)
        validation_normal = arrays["validation_labels"] == 0
        calibration_mode = metrics["zero_day"]["calibration"]["mode"]
        calibrator = NormalOnlyCalibrator(calibration_mode).fit(
            arrays["validation_novelty_score"][validation_normal], arrays["validation_snr"][validation_normal]
        )
        thresholds = calibrator.threshold(arrays["snr"], 0.01)
        rejected = arrays["novelty_score"] > thresholds
        edges = np.unique(np.quantile(arrays["snr"], np.linspace(0, 1, 6)))
        assignment = np.clip(np.digitize(arrays["snr"], edges[1:-1]), 0, len(edges) - 2)
        centers_bin, far_bin, recall_bin = [], [], []
        for index in range(len(edges) - 1):
            mask = assignment == index; normal_mask = mask & (arrays["labels"] == 0); unknown_mask = mask & unknown
            centers_bin.append(float(np.median(arrays["snr"][mask])))
            far_bin.append(float(rejected[normal_mask].mean()) if normal_mask.any() else np.nan)
            recall_bin.append(float(rejected[unknown_mask].mean()) if unknown_mask.any() else np.nan)
        fig, ax = plt.subplots(figsize=(7, 4)); ax.plot(centers_bin, far_bin, marker="o", label="normal FAR")
        ax.plot(centers_bin, recall_bin, marker="s", label="unknown recall"); ax.axhline(.01, color="gray", linestyle="--", label="1% target")
        ax.set(xlabel="Standardized SNR bin center", ylabel="Rate", title="SNR-conditional calibration transfer"); ax.legend()
        path = plots / "snr_conditional_calibration.png"; _save(fig, path); created.append(path.name)
    recipes_path = study_root / "configs" / "event_recipes.json"
    if recipes_path.exists():
        recipes = json.loads(recipes_path.read_text(encoding="utf-8"))
        matrix = np.asarray([row["mean"] for row in recipes["classes"]])
        fig, ax = plt.subplots(figsize=(12, 5)); image = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="viridis")
        ax.set(yticks=np.arange(8), yticklabels=[row["name"] for row in recipes["classes"]],
               xticks=np.arange(len(recipes["factor_names"])), xticklabels=recipes["factor_names"],
               title="Frozen probabilistic class-recipe means")
        ax.tick_params(axis="x", rotation=60); fig.colorbar(image, ax=ax, label="factor mean")
        path = plots / "class_recipe_diagram.png"; _save(fig, path); created.append(path.name)
    ablation_path = study_root / "tables" / "ablation_summary.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path).copy()
        ablation["label"] = ablation["approach"].str.upper() + ": " + ablation["ablation"]
        ablation = ablation.sort_values(["approach", "ablation"])
        fig, axes = plt.subplots(1, 3, figsize=(18, max(6, len(ablation) * .28)), sharey=True)
        post_value = ablation["sgme_1shot_buffer128_h"].where(
            ablation["ablation"].str.startswith("sgme_") | (ablation["ablation"] == "finalist_reference"),
            ablation["inductive_1shot_h"],
        )
        for ax, values, title in (
            (axes[0], ablation["unknown_recall"], "Unknown recall near 1% FAR"),
            (axes[1], ablation["strict_zsl"], "Strict-ZSL balanced accuracy"),
            (axes[2], post_value, "1-shot H (SGME@128 where applicable)"),
        ):
            ax.barh(ablation["label"], values)
            ax.set(xlabel=title)
        fig.suptitle("Predeclared ablations versus finalist reference")
        path = plots / "ablation_results.png"; _save(fig, path); created.append(path.name)
    return created


def analyze_event_study(study_root: Path) -> dict[str, Any]:
    frames = collect_results(study_root)
    aggregate = aggregate_tables(frames, study_root)
    comparisons = paired_previous_comparisons(frames, study_root)
    reconstruction = reconstruct_saved_metrics(study_root)
    ablations = collect_ablation_summary(study_root)
    plots = make_core_plots(frames, study_root)
    result = {"runs": len(frames["runs"]), "aggregate_rows": len(aggregate),
              "paired_comparisons": len(comparisons), "ablation_rows": len(ablations),
              "reconstruction": reconstruction, "plots": plots,
              "dependence_warning": "Pairs are the top-level clusters; seeds and support draws are nested and not independent."}
    atomic_json(study_root / "tables" / "analysis_summary.json", result)
    return result
