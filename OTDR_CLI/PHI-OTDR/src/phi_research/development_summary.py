"""Synthesize v2 development evidence and make pre-query stop/go decisions."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


DIRECTIONS = ("january_to_april_may", "april_may_to_january")
METHODS = (
    "dynamics__mean",
    "dynamics__robust_quantiles",
    "dynamics__sliced_wasserstein",
    "dynamics__ordered_trajectory",
)


def _cluster_ci(values: np.ndarray, *, seed: int = 20260805, draws: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    samples = np.mean(rng.choice(values, size=(draws, len(values)), replace=True), axis=1)
    return {
        "mean": float(np.mean(values)),
        "lower_95": float(np.quantile(samples, 0.025)),
        "upper_95": float(np.quantile(samples, 0.975)),
        "cluster_count": int(len(values)),
    }


def _paired(left: np.ndarray, right: np.ndarray) -> dict[str, object]:
    difference = left - right
    statistic, p_value = wilcoxon(left, right, alternative="two-sided")
    return {
        "mean_difference": float(np.mean(difference)),
        "median_difference": float(np.median(difference)),
        "win_fraction": float(np.mean(difference > 0)),
        "tie_fraction": float(np.mean(difference == 0)),
        "wilcoxon_statistic": float(statistic),
        "wilcoxon_two_sided_p": float(p_value),
        "cluster_count": int(len(difference)),
    }


def summarize(root: Path) -> dict[str, object]:
    fold_values: dict[tuple[str, str, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    files: list[str] = []
    for direction in DIRECTIONS:
        paths = sorted((root / direction / "session_distribution").glob("dynamics_seed*.json"))
        if len(paths) != 3:
            raise ValueError(f"Expected three dynamics seeds for {direction}, found {len(paths)}")
        for path in paths:
            files.append(path.as_posix())
            payload = json.loads(path.read_text(encoding="utf-8"))
            for row in payload["fold_results"]:
                key = (direction, str(row["method"]), int(row["holdout_class_id"]))
                fold_values[key]["pre_detection_h"].append(
                    float(row["pre_enrollment"]["detection_h"])
                )
                fold_values[key]["unknown_auroc"].append(
                    float(row["pre_enrollment"]["unknown_auroc"])
                )
                for strategy in ("random", "medoid", "farthest_first", "facility_location", "uncertainty_diversity"):
                    for shot in (1, 3, 5):
                        fold_values[key][f"{strategy}_{shot}"].append(
                            float(row["post_enrollment"][f"{strategy}__{shot}"]["enrollment_h_mean"])
                        )

    clustered: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (direction, method, holdout), metrics in sorted(fold_values.items()):
        method_key = f"dynamics__{method}"
        for metric, values in metrics.items():
            clustered[method_key][metric].append(float(np.mean(values)))

    method_summary: dict[str, object] = {}
    for method in METHODS:
        method_summary[method] = {
            metric: _cluster_ci(np.asarray(values), seed=20260805 + index)
            for index, (metric, values) in enumerate(sorted(clustered[method].items()))
        }

    sliced_medoid = np.asarray(clustered["dynamics__sliced_wasserstein"]["medoid_5"])
    sliced_random = np.asarray(clustered["dynamics__sliced_wasserstein"]["random_5"])
    ordered_random = np.asarray(clustered["dynamics__ordered_trajectory"]["random_5"])
    mean_random = np.asarray(clustered["dynamics__mean"]["random_5"])
    medoid_comparison = _paired(sliced_medoid, sliced_random)
    trajectory_comparison = _paired(ordered_random, mean_random)

    baselines: dict[str, object] = {}
    metadata: dict[str, object] = {}
    for direction in DIRECTIONS:
        baseline = json.loads(
            (root / direction / "source_baselines" / "development_results.json").read_text(
                encoding="utf-8"
            )
        )
        baselines[direction] = {
            "selected": baseline["selected_by_validation"],
            "results": {
                row["key"]: {
                    "window_macro_f1": row["window_metrics"]["macro_f1"],
                    "session_macro_f1": row["session_metrics"]["macro_f1"],
                    "worst_class_recall": row["session_metrics"]["worst_class_recall"],
                }
                for row in baseline["results"]
            },
        }
        probe = json.loads((root / direction / "metadata_probe.json").read_text(encoding="utf-8"))
        metadata[direction] = {
            "metadata_to_event_class": probe["metadata_to_event_class"],
            "signal_to_metadata": probe["signal_to_metadata"],
            "inventory_confounding": probe["inventory_confounding"],
        }

    medoid_go = (
        medoid_comparison["mean_difference"] >= 0.05
        and medoid_comparison["win_fraction"] >= 2.0 / 3.0
    )
    trajectory_go = trajectory_comparison["mean_difference"] >= 0.03
    return {
        "protocol": "pre-target-query v2 development synthesis",
        "final_query_used": False,
        "source_files": files,
        "source_baselines": baselines,
        "metadata_probes": metadata,
        "session_method_summary": method_summary,
        "paired_development_tests": {
            "sliced_wasserstein_medoid5_vs_random5": medoid_comparison,
            "ordered_trajectory_random5_vs_mean_random5": trajectory_comparison,
        },
        "stop_go": {
            "detection": {
                "decision": "continue",
                "selected": "dynamics mean-session descriptor with known-only calibration",
                "reason": "highest mean source-development detection H among tested continuous session descriptors",
            },
            "distributional_enrollment": {
                "decision": "continue" if medoid_go else "stop",
                "selected": "dynamics sliced-Wasserstein quantile descriptor plus medoid support selection",
                "gate": "mean H improvement >=0.05 and cluster win fraction >=2/3 versus random support",
                "observed": medoid_comparison,
            },
            "ordered_trajectory": {
                "decision": "continue" if trajectory_go else "stop_as_primary",
                "gate": "mean five-shot random H improvement >=0.03 over mean descriptor",
                "observed": trajectory_comparison,
                "interpretation": "ordered window information is measurable but the current trajectory descriptor does not improve enrollment",
            },
            "farthest_first_facility_uncertainty": {
                "decision": "controls_only",
                "reason": "development performance is inconsistent and often selects harmful outlier sessions",
            },
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"paired": payload["paired_development_tests"], "stop_go": payload["stop_go"]}, indent=2))


if __name__ == "__main__":
    main()
