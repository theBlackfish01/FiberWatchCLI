"""Aggregate confirmatory acquisition-era evidence without model selection."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon


def _cluster_means(rows: list[dict[str, object]], method: str, getter) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["method"] == method:
            grouped[f"{row['direction']}|{row['holdout_class']}"] .append(float(getter(row)))
    return {key: float(np.mean(values)) for key, values in sorted(grouped.items())}


def _bootstrap(values: list[float], seed: int = 20260805) -> dict[str, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    samples = np.mean(rng.choice(array, size=(10000, len(array)), replace=True), axis=1)
    return {"mean": float(array.mean()), "ci95_low": float(np.quantile(samples, .025)),
            "ci95_high": float(np.quantile(samples, .975)), "minimum": float(array.min()),
            "clusters": int(len(array))}


def _paired(primary: dict[str, float], control: dict[str, float]) -> dict[str, object]:
    keys = sorted(set(primary) & set(control))
    differences = np.asarray([primary[key] - control[key] for key in keys])
    nonzero = differences[differences != 0]
    if len(nonzero):
        test = wilcoxon(differences, alternative="two-sided", zero_method="wilcox")
        statistic, pvalue = float(test.statistic), float(test.pvalue)
    else:
        statistic, pvalue = 0.0, 1.0
    return {"unit": "direction x held-out-class after seed averaging", "clusters": len(keys),
            "mean_difference": float(differences.mean()), "median_difference": float(np.median(differences)),
            "win_fraction": float(np.mean(differences > 0)), "tie_fraction": float(np.mean(differences == 0)),
            "wilcoxon_statistic": statistic, "wilcoxon_pvalue": pvalue,
            "differences": {key: float(value) for key, value in zip(keys, differences)}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--january-to-april-may", type=Path, required=True)
    parser.add_argument("--april-may-to-january", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for direction, path in (("january_to_april_may", args.january_to_april_may),
                            ("april_may_to_january", args.april_may_to_january)):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            rows.append({**row, "direction": direction})
    detection = lambda row: row["detection"]["target_calibration_primary"]["detection_h"]
    auroc = lambda row: row["detection"]["target_calibration_primary"]["unknown_auroc"]
    enroll = lambda shot, strategy: lambda row: row["enrollment"][str(shot)][strategy]["mean"]["enrollment_h"]
    methods = ("pca24", "aligned_weight_0", "aligned_weight_100")
    estimates: dict[str, object] = {}
    for method in methods:
        estimates[method] = {
            "detection_h": _bootstrap(list(_cluster_means(rows, method, detection).values())),
            "unknown_auroc": _bootstrap(list(_cluster_means(rows, method, auroc).values())),
            **{f"{shot}shot_{strategy}": _bootstrap(list(_cluster_means(rows, method, enroll(shot, strategy)).values()))
               for shot in (1, 3, 5) for strategy in ("medoid", "random")},
        }
    tests = {
        "alignment100_vs_unaligned_detection_h": _paired(
            _cluster_means(rows, "aligned_weight_100", detection),
            _cluster_means(rows, "aligned_weight_0", detection)),
        "alignment100_vs_unaligned_5shot_medoid": _paired(
            _cluster_means(rows, "aligned_weight_100", enroll(5, "medoid")),
            _cluster_means(rows, "aligned_weight_0", enroll(5, "medoid"))),
    }
    for method in methods:
        for shot in (1, 3, 5):
            tests[f"{method}_{shot}shot_medoid_vs_random"] = _paired(
                _cluster_means(rows, method, enroll(shot, "medoid")),
                _cluster_means(rows, method, enroll(shot, "random")))
    payload = {"schema_version": "phi-acquisition-statistics-v2", "final_query_used": True,
               "confidence_interval": "10000-draw nonparametric bootstrap over 12 direction-class clusters",
               "paired_test": "two-sided Wilcoxon signed-rank over 12 direction-class seed-averaged differences",
               "estimates": estimates, "paired_tests": tests,
               "interpretation_guardrail": "Six held-out labels are reused across directions and are not equivalent to independent sites; p-values are descriptive."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
