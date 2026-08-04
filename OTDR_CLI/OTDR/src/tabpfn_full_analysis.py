from __future__ import annotations

"""Independent reconstruction, statistics, and plots for the full TabPFN study."""

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .lifecycle_analysis import (
    fault_aware_leave_one_out,
    hierarchical_bootstrap,
    holm_adjust,
    paired_sign_flip,
)
from .lifecycle_metrics import hard_prediction_metrics
from .study_state import atomic_json, file_sha256, validate_run
from .tabpfn_full_study import (
    FROZEN_PROTOCOL_SHA256,
    METHODS,
    PROTOCOL_PATH,
    STUDY_ROOT,
    load_protocol,
    probability_sufficient_statistics,
)


LIFECYCLE_FULL_QUERY_TABLE = (
    STUDY_ROOT.parent
    / "otdr_feature_assisted_lifecycle_study"
    / "tables"
    / "full"
    / "cfe_per_draw.csv"
)


SCALAR_COLUMNS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "base_accuracy",
    "enrolled_accuracy",
    "harmonic_mean",
    "worst_enrolled_recall",
    "normal_far_after_enrollment",
    "nll",
    "brier",
    "ece_15",
    "elapsed_seconds",
)


def discover_units(regime: str) -> list[Path]:
    stage = "full_benchmark" if regime == "full" else "summary_only"
    result = []
    for unit in sorted((STUDY_ROOT / stage).glob("pair_*/seed_*")):
        # A currently running unit creates its directory before its atomic
        # metrics/manifest commit. It is not an analysis unit until committed.
        if not (unit / "manifest.json").is_file():
            continue
        valid, reason = validate_run(
            unit,
            expected={
                "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                "evidence_schema": 3,
            },
        )
        if not valid:
            raise RuntimeError(f"Invalid {regime} unit {unit}: {reason}")
        result.append(unit)
    return result


def load_rows(regime: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    scalar_rows: list[dict[str, Any]] = []
    original_rows: list[dict[str, Any]] = []
    for unit in discover_units(regime):
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        pair = "-".join(str(value) for value in metrics["pair"])
        for row_index, row in enumerate(metrics["rows"]):
            original_rows.append(
                {
                    "unit_dir": str(unit),
                    "row_index": row_index,
                    "row": row,
                }
            )
            scalar_rows.append(
                {
                    "run_id": metrics["run_id"],
                    "regime": regime,
                    "pair": pair,
                    "fault_a": int(metrics["pair"][0]),
                    "fault_b": int(metrics["pair"][1]),
                    "seed": int(metrics["seed"]),
                    "method": row["method"],
                    "shots": int(row["shots"]),
                    "draw": int(row["draw"]),
                    **{name: float(row[name]) for name in SCALAR_COLUMNS},
                }
            )
    return pd.DataFrame(scalar_rows), original_rows


def reconstruct_metrics(
    original_records: list[dict[str, Any]],
    *,
    tolerance: float = 2e-6,
) -> dict[str, Any]:
    maximum = {name: 0.0 for name in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "base_accuracy",
        "enrolled_accuracy",
        "harmonic_mean",
        "worst_enrolled_recall",
        "nll",
        "brier",
        "ece_15",
    )}
    failures = []
    cache: dict[str, dict[str, np.ndarray]] = {}
    for index, record in enumerate(original_records):
        row = record["row"]
        unit_dir = record["unit_dir"]
        if unit_dir not in cache:
            evidence_path = Path(unit_dir) / "prediction_evidence.npz"
            if not evidence_path.is_file():
                failures.append(
                    {
                        "row": index,
                        "method": row["method"],
                        "metric": "prediction_evidence",
                        "difference": "artifact_missing",
                    }
                )
                continue
            with np.load(evidence_path) as evidence:
                cache[unit_dir] = {
                    name: evidence[name].copy()
                    for name in evidence.files
                }
        evidence = cache[unit_dir]
        row_index = int(record["row_index"])
        expected_row_id = (
            f"{row['method']}|shot={row['shots']}|draw={row['draw']}"
        )
        if str(evidence["row_ids"][row_index]) != expected_row_id:
            failures.append(
                {
                    "row": index,
                    "method": row["method"],
                    "metric": "row_id_alignment",
                    "difference": str(evidence["row_ids"][row_index]),
                }
            )
            continue
        labels = evidence["labels"].astype(int)
        probability = evidence["probabilities"][row_index].astype(np.float64)
        prediction = evidence["predictions"][row_index].astype(int)
        if not np.array_equal(prediction, probability.argmax(1)):
            failures.append(
                {
                    "row": index,
                    "method": row["method"],
                    "metric": "prediction_probability_alignment",
                    "difference": "argmax_mismatch",
                }
            )
            continue
        reconstructed = {
            **hard_prediction_metrics(
                labels,
                prediction,
                base_class_ids=tuple(row["base_class_ids"]),
                enrolled_class_ids=tuple(row["enrolled_class_ids"]),
            ),
            **probability_sufficient_statistics(probability, labels),
        }
        if reconstructed["confusion_matrix"] != row["confusion_matrix"]:
            failures.append(
                {
                    "row": index,
                    "method": row["method"],
                    "metric": "confusion_matrix",
                    "difference": "matrix_mismatch",
                }
            )
        for name, value in reconstructed.items():
            if name not in maximum:
                continue
            difference = abs(float(row[name]) - value)
            maximum[name] = max(maximum[name], difference)
            if difference > tolerance:
                failures.append(
                    {
                        "row": index,
                        "method": row["method"],
                        "shots": row["shots"],
                        "draw": row["draw"],
                        "metric": name,
                        "difference": difference,
                    }
                )
    return {
        "schema_version": 1,
        "rows_reconstructed": len(original_records),
        "tolerance": tolerance,
        "maximum_absolute_difference": maximum,
        "failures": failures[:100],
        "passed": not failures,
        "method": (
            "Hard and probability metrics reconstructed from independently "
            "persisted per-example labels, predictions, and probabilities."
        ),
    }


def audit_group_manifests(units: list[Path]) -> dict[str, Any]:
    failures = []
    support_records = 0
    context_records = 0
    for unit in units:
        query = json.loads((unit / "query_manifest.json").read_text(encoding="utf-8"))
        support = json.loads(
            (unit / "support_manifest.json").read_text(encoding="utf-8")
        )
        context = json.loads(
            (unit / "context_manifest.json").read_text(encoding="utf-8")
        )
        query_groups = tuple(query["groups"])
        if len(query_groups) != len(set(query_groups)):
            failures.append({"unit": str(unit), "failure": "duplicate_query_group"})
        support_lookup = {}
        for record in support:
            support_records += 1
            groups = tuple(
                group
                for values in record["groups_by_class"].values()
                for group in values
            )
            key = (int(record["shots"]), int(record["draw"]))
            support_lookup[key] = set(groups)
            if len(groups) != len(set(groups)):
                failures.append(
                    {"unit": str(unit), "failure": "duplicate_support_group", "key": key}
                )
            if set(groups) & set(query_groups):
                failures.append(
                    {"unit": str(unit), "failure": "support_query_overlap", "key": key}
                )
        for record in context:
            context_records += 1
            groups = tuple(
                group
                for values in record["base_groups_by_class"].values()
                for group in values
            )
            key = (int(record["shots"]), int(record["draw"]))
            if len(groups) != len(set(groups)):
                failures.append(
                    {"unit": str(unit), "failure": "duplicate_context_group", "key": key}
                )
            if set(groups) & set(query_groups):
                failures.append(
                    {"unit": str(unit), "failure": "context_query_overlap", "key": key}
                )
            if set(groups) & support_lookup[key]:
                failures.append(
                    {"unit": str(unit), "failure": "context_support_overlap", "key": key}
                )
    return {
        "units": len(units),
        "support_records": support_records,
        "context_records": context_records,
        "failures": failures[:100],
        "passed": not failures,
    }


def _summary(
    frame: pd.DataFrame,
    *,
    bootstrap_iterations: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    units = (
        frame.groupby(["method", "shots", "pair", "seed"], as_index=False)[
            list(SCALAR_COLUMNS)
        ]
        .mean()
    )
    rows = []
    details: dict[str, Any] = {}
    for (method, shots), part in units.groupby(["method", "shots"], sort=True):
        boot = hierarchical_bootstrap(
            part,
            "harmonic_mean",
            iterations=bootstrap_iterations,
            seed=20260728 + int(shots),
        )
        key = f"{method}|{shots}"
        details[key] = {
            "harmonic_mean": boot,
            "base_accuracy": hierarchical_bootstrap(
                part,
                "base_accuracy",
                iterations=bootstrap_iterations,
                seed=20260828 + int(shots),
            ),
            "enrolled_accuracy": hierarchical_bootstrap(
                part,
                "enrolled_accuracy",
                iterations=bootstrap_iterations,
                seed=20260928 + int(shots),
            ),
            "fault_aware": fault_aware_leave_one_out(part, "harmonic_mean"),
            "pair_seed_units": len(part),
            "support_draw_rows": int(
                len(
                    frame[
                        (frame["method"] == method)
                        & (frame["shots"] == shots)
                    ]
                )
            ),
            "fraction_pair_seed_h_ge_0_90": float(
                (part["harmonic_mean"] >= 0.90).mean()
            ),
            "fraction_pair_seed_h_ge_0_80": float(
                (part["harmonic_mean"] >= 0.80).mean()
            ),
            "fraction_pair_seed_h_ge_0_95": float(
                (part["harmonic_mean"] >= 0.95).mean()
            ),
            "minimum_pair_seed_h": float(part["harmonic_mean"].min()),
            "q05_pair_seed_h": float(part["harmonic_mean"].quantile(0.05)),
            "median_pair_seed_h": float(part["harmonic_mean"].median()),
        }
        rows.append(
            {
                "method": method,
                "shots": int(shots),
                "pair_seed_units": len(part),
                "harmonic_mean": boot["mean"],
                "harmonic_ci_low": boot["ci_low"],
                "harmonic_ci_high": boot["ci_high"],
                "base_accuracy": float(part["base_accuracy"].mean()),
                "enrolled_accuracy": float(part["enrolled_accuracy"].mean()),
                "accuracy": float(part["accuracy"].mean()),
                "balanced_accuracy": float(part["balanced_accuracy"].mean()),
                "macro_f1": float(part["macro_f1"].mean()),
                "worst_enrolled_recall": float(
                    part["worst_enrolled_recall"].mean()
                ),
                "normal_far_after_enrollment": float(
                    part["normal_far_after_enrollment"].mean()
                ),
                "nll": float(part["nll"].mean()),
                "brier": float(part["brier"].mean()),
                "ece_15": float(part["ece_15"].mean()),
                "fraction_pair_seed_h_ge_0_90": float(
                    (part["harmonic_mean"] >= 0.90).mean()
                ),
                "fraction_pair_seed_h_ge_0_80": float(
                    (part["harmonic_mean"] >= 0.80).mean()
                ),
                "fraction_pair_seed_h_ge_0_95": float(
                    (part["harmonic_mean"] >= 0.95).mean()
                ),
                "minimum_pair_seed_h": float(part["harmonic_mean"].min()),
                "q05_pair_seed_h": float(
                    part["harmonic_mean"].quantile(0.05)
                ),
            }
        )
    return pd.DataFrame(rows), details


def _comparisons(
    frame: pd.DataFrame,
    *,
    sign_flip_iterations: int,
) -> pd.DataFrame:
    units = (
        frame.groupby(["method", "shots", "pair", "seed"], as_index=False)[
            "harmonic_mean"
        ]
        .mean()
    )
    rows = []
    for shots in sorted(units["shots"].unique()):
        tabpfn = units[
            (units["method"] == "tabpfn_v2") & (units["shots"] == shots)
        ]
        shot_rows = []
        for method in sorted(set(units["method"]) - {"tabpfn_v2"}):
            comparator = units[
                (units["method"] == method) & (units["shots"] == shots)
            ]
            result = paired_sign_flip(
                tabpfn,
                comparator,
                value="harmonic_mean",
                iterations=sign_flip_iterations,
                seed=20260728 + int(shots),
            )
            shot_rows.append(
                {
                    "shots": int(shots),
                    "comparison": f"tabpfn_v2 - {method}",
                    "comparator": method,
                    **result,
                }
            )
        adjusted = holm_adjust(
            [row["two_sided_permutation_p"] for row in shot_rows]
        )
        for row, p_value in zip(shot_rows, adjusted, strict=True):
            row["holm_adjusted_p"] = p_value
        rows.extend(shot_rows)
    return pd.DataFrame(rows)


def compare_feature_regimes(
    *,
    bootstrap_iterations: int = 5000,
    sign_flip_iterations: int = 20000,
) -> pd.DataFrame:
    """Paired full-minus-summary analysis on identical pair/seed units."""
    table_root = STUDY_ROOT / "tables"
    full_path = table_root / "full" / "pair_seed_units.csv"
    summary_path = table_root / "summary_only" / "pair_seed_units.csv"
    if not full_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            "Both regime analyses must exist before feature comparison."
        )
    full = pd.read_csv(full_path)
    summary = pd.read_csv(summary_path)
    rows = []
    common_methods = sorted(set(full["method"]) & set(summary["method"]))
    for shots in sorted(set(full["shots"]) & set(summary["shots"])):
        shot_rows = []
        for method in common_methods:
            left = full[(full["method"] == method) & (full["shots"] == shots)]
            right = summary[
                (summary["method"] == method) & (summary["shots"] == shots)
            ]
            merged = left.merge(
                right,
                on=["pair", "seed", "method", "shots", "fault_a", "fault_b"],
                suffixes=("_full", "_summary"),
            )
            if merged.empty:
                continue
            merged["harmonic_difference"] = (
                merged["harmonic_mean_full"]
                - merged["harmonic_mean_summary"]
            )
            interval = hierarchical_bootstrap(
                merged.rename(columns={"harmonic_difference": "value"}),
                "value",
                iterations=bootstrap_iterations,
                seed=20261001 + int(shots),
            )
            permutation = paired_sign_flip(
                left,
                right,
                value="harmonic_mean",
                iterations=sign_flip_iterations,
                seed=20261101 + int(shots),
            )
            shot_rows.append(
                {
                    "method": method,
                    "shots": int(shots),
                    "matched_pair_seed_units": len(merged),
                    "full_harmonic_mean": float(
                        merged["harmonic_mean_full"].mean()
                    ),
                    "summary_harmonic_mean": float(
                        merged["harmonic_mean_summary"].mean()
                    ),
                    "full_minus_summary_mean": interval["mean"],
                    "full_minus_summary_ci_low": interval["ci_low"],
                    "full_minus_summary_ci_high": interval["ci_high"],
                    "full_minus_summary_median": float(
                        merged["harmonic_difference"].median()
                    ),
                    "two_sided_permutation_p": permutation[
                        "two_sided_permutation_p"
                    ],
                }
            )
        adjusted = holm_adjust(
            [row["two_sided_permutation_p"] for row in shot_rows]
        )
        for row, adjusted_p in zip(shot_rows, adjusted, strict=True):
            row["holm_adjusted_p"] = adjusted_p
        rows.extend(shot_rows)
    result = pd.DataFrame(rows)
    result.to_csv(table_root / "feature_regime_comparison.csv", index=False)
    atomic_json(
        table_root / "feature_regime_comparison_manifest.json",
        {
            "bootstrap_iterations": bootstrap_iterations,
            "sign_flip_iterations": sign_flip_iterations,
            "difference_orientation": "full minus summary_only",
            "independence_unit": "pair/seed after support-draw averaging",
            "matched_on": [
                "pair",
                "seed",
                "method",
                "shots",
                "fault_a",
                "fault_b",
            ],
        },
    )
    return result


def _per_pair_fault(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    units = (
        frame.groupby(["method", "shots", "pair", "fault_a", "fault_b", "seed"], as_index=False)[
            ["harmonic_mean", "base_accuracy", "enrolled_accuracy"]
        ]
        .mean()
    )
    per_pair = (
        units.groupby(
            ["method", "shots", "pair", "fault_a", "fault_b"], as_index=False
        )[
            ["harmonic_mean", "base_accuracy", "enrolled_accuracy"]
        ]
        .agg(["mean", "std", "min", "max"])
    )
    per_pair.columns = [
        "_".join(str(value) for value in column if value != "")
        for column in per_pair.columns
    ]
    per_fault_rows = []
    for (method, shots), part in units.groupby(["method", "shots"]):
        for fault in range(1, 8):
            selected = part[
                (part["fault_a"] == fault) | (part["fault_b"] == fault)
            ]
            per_fault_rows.append(
                {
                    "method": method,
                    "shots": int(shots),
                    "fault": fault,
                    "pair_seed_units": len(selected),
                    "harmonic_mean": float(selected["harmonic_mean"].mean()),
                    "harmonic_std": float(selected["harmonic_mean"].std()),
                    "base_accuracy": float(selected["base_accuracy"].mean()),
                    "enrolled_accuracy": float(
                        selected["enrolled_accuracy"].mean()
                    ),
                }
            )
    return per_pair, pd.DataFrame(per_fault_rows)


def _per_fault_recall(
    units: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return exact class recall at draw, pair/seed, and summary levels.

    The older ``per_fault.csv`` table answers a different question: how well
    pairs *containing* a fault perform in aggregate.  This function reads each
    saved row's exact eight-class recall so enrolled-fault failures cannot be
    hidden by strong base-class performance.
    """
    rows: list[dict[str, Any]] = []
    for unit in units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        pair = "-".join(str(value) for value in metrics["pair"])
        for row in metrics["rows"]:
            enrolled = {int(value) for value in row["enrolled_class_ids"]}
            for fault in range(1, 8):
                rows.append(
                    {
                        "run_id": metrics["run_id"],
                        "pair": pair,
                        "fault_a": int(metrics["pair"][0]),
                        "fault_b": int(metrics["pair"][1]),
                        "seed": int(metrics["seed"]),
                        "method": row["method"],
                        "shots": int(row["shots"]),
                        "draw": int(row["draw"]),
                        "fault": fault,
                        "role": "enrolled" if fault in enrolled else "base",
                        "recall": float(row["per_class_recall"][str(fault)]),
                    }
                )
    draw_frame = pd.DataFrame(rows)
    pair_seed = (
        draw_frame.groupby(
            [
                "method",
                "shots",
                "pair",
                "fault_a",
                "fault_b",
                "seed",
                "fault",
                "role",
            ],
            as_index=False,
        )["recall"]
        .mean()
    )
    summary = (
        pair_seed.groupby(
            ["method", "shots", "fault", "role"], as_index=False
        )
        .agg(
            pair_seed_units=("recall", "size"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            recall_minimum=("recall", "min"),
            recall_q05=("recall", lambda values: values.quantile(0.05)),
            recall_median=("recall", "median"),
            near_zero_pair_seed_units=(
                "recall",
                lambda values: int((values <= 0.05).sum()),
            ),
            weak_pair_seed_units=(
                "recall",
                lambda values: int((values < 0.50).sum()),
            ),
        )
    )
    summary["near_zero_threshold"] = 0.05
    summary["weak_threshold"] = 0.50
    summary["near_zero_fraction"] = (
        summary["near_zero_pair_seed_units"] / summary["pair_seed_units"]
    )
    summary["weak_fraction"] = (
        summary["weak_pair_seed_units"] / summary["pair_seed_units"]
    )
    return draw_frame, pair_seed, summary


def _prototype_efficiency(units: list[Path]) -> pd.DataFrame:
    rows = []
    for unit in units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        for row in metrics["rows"]:
            if "storage_bytes" not in row:
                continue
            rows.append(
                {
                    "run_id": metrics["run_id"],
                    "method": row["method"],
                    "shots": int(row["shots"]),
                    "draw": int(row["draw"]),
                    "storage_bytes": int(row["storage_bytes"]),
                    "enroll_and_predict_seconds": float(row["elapsed_seconds"]),
                    "query_examples": int(row["query_examples"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["method", "shots"], as_index=False)
        .agg(
            evaluations=("run_id", "size"),
            prototype_storage_bytes_mean=("storage_bytes", "mean"),
            prototype_storage_bytes_max=("storage_bytes", "max"),
            enroll_and_predict_seconds_mean=(
                "enroll_and_predict_seconds",
                "mean",
            ),
            enroll_and_predict_seconds_p95=(
                "enroll_and_predict_seconds",
                lambda values: values.quantile(0.95),
            ),
            query_examples=("query_examples", "median"),
        )
    )


def _prior_full_query_sensitivity(
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Summarize frozen full-query CFE/1NN results as unmatched sensitivity.

    These rows use roughly 33k--39k queries per pair/seed, not the bounded
    800-example TabPFN cohort.  They are useful for sensitivity only and must
    never be presented as a matched TabPFN comparison.
    """
    if not LIFECYCLE_FULL_QUERY_TABLE.is_file():
        raise FileNotFoundError(LIFECYCLE_FULL_QUERY_TABLE)
    mapping = {
        "finalist": "cfe_finalist",
        "uncalibrated_mean": "cfe_uncalibrated_mean",
        "encoder_cosine_1nn": "encoder_cosine_1nn",
        "raw_cosine_1nn": "raw_cosine_1nn",
        "raw_euclidean_1nn": "raw_euclidean_1nn",
        "raw_mahalanobis_1nn": "raw_mahalanobis_1nn",
    }
    columns = [
        "run_id",
        "pair",
        "seed",
        "method",
        "shots",
        "base_accuracy",
        "enrolled_accuracy",
        "harmonic_mean",
        "balanced_accuracy",
        "macro_f1",
        "worst_enrolled_recall",
        "normal_far_after_enrollment",
        "storage_bytes",
        "confusion_matrix",
    ]
    frame = pd.read_csv(LIFECYCLE_FULL_QUERY_TABLE, usecols=columns)
    frame = frame[frame["method"].isin(mapping)].copy()
    frame["method"] = frame["method"].map(mapping)
    query_sizes = frame.drop_duplicates("run_id")[["run_id", "confusion_matrix"]]
    query_sizes["query_examples"] = query_sizes["confusion_matrix"].map(
        lambda value: int(np.asarray(ast.literal_eval(value)).sum())
    )
    frame = frame.merge(
        query_sizes[["run_id", "query_examples"]], on="run_id", how="left"
    )
    metrics = [
        "base_accuracy",
        "enrolled_accuracy",
        "harmonic_mean",
        "balanced_accuracy",
        "macro_f1",
        "worst_enrolled_recall",
        "normal_far_after_enrollment",
        "storage_bytes",
        "query_examples",
    ]
    pair_seed = (
        frame.groupby(
            ["method", "shots", "pair", "seed", "run_id"], as_index=False
        )[metrics]
        .mean()
    )
    summary = (
        pair_seed.groupby(["method", "shots"], as_index=False)
        .agg(
            pair_seed_units=("run_id", "size"),
            base_accuracy=("base_accuracy", "mean"),
            enrolled_accuracy=("enrolled_accuracy", "mean"),
            harmonic_mean=("harmonic_mean", "mean"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            macro_f1=("macro_f1", "mean"),
            worst_enrolled_recall=("worst_enrolled_recall", "mean"),
            normal_far_after_enrollment=(
                "normal_far_after_enrollment",
                "mean",
            ),
            query_examples_min=("query_examples", "min"),
            query_examples_median=("query_examples", "median"),
            query_examples_max=("query_examples", "max"),
            prototype_storage_bytes=("storage_bytes", "mean"),
        )
    )
    provenance = {
        "source_path": str(LIFECYCLE_FULL_QUERY_TABLE),
        "source_sha256": file_sha256(LIFECYCLE_FULL_QUERY_TABLE),
        "cohort": "frozen lifecycle full-query cohort",
        "matched_to_bounded_tabpfn_query": False,
        "per_unit_query_range": [
            int(pair_seed["query_examples"].min()),
            int(pair_seed["query_examples"].max()),
        ],
        "interpretation": (
            "Sensitivity-only evidence. Do not compare these values with the "
            "800-query TabPFN rows as though the cohorts were matched."
        ),
    }
    return pair_seed, summary, provenance


def _context_sensitivity(units: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for unit in units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        pair = "-".join(str(value) for value in metrics["pair"])
        for row in metrics.get("context_sensitivity_rows", []):
            rows.append(
                {
                    "run_id": metrics["run_id"],
                    "pair": pair,
                    "seed": int(metrics["seed"]),
                    **row,
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, frame
    within = (
        frame.groupby(["pair", "seed", "shots", "draw"], as_index=False)[
            "harmonic_mean"
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    within["range"] = within["max"] - within["min"]
    summary = (
        within.groupby("shots", as_index=False)[["std", "range"]]
        .agg(["mean", "median", "max"])
    )
    summary.columns = [
        "_".join(str(value) for value in column if value != "")
        for column in summary.columns
    ]
    return frame, summary


def _probability_diagnostics(
    units: list[Path],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    reliability: dict[tuple[str, int, int], list[float]] = {}
    confidence_hist: dict[tuple[str, int, str, int], int] = {}
    classwise: dict[tuple[str, int, int], list[float]] = {}
    correct_confidence: dict[tuple[str, int, str], list[float]] = {}
    reliability_edges = np.linspace(0, 1, 16)
    histogram_edges = np.linspace(0, 1, 51)
    for unit in units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        with np.load(unit / "prediction_evidence.npz") as evidence:
            labels = evidence["labels"].astype(int)
            probabilities = evidence["probabilities"]
            predictions = evidence["predictions"].astype(int)
            for row_index, row in enumerate(metrics["rows"]):
                method = str(row["method"])
                shots = int(row["shots"])
                probability = probabilities[row_index].astype(np.float64)
                prediction = predictions[row_index]
                confidence = probability.max(1)
                true_probability = probability[np.arange(len(labels)), labels]
                correct = prediction == labels
                reliability_bin = np.clip(
                    np.digitize(confidence, reliability_edges, right=True) - 1,
                    0,
                    14,
                )
                for bin_index in range(15):
                    mask = reliability_bin == bin_index
                    if not mask.any():
                        continue
                    key = (method, shots, bin_index)
                    values = reliability.setdefault(key, [0.0, 0.0, 0.0])
                    values[0] += int(mask.sum())
                    values[1] += float(confidence[mask].sum())
                    values[2] += int(correct[mask].sum())
                histogram_bin = np.clip(
                    np.digitize(confidence, histogram_edges, right=True) - 1,
                    0,
                    49,
                )
                for correctness, mask in (
                    ("correct", correct),
                    ("incorrect", ~correct),
                ):
                    if mask.any():
                        counts = np.bincount(
                            histogram_bin[mask], minlength=50
                        )
                        for bin_index, count in enumerate(counts):
                            if count:
                                key = (method, shots, correctness, bin_index)
                                confidence_hist[key] = (
                                    confidence_hist.get(key, 0) + int(count)
                                )
                        key = (method, shots, correctness)
                        values = correct_confidence.setdefault(
                            key, [0.0, 0.0]
                        )
                        values[0] += int(mask.sum())
                        values[1] += float(confidence[mask].sum())
                for class_id in range(8):
                    mask = labels == class_id
                    key = (method, shots, class_id)
                    values = classwise.setdefault(
                        key, [0.0, 0.0, 0.0, 0.0]
                    )
                    values[0] += int(mask.sum())
                    values[1] += float(confidence[mask].sum())
                    values[2] += float(true_probability[mask].sum())
                    values[3] += int(correct[mask].sum())
    reliability_rows = []
    for (method, shots, bin_index), (count, confidence_sum, correct_count) in sorted(
        reliability.items()
    ):
        reliability_rows.append(
            {
                "method": method,
                "shots": shots,
                "bin": bin_index,
                "left": reliability_edges[bin_index],
                "right": reliability_edges[bin_index + 1],
                "count": int(count),
                "mean_confidence": confidence_sum / count,
                "empirical_accuracy": correct_count / count,
            }
        )
    histogram_rows = [
        {
            "method": method,
            "shots": shots,
            "correctness": correctness,
            "bin": bin_index,
            "left": histogram_edges[bin_index],
            "right": histogram_edges[bin_index + 1],
            "count": count,
        }
        for (method, shots, correctness, bin_index), count in sorted(
            confidence_hist.items()
        )
    ]
    classwise_rows = [
        {
            "method": method,
            "shots": shots,
            "class_id": class_id,
            "count": int(values[0]),
            "mean_max_confidence": values[1] / values[0],
            "mean_true_class_probability": values[2] / values[0],
            "accuracy": values[3] / values[0],
        }
        for (method, shots, class_id), values in sorted(classwise.items())
    ]
    correctness_rows = [
        {
            "method": method,
            "shots": shots,
            "correctness": correctness,
            "count": int(values[0]),
            "mean_confidence": values[1] / values[0],
        }
        for (method, shots, correctness), values in sorted(
            correct_confidence.items()
        )
    ]
    return (
        pd.DataFrame(reliability_rows),
        pd.DataFrame(histogram_rows),
        pd.DataFrame(classwise_rows),
        pd.DataFrame(correctness_rows),
    )


def _unit_efficiency(units: list[Path]) -> pd.DataFrame:
    rows = []
    for unit in units:
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        cfe = metrics.get("cfe_source") or {}
        cuda = metrics.get("cuda_diagnostics") or {}
        rows.append(
            {
                "run_id": metrics["run_id"],
                "regime": metrics["regime"],
                "pair": "-".join(str(value) for value in metrics["pair"]),
                "seed": int(metrics["seed"]),
                "total_duration_seconds": float(metrics["duration_seconds"]),
                "query_examples": int(metrics["query_examples"]),
                "checkpoint_resolution_seconds": float(
                    metrics.get("tabpfn_checkpoint_resolution_seconds", 0)
                ),
                "model_initialization_seconds": float(
                    metrics.get("tabpfn_model_initialization_seconds", 0)
                ),
                "first_context_seconds_including_lazy_load": float(
                    metrics.get(
                        "tabpfn_first_context_seconds_including_lazy_load", 0
                    )
                ),
                "checkpoint_size_bytes": metrics.get(
                    "tabpfn_checkpoint_size_bytes"
                ),
                "peak_cuda_allocated_bytes": cuda.get("peak_allocated_bytes"),
                "peak_cuda_reserved_bytes": cuda.get("peak_reserved_bytes"),
                "cfe_embedding_inference_seconds": float(
                    cfe.get("embedding_inference_seconds", 0)
                ),
                "cfe_embedding_source": cfe.get("embedding_source"),
            }
        )
    return pd.DataFrame(rows)


def _save_plots(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    reliability: pd.DataFrame,
    confidence_histogram: pd.DataFrame,
    per_fault_recall_summary: pd.DataFrame,
    *,
    regime: str,
) -> list[str]:
    plot_root = STUDY_ROOT / "plots" / regime
    plot_root.mkdir(parents=True, exist_ok=True)
    created = []
    method_order = [
        method
        for method in METHODS
        if method in set(summary["method"])
    ]
    colors = plt.cm.tab10(np.linspace(0, 1, len(method_order)))

    plt.figure(figsize=(11, 7))
    for method, color in zip(method_order, colors, strict=True):
        part = summary[summary["method"] == method].sort_values("shots")
        plt.plot(
            part["shots"],
            part["harmonic_mean"],
            marker="o",
            label=method,
            color=color,
        )
        plt.fill_between(
            part["shots"],
            part["harmonic_ci_low"],
            part["harmonic_ci_high"],
            alpha=0.12,
            color=color,
        )
    plt.axhline(0.95, color="black", linestyle="--", linewidth=1, label="0.95 target")
    plt.xticks([1, 3, 5])
    plt.ylim(0, 1.02)
    plt.xlabel("Support examples per enrolled class")
    plt.ylabel("Harmonic mean")
    plt.title(f"{regime}: enrollment performance across 21 fault pairs")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path = plot_root / "enrollment_curves.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    enrolled_recall = per_fault_recall_summary[
        (per_fault_recall_summary["method"] == "tabpfn_v2")
        & (per_fault_recall_summary["role"] == "enrolled")
    ]
    recall_matrix = (
        enrolled_recall.pivot(index="shots", columns="fault", values="recall_mean")
        .reindex(index=[1, 3, 5], columns=range(1, 8))
        .to_numpy(dtype=float)
    )
    plt.figure(figsize=(10, 4.8))
    image = plt.imshow(recall_matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    for row in range(recall_matrix.shape[0]):
        for column in range(recall_matrix.shape[1]):
            value = recall_matrix[row, column]
            plt.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value < 0.55 else "black",
            )
    plt.xticks(range(7), range(1, 8))
    plt.yticks(range(3), ["1", "3", "5"])
    plt.xlabel("Enrolled fault")
    plt.ylabel("Support examples per enrolled class")
    plt.title(f"{regime}: TabPFN enrolled-fault recall")
    plt.colorbar(image, label="Mean recall")
    plt.tight_layout()
    path = plot_root / "tabpfn_enrolled_fault_recall.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    units = (
        frame.groupby(["method", "shots", "pair", "fault_a", "fault_b", "seed"], as_index=False)[
            ["harmonic_mean", "base_accuracy", "enrolled_accuracy"]
        ]
        .mean()
    )
    tab5 = units[
        (units["method"] == "tabpfn_v2") & (units["shots"] == 5)
    ]
    pair_mean = tab5.groupby(["fault_a", "fault_b"])["harmonic_mean"].mean()
    matrix = np.full((7, 7), np.nan)
    for (left, right), value in pair_mean.items():
        matrix[left - 1, right - 1] = value
        matrix[right - 1, left - 1] = value
    # A pair cannot hold out the same fault twice. Keep the diagonal missing
    # rather than rendering an invented perfect score.
    np.fill_diagonal(matrix, np.nan)
    plt.figure(figsize=(8, 7))
    pair_cmap = plt.get_cmap("viridis").copy()
    pair_cmap.set_bad(color="#eeeeee")
    image = plt.imshow(matrix, vmin=0, vmax=1, cmap=pair_cmap)
    for row in range(7):
        for column in range(7):
            value = matrix[row, column]
            plt.text(
                column,
                row,
                "N/A" if np.isnan(value) else f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if np.isfinite(value) and value < 0.55 else "black",
                fontsize=8,
            )
    plt.xticks(range(7), range(1, 8))
    plt.yticks(range(7), range(1, 8))
    plt.xlabel("Held-out fault")
    plt.ylabel("Held-out fault")
    plt.title("TabPFN-v2 five-shot H by held-out pair")
    plt.colorbar(image, label="Harmonic mean")
    plt.tight_layout()
    path = plot_root / "tabpfn_five_shot_pair_heatmap.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    plt.figure(figsize=(8, 7))
    plt.scatter(
        tab5["base_accuracy"],
        tab5["enrolled_accuracy"],
        c=tab5["harmonic_mean"],
        vmin=0,
        vmax=1,
        cmap="viridis",
        alpha=0.75,
        edgecolor="none",
    )
    plt.axvline(0.95, color="black", linestyle="--", linewidth=1)
    plt.axhline(0.95, color="black", linestyle="--", linewidth=1)
    plt.xlim(0, 1.02)
    plt.ylim(0, 1.02)
    plt.xlabel("Retained base accuracy")
    plt.ylabel("Enrolled accuracy")
    plt.title("TabPFN-v2 five-shot pair/seed trade-off")
    plt.colorbar(label="Harmonic mean")
    plt.tight_layout()
    path = plot_root / "tabpfn_five_shot_base_vs_enrolled.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    tab_draws = frame[frame["method"] == "tabpfn_v2"]
    plt.figure(figsize=(9, 6))
    plt.boxplot(
        [
            tab_draws[tab_draws["shots"] == shots]["harmonic_mean"]
            for shots in (1, 3, 5)
        ],
        tick_labels=["1", "3", "5"],
        showfliers=False,
    )
    plt.axhline(0.95, color="black", linestyle="--", linewidth=1)
    plt.ylim(0, 1.02)
    plt.xlabel("Shots")
    plt.ylabel("Harmonic mean across support draws")
    plt.title("TabPFN-v2 support-draw stability")
    plt.tight_layout()
    path = plot_root / "tabpfn_support_draw_stability.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    calibration = summary[summary["shots"] == 5].set_index("method").loc[
        method_order
    ]
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, title in zip(
        axes,
        ("nll", "brier", "ece_15"),
        ("NLL", "Multiclass Brier", "ECE-15"),
        strict=True,
    ):
        axis.bar(range(len(calibration)), calibration[metric], color=colors)
        axis.set_xticks(range(len(calibration)))
        axis.set_xticklabels(calibration.index, rotation=75, ha="right", fontsize=7)
        axis.set_title(title)
    figure.suptitle(f"{regime}: five-shot probability quality")
    figure.tight_layout()
    path = plot_root / "five_shot_calibration.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    created.append(str(path))

    latency = (
        frame.groupby(["method", "shots"])["elapsed_seconds"].median().unstack()
    )
    latency = latency.loc[[method for method in method_order if method in latency.index]]
    latency.plot(kind="bar", figsize=(11, 6))
    plt.yscale("log")
    plt.ylabel("Median fit/enroll + prediction seconds per draw (log scale)")
    plt.xlabel("Method")
    plt.title(f"{regime}: computational cost")
    plt.xticks(rotation=60, ha="right")
    plt.tight_layout()
    path = plot_root / "efficiency.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    plt.figure(figsize=(9, 7))
    five_shot_reliability = reliability[reliability["shots"] == 5]
    for method, color in zip(method_order, colors, strict=True):
        part = five_shot_reliability[
            five_shot_reliability["method"] == method
        ]
        if part.empty:
            continue
        plt.plot(
            part["mean_confidence"],
            part["empirical_accuracy"],
            marker="o",
            markersize=3,
            label=method,
            color=color,
        )
    plt.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Empirical accuracy")
    plt.title(f"{regime}: five-shot reliability")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    path = plot_root / "five_shot_reliability.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))

    tab_hist = confidence_histogram[
        (confidence_histogram["method"] == "tabpfn_v2")
        & (confidence_histogram["shots"] == 5)
    ]
    plt.figure(figsize=(9, 6))
    for correctness, color in (("correct", "#277da1"), ("incorrect", "#f94144")):
        part = tab_hist[tab_hist["correctness"] == correctness]
        if part.empty:
            continue
        total = part["count"].sum()
        center = (part["left"] + part["right"]) / 2
        plt.step(
            center,
            part["count"] / total,
            where="mid",
            label=correctness,
            color=color,
        )
    plt.xlabel("Maximum predicted probability")
    plt.ylabel("Within-group fraction")
    plt.title(f"{regime}: TabPFN five-shot confidence")
    plt.legend()
    plt.tight_layout()
    path = plot_root / "tabpfn_five_shot_confidence_histogram.png"
    plt.savefig(path, dpi=180)
    plt.close()
    created.append(str(path))
    return created


def analyze(
    regime: str,
    *,
    expected_units: int,
    bootstrap_iterations: int = 5000,
    sign_flip_iterations: int = 20000,
) -> dict[str, Any]:
    protocol = load_protocol()
    units = discover_units(regime)
    if len(units) != expected_units:
        raise RuntimeError(
            f"{regime} has {len(units)} valid units; expected {expected_units}."
        )
    frame, originals = load_rows(regime)
    reconstruction = reconstruct_metrics(originals)
    if not reconstruction["passed"]:
        raise RuntimeError("Independent metric reconstruction failed.")
    group_audit = audit_group_manifests(units)
    if not group_audit["passed"]:
        raise RuntimeError("Independent group-manifest audit failed.")
    table_root = STUDY_ROOT / "tables" / regime
    table_root.mkdir(parents=True, exist_ok=True)
    atomic_json(table_root / "metric_reconstruction.json", reconstruction)
    atomic_json(table_root / "group_manifest_audit.json", group_audit)
    frame.to_csv(table_root / "per_draw.csv", index=False)
    unit_frame = (
        frame.groupby(["method", "shots", "pair", "fault_a", "fault_b", "seed"], as_index=False)[
            list(SCALAR_COLUMNS)
        ]
        .mean()
    )
    unit_frame.to_csv(table_root / "pair_seed_units.csv", index=False)
    summary, details = _summary(
        frame, bootstrap_iterations=bootstrap_iterations
    )
    summary.to_csv(table_root / "headline_summary.csv", index=False)
    atomic_json(table_root / "headline_details.json", details)
    comparisons = _comparisons(
        frame, sign_flip_iterations=sign_flip_iterations
    )
    comparisons.to_csv(table_root / "paired_comparisons.csv", index=False)
    per_pair, per_fault = _per_pair_fault(frame)
    per_pair.to_csv(table_root / "per_pair.csv", index=False)
    per_fault.to_csv(table_root / "per_fault.csv", index=False)
    fault_draw, fault_pair_seed, fault_summary = _per_fault_recall(units)
    fault_draw.to_csv(table_root / "per_fault_recall_draw.csv", index=False)
    fault_pair_seed.to_csv(
        table_root / "per_fault_recall_pair_seed.csv", index=False
    )
    fault_summary.to_csv(
        table_root / "per_fault_recall_summary.csv", index=False
    )
    efficiency = (
        frame.groupby(["method", "shots"])["elapsed_seconds"]
        .agg(
            mean="mean",
            median="median",
            std="std",
            minimum="min",
            maximum="max",
            p95=lambda values: values.quantile(0.95),
        )
        .reset_index()
    )
    efficiency["milliseconds_per_query"] = (
        efficiency["mean"] * 1000 / 800
    )
    efficiency.to_csv(table_root / "efficiency.csv", index=False)
    unit_efficiency = _unit_efficiency(units)
    unit_efficiency.to_csv(table_root / "unit_efficiency.csv", index=False)
    prototype_efficiency = _prototype_efficiency(units)
    prototype_efficiency.to_csv(
        table_root / "prototype_efficiency.csv", index=False
    )
    atomic_json(
        table_root / "efficiency_summary.json",
        {
            "total_benchmark_duration_seconds_sum": float(
                unit_efficiency["total_duration_seconds"].sum()
            ),
            "median_unit_duration_seconds": float(
                unit_efficiency["total_duration_seconds"].median()
            ),
            "p95_unit_duration_seconds": float(
                unit_efficiency["total_duration_seconds"].quantile(0.95)
            ),
            "peak_cuda_allocated_bytes_max": int(
                unit_efficiency["peak_cuda_allocated_bytes"].max()
            ),
            "peak_cuda_reserved_bytes_max": int(
                unit_efficiency["peak_cuda_reserved_bytes"].max()
            ),
            "checkpoint_size_bytes": int(
                unit_efficiency["checkpoint_size_bytes"].dropna().iloc[0]
            ),
            "checkpoint_initialization_seconds_mean": float(
                unit_efficiency["model_initialization_seconds"].mean()
            ),
            "first_context_seconds_including_lazy_load_mean": float(
                unit_efficiency[
                    "first_context_seconds_including_lazy_load"
                ].mean()
            ),
            "cfe_embedding_inference_seconds_mean": float(
                unit_efficiency["cfe_embedding_inference_seconds"].mean()
            ),
        },
    )
    context_rows, context_summary = _context_sensitivity(units)
    context_rows.to_csv(table_root / "context_seed_per_draw.csv", index=False)
    context_summary.to_csv(
        table_root / "context_seed_sensitivity.csv", index=False
    )
    (
        reliability,
        confidence_histogram,
        classwise_confidence,
        correct_incorrect_confidence,
    ) = _probability_diagnostics(units)
    reliability.to_csv(table_root / "reliability.csv", index=False)
    confidence_histogram.to_csv(
        table_root / "confidence_histogram.csv", index=False
    )
    classwise_confidence.to_csv(
        table_root / "classwise_confidence.csv", index=False
    )
    correct_incorrect_confidence.to_csv(
        table_root / "correct_incorrect_confidence.csv", index=False
    )
    per_seed = (
        unit_frame.groupby(["method", "shots", "seed"], as_index=False)[
            ["harmonic_mean", "base_accuracy", "enrolled_accuracy"]
        ]
        .mean()
    )
    per_seed.to_csv(table_root / "per_seed.csv", index=False)
    support_stability = (
        frame.groupby(["method", "shots", "pair", "seed"], as_index=False)[
            "harmonic_mean"
        ]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    support_stability["range"] = (
        support_stability["max"] - support_stability["min"]
    )
    support_stability.to_csv(
        table_root / "support_draw_stability.csv", index=False
    )
    if regime == "full":
        full_query_units, full_query_summary, full_query_provenance = (
            _prior_full_query_sensitivity()
        )
        full_query_units.to_csv(
            table_root / "prior_full_query_pair_seed_units.csv", index=False
        )
        full_query_summary.to_csv(
            table_root / "prior_full_query_sensitivity.csv", index=False
        )
        atomic_json(
            table_root / "prior_full_query_provenance.json",
            full_query_provenance,
        )
    plots = _save_plots(
        frame,
        summary,
        reliability,
        confidence_histogram,
        fault_summary,
        regime=regime,
    )
    result = {
        "schema_version": 1,
        "regime": regime,
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "valid_units": len(units),
        "methods": sorted(frame["method"].unique()),
        "per_method_rows": {
            method: int(count)
            for method, count in frame["method"].value_counts().sort_index().items()
        },
        "bootstrap_iterations": bootstrap_iterations,
        "sign_flip_iterations": sign_flip_iterations,
        "metric_reconstruction": reconstruction,
        "group_manifest_audit": group_audit,
        "per_fault_recall": {
            "near_zero_threshold": 0.05,
            "weak_threshold": 0.50,
            "aggregation": "support draws averaged within pair/seed before threshold counts",
        },
        "plots": plots,
        "primary_metric": "harmonic_mean",
        "independence_unit": "pair/seed after support-draw averaging",
        "protocol_targets": protocol["targets"],
    }
    atomic_json(table_root / "analysis_manifest.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regime", choices=("full", "summary_only"), default="full")
    parser.add_argument("--expected-units", type=int, required=True)
    parser.add_argument("--bootstrap-iterations", type=int, default=5000)
    parser.add_argument("--sign-flip-iterations", type=int, default=20000)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            analyze(
                args.regime,
                expected_units=args.expected_units,
                bootstrap_iterations=args.bootstrap_iterations,
                sign_flip_iterations=args.sign_flip_iterations,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
