from __future__ import annotations

"""Synthesize stress, external, ablation, and TabPFN artifacts."""

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .study_state import atomic_json, file_sha256, utc_now, validate_run


def symmetric_color_limit(values: np.ndarray, floor: float = 0.01) -> float:
    """Return a finite symmetric color limit without clipping either tail."""
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return max(float(floor), float(np.max(np.abs(finite))) if finite.size else 0.0)


def nested_two_key_mapping(
    values: dict[tuple[Any, Any], Any],
) -> dict[str, dict[str, Any]]:
    """Convert a two-level pandas mapping into a JSON-native hierarchy."""
    nested: dict[str, dict[str, Any]] = {}
    for (outer, inner), value in values.items():
        nested.setdefault(str(outer), {})[str(inner)] = value
    return nested


def write_source_snapshot(study_root: Path) -> dict[str, Any]:
    """Hash the final lifecycle implementation and protocol documents.

    This is a completion-time provenance record.  It complements (but cannot
    retroactively strengthen) the source metadata embedded by already-running
    processes.
    """
    module_root = study_root.parents[1]
    source_root = module_root / "src"
    selected: set[Path] = set(source_root.glob("lifecycle_*.py"))
    selected.update(
        path
        for path in (
            source_root / "model_functions" / "lifecycle.py",
            source_root / "model_functions" / "tcn.py",
            source_root / "model_functions" / "zero_shot.py",
            module_root / "tests" / "test_lifecycle.py",
        )
        if path.is_file()
    )
    selected.update(study_root.glob("*.md"))
    selected.update((study_root / "configs").glob("*.json"))
    files = {
        path.relative_to(module_root).as_posix(): {
            "bytes": path.stat().st_size,
            "sha256": file_sha256(path),
        }
        for path in sorted(selected)
        if path.is_file()
    }
    payload = {
        "schema_version": 1,
        "created_at": utc_now(),
        "scope": (
            "Completion-time lifecycle source, tests, frozen configs, and "
            "study protocol documents. Earlier run manifests remain the "
            "authoritative record for code metadata captured during each run."
        ),
        "files": files,
    }
    atomic_json(study_root / "SOURCE_SNAPSHOT.json", payload)
    return payload


def _stress(study_root: Path, table_root: Path, plot_root: Path) -> dict[str, Any]:
    path = study_root / "stress" / "metrics.json"
    if not path.exists():
        return {"available": False}
    valid, reason = validate_run(
        path.parent,
        expected={"run_id": "lifecycle-stress-validation-v1"},
    )
    if not valid:
        return {"available": False, "reason": reason}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame([
        {
            **row,
            "pair": "-".join(str(value) for value in row["pair"]),
        }
        for row in payload["rows"]
    ])
    feature = pd.DataFrame([
        {
            **row,
            "pair": "-".join(str(value) for value in row["pair"]),
        }
        for row in payload["feature_contribution"]
    ])
    stress_class = pd.DataFrame([
        {
            "pair": "-".join(str(value) for value in row["pair"]),
            "variant": row["variant"],
            "stress": row["stress"],
            "severity": row["severity"],
            "class_id": int(class_id),
            "recall": float(recall),
            "recall_delta": float(
                row.get("per_class_recall_delta", {}).get(class_id, 0.0)
            ),
        }
        for row in payload["rows"]
        for class_id, recall in row.get("per_class_recall", {}).items()
    ])
    feature_class = pd.DataFrame([
        {
            "pair": "-".join(str(value) for value in row["pair"]),
            "variant": row["variant"],
            "condition": row["condition"],
            "class_id": int(class_id),
            "recall": float(recall),
            "recall_delta": float(
                row.get("per_class_recall_delta", {}).get(class_id, 0.0)
            ),
        }
        for row in payload["feature_contribution"]
        for class_id, recall in row.get("per_class_recall", {}).items()
    ])
    rows.to_csv(table_root / "stress_results.csv", index=False)
    feature.to_csv(table_root / "feature_contribution.csv", index=False)
    stress_class.to_csv(
        table_root / "stress_per_class_recall.csv", index=False
    )
    feature_class.to_csv(
        table_root / "feature_per_class_recall.csv", index=False
    )
    severe = rows[(rows["severity"] == 1.0) & (rows["stress"] != "clean")]
    heat = severe.groupby(["variant", "stress"])[
        "balanced_accuracy_delta"
    ].mean().unstack()
    heat_values = heat.to_numpy(dtype=float)
    color_limit = symmetric_color_limit(heat_values)
    plt.figure(figsize=(12, max(4, 0.45 * len(heat))))
    image = plt.imshow(
        heat_values,
        aspect="auto",
        cmap="coolwarm",
        vmin=-color_limit,
        vmax=color_limit,
    )
    plt.yticks(range(len(heat.index)), heat.index)
    plt.xticks(
        range(len(heat.columns)), heat.columns, rotation=60, ha="right"
    )
    plt.colorbar(image, label="Balanced-accuracy delta")
    plt.title("Acquisition stress at maximum declared severity")
    plt.tight_layout()
    stress_plot = plot_root / "stress_grid.png"
    plt.savefig(stress_plot, dpi=180)
    plt.close()

    feature_summary = feature.groupby(
        ["variant", "condition"]
    )["balanced_accuracy_delta"].mean().unstack()
    selected = [
        name
        for name in (
            "matched_late_fusion",
            "late_fusion_no_scalar_dropout",
            "morphology_only",
            "context_only",
        )
        if name in feature_summary.index
    ]
    feature_summary.loc[selected].T.plot.barh(figsize=(10, 6))
    plt.axvline(0, color="black", linewidth=0.8)
    plt.xlabel("Balanced-accuracy delta")
    plt.ylabel("Feature intervention")
    plt.title("Feature removal and conditional permutation")
    plt.tight_layout()
    feature_plot = plot_root / "feature_contribution.png"
    plt.savefig(feature_plot, dpi=180)
    plt.close()
    clean = rows[rows["stress"] == "clean"]
    return {
        "available": True,
        "rows": len(rows),
        "feature_rows": len(feature),
        "per_class_rows": len(stress_class),
        "feature_per_class_rows": len(feature_class),
        "variants": sorted(rows["variant"].unique()),
        "clean_balanced_accuracy": clean.groupby("variant")[
            "balanced_accuracy"
        ].mean().to_dict(),
        "maximum_severity_mean_delta": severe.groupby("variant")[
            "balanced_accuracy_delta"
        ].mean().to_dict(),
        "maximum_severity_mean_ece_delta": severe.groupby("variant")[
            "ece_delta"
        ].mean().to_dict(),
        "plots": [stress_plot.name, feature_plot.name],
    }


def _external(study_root: Path, table_root: Path, plot_root: Path) -> dict[str, Any]:
    path = study_root / "external" / "metrics.json"
    if not path.exists():
        return {"available": False}
    valid, reason = validate_run(
        path.parent,
        expected={"run_id": "lifecycle-external-validation-v1"},
    )
    if not valid:
        return {"available": False, "reason": reason}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for variant, variant_payload in payload["zero_target"].items():
        for cohort, metrics in variant_payload["cohorts"].items():
            rows.append({
                "setting": "zero_target",
                "variant": variant,
                "cohort": cohort,
                **{
                    key: value
                    for key, value in metrics.items()
                    if not isinstance(value, (dict, list))
                },
            })
    for cohort, variant_payload in payload.get(
        "unsupervised_target_adaptation", {}
    ).items():
        metrics = variant_payload["metrics"]
        rows.append({
            "setting": "transductive_unlabeled",
            "variant": variant_payload["variant"],
            "cohort": cohort,
            **{
                key: value
                for key, value in metrics.items()
                if not isinstance(value, (dict, list))
            },
        })
    performance = pd.DataFrame(rows)
    calibration = pd.DataFrame(payload["few_shot_target_calibration"])
    performance.to_csv(table_root / "external_performance.csv", index=False)
    calibration.to_csv(table_root / "external_target_calibration.csv", index=False)

    zero = performance[performance["setting"] == "zero_target"]
    pivot = zero.pivot(index="variant", columns="cohort", values="event_auroc")
    pivot.plot.barh(figsize=(10, 6))
    plt.axvline(0.5, color="black", linestyle=":")
    plt.xlim(0, 1)
    plt.xlabel("Event/no-event AUROC")
    plt.ylabel("Model variant")
    plt.title("Zero-target conventional-OTDR transfer")
    plt.tight_layout()
    performance_plot = plot_root / "external_zero_target.png"
    plt.savefig(performance_plot, dpi=180)
    plt.close()

    curve = calibration.groupby(
        ["variant", "cohort", "calibration_groups"]
    )["test_balanced_accuracy"].agg(["mean", "std"]).reset_index()
    plt.figure(figsize=(10, 6))
    for (variant, cohort), part in curve.groupby(["variant", "cohort"]):
        plt.errorbar(
            part["calibration_groups"],
            part["mean"],
            yerr=part["std"].fillna(0),
            marker="o",
            capsize=3,
            label=f"{variant} / {cohort}",
        )
    plt.xscale("log")
    plt.xticks([1, 5, 10, 20], [1, 5, 10, 20])
    plt.ylim(0, 1)
    plt.xlabel("Labeled target calibration groups")
    plt.ylabel("Disjoint target balanced accuracy")
    plt.title("Few-shot target calibration")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    calibration_plot = plot_root / "external_calibration_curve.png"
    plt.savefig(calibration_plot, dpi=180)
    plt.close()

    monotonicity = {}
    for (variant, cohort), part in curve.groupby(["variant", "cohort"]):
        ordered = part.sort_values("calibration_groups")["mean"].to_numpy()
        monotonicity[f"{variant}/{cohort}"] = {
            "nondecreasing": bool(np.all(np.diff(ordered) >= -1e-12)),
            "near_monotonic_with_0_02_tolerance": bool(
                np.all(np.diff(ordered) >= -0.02)
            ),
            "curve": ordered.tolist(),
        }
    return {
        "available": True,
        "task_definition": payload["task_definition"],
        "zero_target_rows": len(zero),
        "transductive_rows": int(
            (performance["setting"] == "transductive_unlabeled").sum()
        ),
        "few_shot_rows": len(calibration),
        "zero_target_event_auroc": nested_two_key_mapping(
            zero.set_index(["variant", "cohort"])["event_auroc"].to_dict()
        ),
        "target_calibration_monotonicity": monotonicity,
        "plots": [performance_plot.name, calibration_plot.name],
    }


def _ablation(study_root: Path, table_root: Path, plot_root: Path) -> dict[str, Any]:
    path = study_root / "ablations" / "kpsc_ablation.json"
    if not path.exists():
        return {"available": False}
    valid, reason = validate_run(
        path.parent,
        expected={"run_id": "lifecycle-kpsc-ablation-v1"},
    )
    if not valid:
        return {"available": False, "reason": reason}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame([
        {
            **{
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            },
            "pair": "-".join(str(value) for value in row["pair"]),
        }
        for row in payload["rows"]
    ])
    rows.to_csv(table_root / "kpsc_ablation.csv", index=False)
    projection = pd.DataFrame([
        {
            **{
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            },
            "pair": "-".join(str(value) for value in row["pair"]),
        }
        for row in payload.get("cfe_projection_adapter", [])
    ])
    if len(projection):
        projection.to_csv(
            table_root / "cfe_projection_adapter.csv", index=False
        )
    oe = rows[rows["score_ablation"] == "finalist"]
    summary = oe.groupby("oe_mode").agg({
        "unknown_recall": "mean",
        "known_fault_acceptance": "mean",
        "normal_far": "mean",
        "constraints_met": "mean",
    })
    plt.figure(figsize=(7, 5))
    for name, row in summary.iterrows():
        plt.scatter(
            row["known_fault_acceptance"],
            row["unknown_recall"],
            s=80,
            label=name,
        )
    plt.axvline(0.95, color="black", linestyle="--")
    plt.xlabel("Known-fault acceptance")
    plt.ylabel("Held-out-fault recall")
    plt.title("Physics outlier-exposure ablation")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plot = plot_root / "kpsc_physics_oe_ablation.png"
    plt.savefig(plot, dpi=180)
    plt.close()
    projection_plot = None
    projection_summary: dict[str, Any] = {}
    if len(projection):
        projection_units = projection.groupby(
            ["pair", "shots", "method"], as_index=False
        ).agg({
            "harmonic_mean": "mean",
            "base_accuracy": "mean",
            "enrolled_accuracy": "mean",
            "forgetting": "mean",
            "retention_ratio": "mean",
        })
        projection_summary = nested_two_key_mapping(
            projection_units.groupby(
                ["method", "shots"]
            ).mean(numeric_only=True).to_dict(orient="index")
        )
        curve = projection_units.groupby(
            ["method", "shots"]
        )["harmonic_mean"].agg(["mean", "std"]).reset_index()
        plt.figure(figsize=(7, 5))
        for method, part in curve.groupby("method"):
            plt.errorbar(
                part["shots"],
                part["mean"],
                yerr=part["std"].fillna(0),
                marker="o",
                capsize=3,
                label=method,
            )
        plt.ylim(0, 1)
        plt.xticks([1, 3, 5])
        plt.xlabel("Shots per novel class")
        plt.ylabel("Seen/enrolled harmonic mean")
        plt.title("Projection-only adaptation diagnostic")
        plt.legend()
        plt.tight_layout()
        projection_plot = plot_root / "cfe_projection_adapter.png"
        plt.savefig(projection_plot, dpi=180)
        plt.close()
    return {
        "available": True,
        "rows": len(rows),
        "representative_pairs_only": payload["representative_pairs_only"],
        "oe_summary": summary.to_dict(orient="index"),
        "plot": plot.name,
        "projection_adapter_rows": len(projection),
        "projection_adapter_summary": projection_summary,
        "projection_adapter_plot": (
            projection_plot.name if projection_plot else None
        ),
    }


def _tabpfn(study_root: Path, table_root: Path) -> dict[str, Any]:
    path = study_root / "baselines" / "tabpfn_v2" / "metrics.json"
    if not path.exists():
        return {"available": False}
    valid, reason = validate_run(
        path.parent,
        expected={"run_id": "tabpfn-v2-representative-pilot"},
    )
    if not valid:
        return {"available": False, "reason": reason}
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = pd.DataFrame([
        {
            **{
                key: value
                for key, value in row.items()
                if not isinstance(value, (dict, list))
            },
            "pair": "-".join(str(value) for value in row["pair"]),
        }
        for row in payload["rows"]
    ])
    rows.to_csv(table_root / "tabpfn_v2.csv", index=False)
    summary = rows.groupby(["regime", "shots"]).agg({
        "harmonic_mean": ["mean", "std"],
        "base_accuracy": "mean",
        "enrolled_accuracy": "mean",
        "inference_seconds": "mean",
    })
    return {
        "available": True,
        "package_version": payload["package_version"],
        "resolved_model_path": payload["resolved_model_path"],
        "weight_hashes": payload["weight_hashes"],
        "rows": len(rows),
        "summary": {
            "/".join(str(value) for value in index): {
                "/".join(column): float(value)
                for column, value in row.items()
            }
            for index, row in summary.iterrows()
        },
    }


def _regime_comparison(
    study_root: Path,
    table_root: Path,
    plot_root: Path,
) -> dict[str, Any]:
    payloads = {}
    for regime in ("full", "trace_only", "summary_only"):
        path = study_root / "tables" / regime / "headline_summary.json"
        if path.exists():
            payloads[regime] = json.loads(path.read_text(encoding="utf-8"))
    if not payloads:
        return {"available": False, "complete": False}
    kpsc_rows, cfe_rows = [], []
    for regime, payload in payloads.items():
        kpsc_rows.append({
            "regime": regime,
            "validated_runs": payload["validated_runs"],
            "complete": payload["complete"],
            **{
                key: payload["kpsc"][key]["mean"]
                for key in (
                    "known_balanced_accuracy",
                    "known_nll",
                    "known_brier",
                    "known_ece",
                    "normal_far",
                    "known_fault_acceptance",
                    "unknown_recall",
                    "worst_fault_recall",
                    "auroc",
                    "oscr",
                )
            },
            "joint_feasibility_rate": payload["kpsc"][
                "joint_feasibility_rate"
            ],
        })
        for shots, values in payload["cfe"].items():
            cfe_rows.append({
                "regime": regime,
                "shots": int(shots),
                **{
                    key: values[key]["mean"]
                    for key in (
                        "harmonic_mean",
                        "base_accuracy",
                        "enrolled_accuracy",
                        "macro_f1",
                        "normal_far_after_enrollment",
                        "forgetting",
                        "retention_ratio",
                    )
                },
            })
    kpsc = pd.DataFrame(kpsc_rows)
    cfe = pd.DataFrame(cfe_rows)
    kpsc.to_csv(table_root / "feature_regime_kpsc.csv", index=False)
    cfe.to_csv(table_root / "feature_regime_cfe.csv", index=False)

    kpsc_plot = None
    cfe_plot = None
    if len(kpsc):
        metrics = (
            "known_balanced_accuracy",
            "unknown_recall",
            "known_fault_acceptance",
            "joint_feasibility_rate",
        )
        kpsc.set_index("regime")[list(metrics)].plot.bar(
            figsize=(10, 5)
        )
        plt.ylim(0, 1)
        plt.ylabel("Rate")
        plt.title("Feature-regime closed/open-world comparison")
        plt.xticks(rotation=0)
        plt.tight_layout()
        kpsc_plot = plot_root / "feature_regime_kpsc.png"
        plt.savefig(kpsc_plot, dpi=180)
        plt.close()
    if len(cfe):
        pivot = cfe.pivot(
            index="shots", columns="regime", values="harmonic_mean"
        )
        pivot.plot(marker="o", figsize=(8, 5))
        plt.ylim(0, 1)
        plt.xticks(sorted(cfe["shots"].unique()))
        plt.xlabel("Shots per enrolled class")
        plt.ylabel("Seen/enrolled harmonic mean")
        plt.title("Feature-regime enrollment comparison")
        plt.tight_layout()
        cfe_plot = plot_root / "feature_regime_cfe.png"
        plt.savefig(cfe_plot, dpi=180)
        plt.close()

    deltas: dict[str, Any] = {}
    if "full" in payloads:
        full_kpsc = kpsc.set_index("regime").loc["full"]
        full_cfe = cfe[cfe["regime"] == "full"].set_index("shots")
        for regime in ("trace_only", "summary_only"):
            if regime not in payloads:
                continue
            target_kpsc = kpsc.set_index("regime").loc[regime]
            target_cfe = cfe[cfe["regime"] == regime].set_index("shots")
            deltas[f"{regime}_minus_full"] = {
                "kpsc": {
                    key: float(target_kpsc[key] - full_kpsc[key])
                    for key in (
                        "known_balanced_accuracy",
                        "unknown_recall",
                        "worst_fault_recall",
                        "normal_far",
                        "known_fault_acceptance",
                        "joint_feasibility_rate",
                    )
                },
                "cfe_harmonic_mean": {
                    str(int(shots)): float(
                        target_cfe.loc[shots, "harmonic_mean"]
                        - full_cfe.loc[shots, "harmonic_mean"]
                    )
                    for shots in target_cfe.index.intersection(full_cfe.index)
                },
            }
    return {
        "available": True,
        "complete": (
            set(payloads) == {"full", "trace_only", "summary_only"}
            and all(payload["complete"] for payload in payloads.values())
        ),
        "regimes": sorted(payloads),
        "deltas": deltas,
        "plots": [
            path.name
            for path in (kpsc_plot, cfe_plot)
            if path is not None
        ],
    }


def synthesize_auxiliary_results(study_root: Path) -> dict[str, Any]:
    table_root = study_root / "tables" / "auxiliary"
    plot_root = study_root / "plots" / "auxiliary"
    table_root.mkdir(parents=True, exist_ok=True)
    plot_root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "source_snapshot": {
            "path": "SOURCE_SNAPSHOT.json",
            "files": len(write_source_snapshot(study_root)["files"]),
        },
        "feature_regimes": _regime_comparison(
            study_root, table_root, plot_root
        ),
        "stress": _stress(study_root, table_root, plot_root),
        "external": _external(study_root, table_root, plot_root),
        "kpsc_ablation": _ablation(study_root, table_root, plot_root),
        "tabpfn": _tabpfn(study_root, table_root),
    }
    result["complete"] = all(
        result[name]["available"]
        for name in ("stress", "external", "kpsc_ablation", "tabpfn")
    ) and result["feature_regimes"]["complete"]
    atomic_json(table_root / "auxiliary_summary.json", result)
    return result
