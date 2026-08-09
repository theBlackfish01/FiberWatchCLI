"""Paired statistics, figures, and gates for the PHI acquisition-shift study."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, f1_score, recall_score
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES
from .morphology_attributes_v3 import _view_indices
from .shift_protocol_v1 import (
    finalize_payload,
    process_memory_snapshot,
    sha256_file,
    verify_payload,
    write_csv,
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _probabilities(rows: Sequence[Mapping[str, object]]) -> np.ndarray:
    return np.asarray([[float(row[f"prob_{name}"]) for name in CLASS_NAMES] for row in rows])


def _macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            np.argmax(probabilities, axis=1),
            labels=np.arange(len(CLASS_NAMES)),
            average="macro",
            zero_division=0,
        )
    )


def _worst_recall(labels: np.ndarray, probabilities: np.ndarray) -> float:
    prediction = np.argmax(probabilities, axis=1)
    recall = recall_score(
        labels,
        prediction,
        labels=np.arange(len(CLASS_NAMES)),
        average=None,
        zero_division=0,
    )
    present = np.unique(labels)
    return float(np.min(recall[present]))


def _confusion_metrics(matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized six-class macro-F1 and present-class worst recall."""
    matrices = np.asarray(matrices, dtype=float)
    diagonal = np.diagonal(matrices, axis1=-2, axis2=-1)
    false_positive = np.sum(matrices, axis=-2) - diagonal
    false_negative = np.sum(matrices, axis=-1) - diagonal
    denominator = 2.0 * diagonal + false_positive + false_negative
    class_f1 = np.divide(
        2.0 * diagonal,
        denominator,
        out=np.zeros_like(diagonal),
        where=denominator > 0,
    )
    macro_f1 = np.mean(class_f1, axis=-1)
    support = np.sum(matrices, axis=-1)
    recall = np.divide(
        diagonal,
        support,
        out=np.full_like(diagonal, np.nan),
        where=support > 0,
    )
    worst = np.nanmin(recall, axis=-1)
    return macro_f1, worst


def _bh(rows: list[dict[str, object]], key: str = "p_one_sided") -> None:
    if not rows:
        return
    values = np.asarray([float(row[key]) for row in rows])
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(rows) / np.arange(1, len(rows) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    for row, value in zip(rows, output, strict=True):
        row["bh_q"] = float(value)


def factorization_statistics(
    prediction_rows: Sequence[Mapping[str, str]], *, draws: int, seed: int
) -> list[dict[str, object]]:
    rng = np.random.default_rng(seed)
    methods = sorted({row["method"] for row in prediction_rows if row["method"] != "baseline"})
    comparisons = []
    groups = [
        ("heldout_date_class_cell", "pooled"),
        ("leave_one_date_out", "pooled"),
        ("cross_era", "january_to_april_may"),
        ("cross_era", "april_may_to_january"),
    ]
    for level, direction in groups:
        family = []
        for method in methods:
            base = [
                row
                for row in prediction_rows
                if row["level"] == level
                and row["method"] == "baseline"
                and (direction == "pooled" or row["fold"] == direction)
            ]
            alternative = [
                row
                for row in prediction_rows
                if row["level"] == level
                and row["method"] == method
                and (direction == "pooled" or row["fold"] == direction)
            ]
            key = lambda row: (row["fold"], row["seed"], row["session_id"])
            base_map, alternative_map = {key(row): row for row in base}, {key(row): row for row in alternative}
            if set(base_map) != set(alternative_map):
                raise ValueError(f"Unpaired factorization rows for {level}/{direction}/{method}")
            ordered = sorted(base_map)
            base = [base_map[item] for item in ordered]
            alternative = [alternative_map[item] for item in ordered]
            labels = np.asarray([int(row["true_label"]) for row in base])
            base_prob = _probabilities(base)
            alt_prob = _probabilities(alternative)
            cluster_values = np.asarray(
                [row["fold"] if level != "cross_era" else row["date_token"] for row in base]
            )
            clusters = np.asarray(sorted(set(cluster_values.tolist())))
            base_prediction = np.argmax(base_prob, axis=1)
            alt_prediction = np.argmax(alt_prob, axis=1)
            base_confusions = np.stack(
                [
                    confusion_matrix(
                        labels[cluster_values == item],
                        base_prediction[cluster_values == item],
                        labels=np.arange(len(CLASS_NAMES)),
                    )
                    for item in clusters
                ]
            )
            alt_confusions = np.stack(
                [
                    confusion_matrix(
                        labels[cluster_values == item],
                        alt_prediction[cluster_values == item],
                        labels=np.arange(len(CLASS_NAMES)),
                    )
                    for item in clusters
                ]
            )
            selected = rng.integers(0, len(clusters), size=(draws, len(clusters)))
            base_f1, base_worst = _confusion_metrics(np.sum(base_confusions[selected], axis=1))
            alt_f1, alt_worst = _confusion_metrics(np.sum(alt_confusions[selected], axis=1))
            boot_f1 = alt_f1 - base_f1
            boot_worst = alt_worst - base_worst
            observed_f1 = _macro_f1(labels, alt_prob) - _macro_f1(labels, base_prob)
            observed_worst = _worst_recall(labels, alt_prob) - _worst_recall(labels, base_prob)
            family.append(
                {
                    "level": level,
                    "direction": direction,
                    "method": method,
                    "baseline": "baseline",
                    "clusters": len(clusters),
                    "sessions": len(labels),
                    "delta_macro_f1": observed_f1,
                    "macro_f1_ci_low": float(np.quantile(boot_f1, 0.025)),
                    "macro_f1_ci_high": float(np.quantile(boot_f1, 0.975)),
                    "p_one_sided": float((1 + np.sum(boot_f1 <= 0)) / (draws + 1)),
                    "delta_worst_recall": observed_worst,
                    "worst_recall_ci_low": float(np.quantile(boot_worst, 0.025)),
                    "worst_recall_ci_high": float(np.quantile(boot_worst, 0.975)),
                    "bootstrap_unit": "date_class_cell" if level == "heldout_date_class_cell" else "date",
                }
            )
        _bh(family)
        comparisons.extend(family)
    return comparisons


def enrollment_statistics(
    episodes: Sequence[Mapping[str, str]], *, draws: int, seed: int
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rng = np.random.default_rng(seed)
    baseline = "sliced_wasserstein_gallery"
    alternatives = sorted({row["method"] for row in episodes if row["method"] != baseline})
    comparisons = []
    class_effects = []
    for direction in sorted({row["direction"] for row in episodes}):
        for shot in sorted({int(row["shot"]) for row in episodes}):
            family = []
            for method in alternatives:
                local_base = {
                    (row["heldout_class"], int(row["draw"])): row
                    for row in episodes
                    if row["direction"] == direction and int(row["shot"]) == shot and row["method"] == baseline
                }
                local_alt = {
                    (row["heldout_class"], int(row["draw"])): row
                    for row in episodes
                    if row["direction"] == direction and int(row["shot"]) == shot and row["method"] == method
                }
                if set(local_base) != set(local_alt):
                    raise ValueError(f"Support episodes are not identical: {direction}/{shot}/{method}")
                per_class = []
                for class_name in CLASS_NAMES:
                    keys = sorted(key for key in local_base if key[0] == class_name)
                    delta_h = np.asarray([float(local_alt[key]["enrollment_h"]) - float(local_base[key]["enrollment_h"]) for key in keys])
                    delta_base = np.asarray([float(local_alt[key]["base_class_accuracy"]) - float(local_base[key]["base_class_accuracy"]) for key in keys])
                    delta_worst_draw = float(np.min([float(local_alt[key]["enrollment_h"]) for key in keys]) - np.min([float(local_base[key]["enrollment_h"]) for key in keys]))
                    per_class.append(float(np.mean(delta_h)))
                    class_effects.append(
                        {
                            "direction": direction,
                            "shot": shot,
                            "method": method,
                            "heldout_class": class_name,
                            "delta_enrollment_h": float(np.mean(delta_h)),
                            "delta_base_class_accuracy": float(np.mean(delta_base)),
                            "delta_worst_draw_enrollment_h": delta_worst_draw,
                        }
                    )
                per_class = np.asarray(per_class)
                choices = rng.integers(0, len(per_class), size=(draws, len(per_class)))
                bootstrap = np.mean(per_class[choices], axis=1)
                observed = float(np.mean(per_class))
                permutations = np.asarray(
                    [np.mean(per_class * np.asarray(signs)) for signs in itertools.product((-1.0, 1.0), repeat=len(per_class))]
                )
                all_keys = sorted(local_base)
                delta_base_all = np.asarray([float(local_alt[key]["base_class_accuracy"]) - float(local_base[key]["base_class_accuracy"]) for key in all_keys])
                delta_worst = []
                for class_name in CLASS_NAMES:
                    keys = [key for key in all_keys if key[0] == class_name]
                    delta_worst.append(
                        np.min([float(local_alt[key]["enrollment_h"]) for key in keys])
                        - np.min([float(local_base[key]["enrollment_h"]) for key in keys])
                    )
                family.append(
                    {
                        "direction": direction,
                        "shot": shot,
                        "method": method,
                        "baseline": baseline,
                        "heldout_class_clusters": len(per_class),
                        "support_episodes": len(all_keys),
                        "delta_enrollment_h": observed,
                        "enrollment_h_ci_low": float(np.quantile(bootstrap, 0.025)),
                        "enrollment_h_ci_high": float(np.quantile(bootstrap, 0.975)),
                        "p_one_sided": float(np.mean(permutations >= observed - 1e-15)),
                        "delta_base_class_accuracy": float(np.mean(delta_base_all)),
                        "delta_worst_draw_enrollment_h": float(np.mean(delta_worst)),
                        "inference": "six-class exact sign-flip; heldout-class bootstrap interval",
                    }
                )
            _bh(family)
            comparisons.extend(family)
    return comparisons, class_effects


def _load_session_projection(
    attributes_path: Path, bundle_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(attributes_path, allow_pickle=False) as source:
        attributes = {key: source[key] for key in source.files}
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    indices = _view_indices(attributes["attribute_names"].astype(str), "morphology_only")
    scaled = StandardScaler().fit_transform(attributes["attributes"][:, indices])
    projection = PCA(n_components=2, random_state=20260809).fit_transform(scaled)
    bundle_sessions = bundle["sessions"].astype(str)
    eras = []
    for session in attributes["sessions"].astype(str):
        eras.append(str(np.unique(bundle["eras"][bundle_sessions == session].astype(str)).item()))
    return projection, attributes["labels"].astype(int), np.asarray(eras)


def _projection_figure(path: Path, projection: np.ndarray, labels: np.ndarray, eras: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for class_id, name in enumerate(CLASS_NAMES):
        mask = labels == class_id
        axes[0].scatter(projection[mask, 0], projection[mask, 1], s=17, alpha=0.7, label=name)
    axes[0].legend(fontsize=8, ncol=2)
    axes[0].set_title("Position-free morphology, coloured by event")
    for era, marker, color in (("january", "o", "#4575b4"), ("april_may", "^", "#d73027")):
        mask = eras == era
        axes[1].scatter(projection[mask, 0], projection[mask, 1], s=18, alpha=0.7, label=era, marker=marker, color=color)
    axes[1].legend()
    axes[1].set_title("The same projection, coloured by acquisition era")
    for ax in axes:
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _factorization_figure(path: Path, summaries: Sequence[Mapping[str, object]]) -> None:
    methods = ["baseline", "source_nuisance_projection_rank4", "target_unlabelled_mean_alignment", "target_unlabelled_coral"]
    directions = ["january_to_april_may", "april_may_to_january"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    x = np.arange(len(methods))
    for axis, direction in zip(axes, directions, strict=True):
        local = {row["method"]: row for row in summaries if row["level"] == "cross_era" and row["direction"] == direction}
        f1 = [float(local[method]["macro_f1_six_classes"]) for method in methods]
        worst = [float(local[method]["worst_observed_class_recall"]) for method in methods]
        axis.bar(x - 0.18, f1, 0.36, label="macro-F1")
        axis.bar(x + 0.18, worst, 0.36, label="worst recall")
        axis.set_xticks(x, ["baseline", "nuisance\nprojection", "mean\nalignment", "CORAL"], rotation=10)
        axis.set_title(direction.replace("_", " ").title())
        axis.set_ylim(0, 0.8)
    axes[0].set_ylabel("session score")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _confusion_figure(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    methods = ["baseline", "target_unlabelled_mean_alignment"]
    directions = ["january_to_april_may", "april_may_to_january"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for row_index, direction in enumerate(directions):
        for column, method in enumerate(methods):
            local = [row for row in rows if row["level"] == "cross_era" and row["fold"] == direction and row["method"] == method]
            matrix = confusion_matrix(
                [int(row["true_label"]) for row in local],
                [int(row["predicted_label"]) for row in local],
                labels=np.arange(len(CLASS_NAMES)),
                normalize="true",
            )
            axis = axes[row_index, column]
            image = axis.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
            axis.set_xticks(range(6), CLASS_NAMES, rotation=40, ha="right", fontsize=8)
            axis.set_yticks(range(6), CLASS_NAMES, fontsize=8)
            axis.set_title(f"{direction}\n{method}")
            for i in range(6):
                for j in range(6):
                    axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=7, color="white" if matrix[i, j] > 0.55 else "black")
    fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, label="row-normalized recall")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _calibration_risk_figure(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    directions = ["january_to_april_may", "april_may_to_january"]
    methods = ["baseline", "target_unlabelled_mean_alignment", "target_unlabelled_coral"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row_index, direction in enumerate(directions):
        for method in methods:
            local = [row for row in rows if row["level"] == "cross_era" and row["fold"] == direction and row["method"] == method]
            labels = np.asarray([int(row["true_label"]) for row in local])
            probability = _probabilities(local)
            confidence = np.max(probability, axis=1)
            correct = np.argmax(probability, axis=1) == labels
            bin_conf, bin_acc = [], []
            for left, right in zip(np.linspace(0, 0.9, 10), np.linspace(0.1, 1.0, 10), strict=True):
                mask = (confidence > left) & (confidence <= right)
                if np.any(mask):
                    bin_conf.append(float(np.mean(confidence[mask])))
                    bin_acc.append(float(np.mean(correct[mask])))
            axes[row_index, 0].plot(bin_conf, bin_acc, marker="o", label=method)
            order = np.argsort(-confidence)
            risk = 1.0 - np.cumsum(correct[order]) / np.arange(1, len(correct) + 1)
            coverage = np.arange(1, len(correct) + 1) / len(correct)
            axes[row_index, 1].plot(coverage, risk, label=method)
        axes[row_index, 0].plot([0, 1], [0, 1], "--", color="gray")
        axes[row_index, 0].set_title(f"Reliability: {direction}")
        axes[row_index, 0].set_xlabel("confidence")
        axes[row_index, 0].set_ylabel("accuracy")
        axes[row_index, 1].set_title(f"Risk-coverage: {direction}")
        axes[row_index, 1].set_xlabel("coverage")
        axes[row_index, 1].set_ylabel("selective risk")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _enrollment_figures(path: Path, distribution_path: Path, episodes: Sequence[Mapping[str, str]]) -> None:
    methods = sorted({row["method"] for row in episodes})
    directions = sorted({row["direction"] for row in episodes})
    colors = dict(zip(methods, plt.cm.tab10(np.linspace(0, 1, len(methods))), strict=True))
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, direction in zip(axes, directions, strict=True):
        for method in methods:
            means, lows, highs = [], [], []
            for shot in (1, 3, 5):
                local = np.asarray([float(row["enrollment_h"]) for row in episodes if row["direction"] == direction and row["method"] == method and int(row["shot"]) == shot])
                means.append(float(np.mean(local)))
                lows.append(float(np.quantile(local, 0.1)))
                highs.append(float(np.quantile(local, 0.9)))
            axis.plot((1, 3, 5), means, marker="o", label=method, color=colors[method])
            axis.fill_between((1, 3, 5), lows, highs, alpha=0.12, color=colors[method])
        axis.set_title(direction.replace("_", " ").title())
        axis.set_xlabel("support sessions")
        axis.set_xticks((1, 3, 5))
    axes[0].set_ylabel("Enrollment-H (10th–90th draw band)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
    for axis, direction in zip(axes, directions, strict=True):
        values = [
            [float(row["enrollment_h"]) for row in episodes if row["direction"] == direction and row["method"] == method and int(row["shot"]) == 5]
            for method in methods
        ]
        axis.boxplot(values, tick_labels=[name.replace("_", "\n") for name in methods], showfliers=False)
        axis.set_title(direction.replace("_", " ").title())
        axis.tick_params(axis="x", labelsize=7)
    axes[0].set_ylabel("five-shot Enrollment-H across class × support draws")
    fig.tight_layout()
    fig.savefig(distribution_path, dpi=180)
    plt.close(fig)


def _ladder_figure(path: Path, ladder_rows: Sequence[Mapping[str, str]]) -> None:
    representations = list(dict.fromkeys(row["representation"] for row in ladder_rows))
    levels = list(dict.fromkeys(row["level_label"] for row in ladder_rows))
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for representation in representations:
        local = {row["level_label"]: row for row in ladder_rows if row["representation"] == representation}
        ax.plot(range(len(levels)), [float(local[level]["session_macro_f1"]) for level in levels], marker="o", label=representation)
    ax.axvspan(len(levels) - 2.5, len(levels) - 0.5, color="#d73027", alpha=0.08, label="complete era separation")
    ax.set_xticks(range(len(levels)), levels, rotation=25, ha="right")
    ax.set_ylabel("session macro-F1")
    ax.set_ylim(0.3, 1.02)
    ax.set_title("Revalidated acquisition-safe evaluation ladder")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _date_cell_figure(path: Path, rows: Sequence[Mapping[str, str]]) -> None:
    local = [row for row in rows if row["representation"] == "registered_position_difference_dynamics"]
    dates = sorted({row["date_token"] for row in local})
    matrix = np.full((len(dates), len(CLASS_NAMES)), np.nan)
    for row in local:
        matrix[dates.index(row["date_token"]), CLASS_NAMES.index(row["class_name"])] = float(row["accuracy"])
    fig, ax = plt.subplots(figsize=(9, max(6, len(dates) * 0.36)))
    image = ax.imshow(matrix, vmin=0, vmax=1, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(6), CLASS_NAMES, rotation=35, ha="right")
    ax.set_yticks(range(len(dates)), dates)
    ax.set_title("Registered-view held-out date × class accuracy")
    fig.colorbar(image, ax=ax, label="session accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(
    *,
    forensics_result: Path,
    factorization_result: Path,
    factorization_predictions: Path,
    robust_result: Path,
    robust_episodes: Path,
    session_attributes: Path,
    morphology_bundle: Path,
    ladder_summary: Path,
    ladder_cells: Path,
    output_dir: Path,
    draws: int = 5000,
) -> dict[str, object]:
    started = time.perf_counter()
    forensics = verify_payload(forensics_result)
    factorization = verify_payload(factorization_result)
    robust = verify_payload(robust_result)
    factor_rows = _read_csv(factorization_predictions)
    episode_rows = _read_csv(robust_episodes)
    factor_stats = factorization_statistics(factor_rows, draws=draws, seed=20260809)
    enrollment_stats, class_effects = enrollment_statistics(episode_rows, draws=draws, seed=20260810)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "factorization_paired_statistics.csv", factor_stats)
    write_csv(output_dir / "enrollment_paired_statistics.csv", enrollment_stats)
    write_csv(output_dir / "enrollment_heldout_class_effects.csv", class_effects)
    projection, labels, eras = _load_session_projection(session_attributes, morphology_bundle)
    _projection_figure(output_dir / "acquisition_morphology_projection.png", projection, labels, eras)
    _factorization_figure(output_dir / "factorization_cross_era.png", factorization["summaries"])
    _confusion_figure(output_dir / "cross_era_confusion_matrices.png", factor_rows)
    _calibration_risk_figure(output_dir / "calibration_risk_coverage.png", factor_rows)
    _enrollment_figures(output_dir / "enrollment_performance.png", output_dir / "support_draw_distributions.png", episode_rows)
    _ladder_figure(output_dir / "updated_evaluation_ladder.png", _read_csv(ladder_summary))
    _date_cell_figure(output_dir / "date_class_failure_map.png", _read_csv(ladder_cells))
    graph = factorization["graph_factorization"]
    gates = {
        "morphology_primitives": {
            "decision": "completed_in_v3; stop further expansion in shift_v1",
            "reason": "The verified 20-attribute/114-position-free representation already supplies interpretable primitives. The new factorization suite found no symmetric acquisition-removal method to justify another primitive search.",
            "evidence": "phi_research_v3 morphology_attributes plus shift_v1 forensics and factorization",
        },
        "modern_neural_control": {
            "decision": "stopped",
            "reason": "Prior CUDA Deep Sets, attention, CNN, TCN, and Siamese controls fit source data but did not solve hard transfer; no classical factorization was promoted and the new robust methods exposed a data/identifiability limitation rather than a capacity deficit.",
            "cuda_available": True,
        },
        "chronological_lifecycle": {
            "decision": "stopped_as_primary_experiment",
            "reason": "Only 39 of 126 date-class cells are observed and 11 of 21 dates contain one class. A prequential score would conflate class arrival/absence with drift and cannot support a defensible general lifecycle claim.",
            "dates": len(graph["date_degrees"]),
            "weak_single_class_dates": len(graph["weak_date_anchors"]),
        },
        "factorization_promotion": {
            "decision": "none promoted",
            "reason": "CORAL's forward gain is asymmetric; mean alignment loses forward worst-class recall; source residualization collapses a class; nuisance projection is negligible.",
        },
        "robust_enrollment_promotion": {
            "decision": "retain sliced-Wasserstein gallery baseline",
            "reason": "Trimmed transport did not improve mean Enrollment-H. Robust barycentres and consensus weighting gain held-out recall in the hard direction by sacrificing more than 0.03 base-class accuracy.",
        },
    }
    (output_dir / "gate_decisions.json").write_text(json.dumps(gates, indent=2, sort_keys=True), encoding="utf-8")
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "PHI-OTDR shift-v1 paired analysis and gates",
        "evidence_status": "retrospective analysis",
        "input_payloads": {
            "forensics": forensics["payload_sha256"],
            "factorization": factorization["payload_sha256"],
            "robust_enrollment": robust["payload_sha256"],
        },
        "bootstrap_draws": draws,
        "factorization_comparison_count": len(factor_stats),
        "enrollment_comparison_count": len(enrollment_stats),
        "gate_decisions": gates,
        "output_hashes": {
            path.name: sha256_file(path)
            for path in sorted(output_dir.iterdir())
            if path.is_file() and path.name != "shift_analysis_results.json"
        },
        "elapsed_seconds": time.perf_counter() - started,
        "process_memory": process_memory_snapshot(),
    }
    return finalize_payload(payload, output_dir / "shift_analysis_results.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--forensics-result", type=Path, required=True)
    parser.add_argument("--factorization-result", type=Path, required=True)
    parser.add_argument("--factorization-predictions", type=Path, required=True)
    parser.add_argument("--robust-result", type=Path, required=True)
    parser.add_argument("--robust-episodes", type=Path, required=True)
    parser.add_argument("--session-attributes", type=Path, required=True)
    parser.add_argument("--morphology-bundle", type=Path, required=True)
    parser.add_argument("--ladder-summary", type=Path, required=True)
    parser.add_argument("--ladder-cells", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=5000)
    args = parser.parse_args()
    result = run(
        forensics_result=args.forensics_result,
        factorization_result=args.factorization_result,
        factorization_predictions=args.factorization_predictions,
        robust_result=args.robust_result,
        robust_episodes=args.robust_episodes,
        session_attributes=args.session_attributes,
        morphology_bundle=args.morphology_bundle,
        ladder_summary=args.ladder_summary,
        ladder_cells=args.ladder_cells,
        output_dir=args.output_dir,
        draws=args.draws,
    )
    print(json.dumps({"payload_sha256": result["payload_sha256"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
