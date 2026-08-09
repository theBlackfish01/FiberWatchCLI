"""Retrospective forensic analysis of PHI-OTDR acquisition failures."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.stats import ttest_ind
from sklearn.preprocessing import RobustScaler, StandardScaler

from .data_contract import CLASS_NAMES
from .shift_protocol_v1 import (
    finalize_payload,
    load_locked_config,
    process_memory_snapshot,
    sha256_file,
    verify_payload,
    write_csv,
)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: source[key] for key in source.files}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _session_rows(bundle: Mapping[str, np.ndarray]) -> dict[str, dict[str, object]]:
    sessions = bundle["sessions"].astype(str)
    rows: dict[str, dict[str, object]] = {}
    for session in sorted(set(sessions.tolist())):
        indices = np.flatnonzero(sessions == session)
        rows[session] = {
            "session_id": session,
            "label": int(np.unique(bundle["labels"][indices]).item()),
            "class_name": CLASS_NAMES[int(np.unique(bundle["labels"][indices]).item())],
            "date": str(np.unique(bundle["date_tokens"][indices].astype(str)).item()),
            "era": str(np.unique(bundle["eras"][indices].astype(str)).item()),
            "source": str(np.unique(bundle["source_tokens"][indices].astype(str)).item()),
            "indices": indices,
            "window_count": len(indices),
        }
    return rows


def aggregate_session_means(
    bundle: Mapping[str, np.ndarray], rows: Mapping[str, Mapping[str, object]]
) -> tuple[np.ndarray, np.ndarray]:
    sessions = np.asarray(sorted(rows))
    features = np.stack(
        [np.mean(bundle["features"][rows[session]["indices"]], axis=0) for session in sessions]
    ).astype(np.float64)
    return sessions, features


def _bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=float)
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    adjusted = ranked * len(ranked) / np.arange(1, len(ranked) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.minimum(adjusted, 1.0)
    return output


def _effect_rows(
    target: np.ndarray,
    reference: np.ndarray,
    feature_names: np.ndarray,
    feature_groups: Mapping[str, Sequence[str]],
    *,
    rng: np.random.Generator,
    draws: int,
) -> list[dict[str, object]]:
    target_sd = np.std(target, axis=0, ddof=1) if len(target) > 1 else np.zeros(target.shape[1])
    reference_sd = (
        np.std(reference, axis=0, ddof=1) if len(reference) > 1 else np.zeros(reference.shape[1])
    )
    denominator = np.sqrt((target_sd**2 + reference_sd**2) / 2.0)
    denominator = np.where(denominator > 1e-12, denominator, 1.0)
    effect = (np.mean(target, axis=0) - np.mean(reference, axis=0)) / denominator
    target_draw = rng.integers(0, len(target), size=(draws, len(target)))
    reference_draw = rng.integers(0, len(reference), size=(draws, len(reference)))
    boot = (
        np.mean(target[target_draw], axis=1) - np.mean(reference[reference_draw], axis=1)
    ) / denominator
    low, high = np.quantile(boot, (0.025, 0.975), axis=0)
    pvalues = np.asarray(
        [
            ttest_ind(target[:, index], reference[:, index], equal_var=False).pvalue
            for index in range(target.shape[1])
        ],
        dtype=float,
    )
    pvalues = np.nan_to_num(pvalues, nan=1.0)
    adjusted = _bh_adjust(pvalues)
    rows = []
    for index, name in enumerate(feature_names.astype(str)):
        groups = [
            group
            for group, patterns in feature_groups.items()
            if any(pattern in name for pattern in patterns)
        ]
        rows.append(
            {
                "feature": name,
                "feature_group": "+".join(groups) if groups else "other",
                "target_mean": float(np.mean(target[:, index])),
                "reference_mean": float(np.mean(reference[:, index])),
                "standardized_effect": float(effect[index]),
                "ci_low": float(low[index]),
                "ci_high": float(high[index]),
                "welch_p": float(pvalues[index]),
                "bh_q": float(adjusted[index]),
            }
        )
    return rows


def _median_cross_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.median(cdist(left, right, metric="euclidean")))


def _cohort_medoid(indices: np.ndarray, scaled_features: np.ndarray) -> int:
    local = scaled_features[indices]
    distances = cdist(local, local)
    return int(indices[int(np.argmin(np.median(distances, axis=1)))])


def _load_raw(path: Path) -> np.ndarray:
    payload = sio.loadmat(path.as_posix())
    keys = [key for key in payload if not key.startswith("__")]
    array = np.asarray(payload["data"] if "data" in payload else payload[keys[0]], dtype=float)
    if array.shape != (10000, 12):
        raise ValueError(f"Unexpected raw array shape {array.shape}: {path}")
    return array


def _raw_summary(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    delta = np.diff(array, axis=0)
    usable = delta[: (len(delta) // 100) * 100]
    heatmap = np.mean(np.abs(usable.reshape(-1, 100, 12)), axis=1)
    scale = np.median(heatmap, axis=0, keepdims=True)
    heatmap = np.log1p(heatmap / np.maximum(scale, 1e-9))
    profile = np.sqrt(np.mean(delta**2, axis=0))
    profile /= max(float(np.sum(profile)), 1e-12)
    return heatmap, profile


def _representative_figure(
    path: Path,
    selected: Sequence[tuple[str, str, Path]],
) -> list[dict[str, object]]:
    fig, axes = plt.subplots(len(selected), 2, figsize=(12, 2.6 * len(selected)))
    rows = []
    for row_index, (cohort, session, raw_path) in enumerate(selected):
        array = _load_raw(raw_path)
        heatmap, profile = _raw_summary(array)
        axes[row_index, 0].imshow(heatmap.T, aspect="auto", origin="lower", cmap="magma")
        axes[row_index, 0].set_title(f"{cohort}: {session}\nmedian-index window")
        axes[row_index, 0].set_ylabel("channel")
        axes[row_index, 1].plot(np.arange(12), profile, marker="o")
        axes[row_index, 1].set_ylim(0, max(0.25, float(np.max(profile)) * 1.15))
        axes[row_index, 1].set_title("normalized temporal-difference RMS")
        axes[row_index, 1].set_xlabel("channel")
        rows.append(
            {
                "cohort": cohort,
                "session_id": session,
                "raw_file": raw_path.name,
                "global_mean": float(np.mean(array)),
                "global_std": float(np.std(array)),
                "delta_rms": float(np.sqrt(np.mean(np.diff(array, axis=0) ** 2))),
                "peak_channel": int(np.argmax(profile)),
                "peak_share": float(np.max(profile)),
            }
        )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return rows


def _feature_group_figure(path: Path, summaries: list[dict[str, object]]) -> None:
    cells = list(dict.fromkeys(str(row["cell"]) for row in summaries))
    groups = list(dict.fromkeys(str(row["feature_group"]) for row in summaries))
    matrix = np.full((len(cells), len(groups)), np.nan)
    for row in summaries:
        matrix[cells.index(str(row["cell"])), groups.index(str(row["feature_group"]))] = float(
            row["median_absolute_effect"]
        )
    fig, ax = plt.subplots(figsize=(max(8, len(groups) * 1.2), max(4, len(cells) * 0.65)))
    image = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(groups)), groups, rotation=35, ha="right")
    ax.set_yticks(range(len(cells)), cells)
    ax.set_title("Median absolute standardized shift vs same-class other dates")
    fig.colorbar(image, ax=ax, label="|standardized effect|")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def run(
    *,
    bundle_path: Path,
    ladder_result_path: Path,
    ladder_predictions_path: Path,
    taxonomy_path: Path,
    data_root: Path,
    config_path: Path,
    config_hash_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    started = time.perf_counter()
    config, config_hash = load_locked_config(config_path, config_hash_path)
    if sha256_file(bundle_path) != config["morphology_bundle_sha256"]:
        raise ValueError("Morphology bundle hash mismatch")
    ladder = verify_payload(ladder_result_path)
    if ladder["payload_sha256"] != config["ladder_payload_sha256"]:
        raise ValueError("Evaluation ladder payload mismatch")
    bundle = _load_npz(bundle_path)
    rows = _session_rows(bundle)
    sessions, features = aggregate_session_means(bundle, rows)
    session_index = {session: index for index, session in enumerate(sessions)}
    names = bundle["feature_names"].astype(str)
    robust = RobustScaler(quantile_range=(10, 90)).fit_transform(features)
    predictions = _read_csv(ladder_predictions_path)
    taxonomy = {row["session_id"]: row for row in _read_csv(taxonomy_path)}
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(config["seed"]))
    effect_output = []
    group_summaries = []
    distance_rows = []
    neighbor_rows = []
    confusion_rows = []
    cell_diagnoses = []
    for date, class_name in config["target_cells"]:
        class_id = CLASS_NAMES.index(class_name)
        target_indices = np.asarray(
            [i for i, session in enumerate(sessions) if rows[session]["date"] == date and rows[session]["label"] == class_id]
        )
        same_class = np.asarray(
            [i for i, session in enumerate(sessions) if rows[session]["date"] != date and rows[session]["label"] == class_id]
        )
        same_date = np.asarray(
            [i for i, session in enumerate(sessions) if rows[session]["date"] == date and rows[session]["label"] != class_id]
        )
        other_era_same_class = np.asarray(
            [
                i
                for i, session in enumerate(sessions)
                if rows[session]["label"] == class_id
                and rows[session]["era"] != rows[sessions[target_indices[0]]]["era"]
            ]
        )
        if not len(target_indices) or not len(same_class):
            raise ValueError(f"Insufficient diagnostic anchors for {date}/{class_name}")
        effects = _effect_rows(
            features[target_indices],
            features[same_class],
            names,
            config["feature_groups"],
            rng=rng,
            draws=int(config["bootstrap_draws"]),
        )
        cell = f"{date}__{class_name}"
        for effect in effects:
            effect_output.append({"cell": cell, "reference": "same_class_other_dates", **effect})
        by_group: dict[str, list[dict[str, object]]] = defaultdict(list)
        for effect in effects:
            for group in str(effect["feature_group"]).split("+"):
                by_group[group].append(effect)
        for group, group_rows in sorted(by_group.items()):
            values = np.abs([float(row["standardized_effect"]) for row in group_rows])
            summary = {
                "cell": cell,
                "feature_group": group,
                "median_absolute_effect": float(np.median(values)),
                "maximum_absolute_effect": float(np.max(values)),
                "bh_significant_features": int(sum(float(row["bh_q"]) < 0.05 for row in group_rows)),
                "feature_count": len(group_rows),
            }
            group_summaries.append(summary)
        distance_rows.extend(
            [
                {
                    "cell": cell,
                    "comparison": "within_target_cell",
                    "median_distance": _median_cross_distance(robust[target_indices], robust[target_indices]),
                    "estimable": True,
                },
                {
                    "cell": cell,
                    "comparison": "same_class_other_dates",
                    "median_distance": _median_cross_distance(robust[target_indices], robust[same_class]),
                    "estimable": True,
                },
                {
                    "cell": cell,
                    "comparison": "same_date_other_classes",
                    "median_distance": (
                        _median_cross_distance(robust[target_indices], robust[same_date])
                        if len(same_date)
                        else ""
                    ),
                    "estimable": bool(len(same_date)),
                },
                {
                    "cell": cell,
                    "comparison": "other_era_same_class",
                    "median_distance": (
                        _median_cross_distance(robust[target_indices], robust[other_era_same_class])
                        if len(other_era_same_class)
                        else ""
                    ),
                    "estimable": bool(len(other_era_same_class)),
                },
            ]
        )
        candidate_indices = np.asarray([i for i in range(len(sessions)) if i not in set(target_indices)])
        scaler = StandardScaler().fit(features[candidate_indices])
        candidate_scaled = scaler.transform(features[candidate_indices])
        target_scaled = scaler.transform(features[target_indices])
        nearest = np.argsort(cdist(target_scaled, candidate_scaled), axis=1)[:, :5]
        neighbor_labels = Counter()
        for local_target, local_neighbors in enumerate(nearest):
            for rank, local_neighbor in enumerate(local_neighbors, start=1):
                target_index = target_indices[local_target]
                neighbor_index = candidate_indices[local_neighbor]
                neighbor_labels[str(rows[sessions[neighbor_index]]["class_name"])] += rank == 1
                neighbor_rows.append(
                    {
                        "cell": cell,
                        "query_session": sessions[target_index],
                        "rank": rank,
                        "neighbor_session": sessions[neighbor_index],
                        "neighbor_class": rows[sessions[neighbor_index]]["class_name"],
                        "neighbor_date": rows[sessions[neighbor_index]]["date"],
                        "neighbor_era": rows[sessions[neighbor_index]]["era"],
                        "distance": float(cdist(target_scaled[[local_target]], candidate_scaled[[local_neighbor]])[0, 0]),
                    }
                )
        fold_predictions = [
            row
            for row in predictions
            if row["level"] == "date_class_cell"
            and row["representation"] == "registered_position_difference_dynamics"
            and row["fold"] == cell
        ]
        prediction_protocol = "heldout_date_class_cell"
        if not fold_predictions:
            fold_predictions = [
                row
                for row in predictions
                if row["level"] == "leave_one_date_out"
                and row["representation"] == "registered_position_difference_dynamics"
                and row["fold"] == date
                and row["true_class"] == class_name
            ]
            prediction_protocol = "leave_one_date_out"
        available_seeds = sorted({int(row["seed"]) for row in fold_predictions})
        if not available_seeds:
            raise ValueError(f"No hard-protocol predictions for {cell}")
        fold_predictions = [row for row in fold_predictions if int(row["seed"]) == available_seeds[0]]
        predicted_counts = Counter(row["predicted_class"] for row in fold_predictions)
        for predicted, count in sorted(predicted_counts.items()):
            confusion_rows.append(
                {"cell": cell, "predicted_class": predicted, "session_count": count, "fraction": count / len(fold_predictions)}
            )
        top_effects = sorted(effects, key=lambda row: abs(float(row["standardized_effect"])), reverse=True)[:10]
        subtype_counts = Counter(
            str(taxonomy.get(sessions[index], {}).get("spatial_contact", "unknown"))
            + "/"
            + str(taxonomy.get(sessions[index], {}).get("speed", "unknown"))
            for index in target_indices
        )
        cell_diagnoses.append(
            {
                "cell": cell,
                "session_count": len(target_indices),
                "prediction_protocol": prediction_protocol,
                "prediction_seed": available_seeds[0],
                "registered_accuracy": float(np.mean([row["predicted_class"] == class_name for row in fold_predictions])),
                "predicted_as": dict(predicted_counts),
                "nearest_neighbor_top1_classes": dict(neighbor_labels),
                "audit_only_subtypes": dict(subtype_counts),
                "largest_feature_effects": [
                    {
                        "feature": row["feature"],
                        "effect": row["standardized_effect"],
                        "ci": [row["ci_low"], row["ci_high"]],
                        "q": row["bh_q"],
                    }
                    for row in top_effects
                ],
            }
        )
    write_csv(output_dir / "feature_effects.csv", effect_output)
    write_csv(output_dir / "feature_group_effects.csv", group_summaries)
    write_csv(output_dir / "cohort_distances.csv", distance_rows)
    write_csv(output_dir / "nearest_neighbors.csv", neighbor_rows)
    write_csv(output_dir / "failure_confusions.csv", confusion_rows)
    _feature_group_figure(output_dir / "feature_group_shift_heatmap.png", group_summaries)

    # Four algorithmically selected cohorts make the principal failure visually comparable.
    principal_date, principal_class = config["target_cells"][0]
    principal_id = CLASS_NAMES.index(principal_class)
    cohorts = {
        "220517 knocking": np.asarray([i for i, s in enumerate(sessions) if rows[s]["date"] == principal_date and rows[s]["label"] == principal_id]),
        "220517 other classes": np.asarray([i for i, s in enumerate(sessions) if rows[s]["date"] == principal_date and rows[s]["label"] != principal_id]),
        "Apr-May knocking other dates": np.asarray([i for i, s in enumerate(sessions) if rows[s]["era"] == "april_may" and rows[s]["date"] != principal_date and rows[s]["label"] == principal_id]),
        "January knocking": np.asarray([i for i, s in enumerate(sessions) if rows[s]["era"] == "january" and rows[s]["label"] == principal_id]),
    }
    selected = []
    for cohort, indices in cohorts.items():
        chosen = _cohort_medoid(indices, robust)
        session = sessions[chosen]
        window_indices = rows[session]["indices"]
        ordered = window_indices[np.argsort(bundle["window_ids"][window_indices])]
        window_index = int(ordered[len(ordered) // 2])
        selected.append((cohort, session, data_root / str(bundle["rel_paths"][window_index])))
    representative_rows = _representative_figure(output_dir / "matched_raw_representatives.png", selected)
    write_csv(output_dir / "representative_sessions.csv", representative_rows)

    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": config["protocol_name"],
        "evidence_status": config["evidence_status"],
        "config_sha256": config_hash,
        "dataset_fingerprint_sha256": config["dataset_fingerprint_sha256"],
        "input_hashes": {
            "morphology_bundle_sha256": sha256_file(bundle_path),
            "ladder_payload_sha256": ladder["payload_sha256"],
            "ladder_predictions_sha256": sha256_file(ladder_predictions_path),
            "taxonomy_sha256": sha256_file(taxonomy_path),
        },
        "session_count": len(sessions),
        "cell_diagnoses": cell_diagnoses,
        "representative_rule": config["representative_rule"],
        "outputs": {},
        "limitations": [
            "Target cells were selected after inspection of retrospective ladder outcomes.",
            "Date, source token, subtype, operator, hardware, environment, and fiber effects are not independently balanced.",
            "Standardized feature effects describe association and cannot identify a physical cause by themselves.",
            "Nearest-neighbour evidence can distinguish atypicality from isolation but cannot prove a label error.",
        ],
        "elapsed_seconds": time.perf_counter() - started,
        "process_memory": process_memory_snapshot(),
    }
    for path in sorted(output_dir.iterdir()):
        if path.is_file() and path.name != "forensics_results.json":
            payload["outputs"][path.name] = sha256_file(path)
    return finalize_payload(payload, output_dir / "forensics_results.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--ladder-result", type=Path, required=True)
    parser.add_argument("--ladder-predictions", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        bundle_path=args.bundle,
        ladder_result_path=args.ladder_result,
        ladder_predictions_path=args.ladder_predictions,
        taxonomy_path=args.taxonomy,
        data_root=args.data_root,
        config_path=args.config,
        config_hash_path=args.config_hash,
        output_dir=args.output_dir,
    )
    print(json.dumps({"payload_sha256": result["payload_sha256"], "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
