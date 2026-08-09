"""Paired statistical analysis of PHI-OTDR v3 morphology-before-name results."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from .data_contract import CLASS_NAMES, canonical_json_hash
from .enrollment_analysis_v3 import (
    METRICS,
    _method_rows,
    bh_qvalues,
    paired_cluster_comparison,
)


ATTRIBUTE_METHODS = (
    "attribute_prototype_morphology_only",
    "attribute_prototype_morphology_plus_position",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = str(payload.pop("payload_sha256"))
    if stored != canonical_json_hash(payload):
        raise ValueError(f"Payload hash mismatch: {path}")
    payload["payload_sha256"] = stored
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _prediction_metrics(labels: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    size = len(CLASS_NAMES)
    matrix = np.bincount(labels * size + prediction, minlength=size * size).reshape(size, size)
    true_count = np.sum(matrix, axis=1)
    predicted_count = np.sum(matrix, axis=0)
    true_positive = np.diag(matrix).astype(np.float64)
    recall = np.divide(true_positive, true_count, out=np.zeros(size), where=true_count > 0)
    precision = np.divide(
        true_positive, predicted_count, out=np.zeros(size), where=predicted_count > 0
    )
    class_f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros(size),
        where=(precision + recall) > 0,
    )
    return {
        "macro_f1": float(np.mean(class_f1)),
        "balanced_accuracy": float(np.mean(recall)),
        "worst_class_recall": float(np.min(recall)),
    }


def stratified_paired_bootstrap(
    labels: np.ndarray,
    left: np.ndarray,
    right: np.ndarray,
    *,
    metric: str,
    seed: int,
    draws: int = 5_000,
) -> dict[str, object]:
    labels = np.asarray(labels, dtype=np.int64)
    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    if not (labels.shape == left.shape == right.shape) or set(np.unique(labels)) != set(range(6)):
        raise ValueError("Paired predictions must contain all six classes")
    observed = _prediction_metrics(labels, left)[metric] - _prediction_metrics(labels, right)[metric]
    class_indices = [np.flatnonzero(labels == class_id) for class_id in range(6)]
    rng = np.random.default_rng(seed)
    effects = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        effects[draw] = (
            _prediction_metrics(labels[sampled], left[sampled])[metric]
            - _prediction_metrics(labels[sampled], right[sampled])[metric]
        )
    probability_nonpositive = float(np.mean(effects <= 0.0))
    probability_nonnegative = float(np.mean(effects >= 0.0))
    return {
        "metric": metric,
        "observed_delta": float(observed),
        "ci95_low": float(np.quantile(effects, 0.025)),
        "ci95_high": float(np.quantile(effects, 0.975)),
        "bootstrap_two_sided_p": min(1.0, 2.0 * min(probability_nonpositive, probability_nonnegative)),
        "bootstrap_probability_delta_le_zero": probability_nonpositive,
        "sessions": int(len(labels)),
        "draws": draws,
        "resampling": "paired within-class session bootstrap preserving class counts",
    }


def _prediction_maps(
    attribute_classification: list[dict[str, str]],
    attribute_retrieval: list[dict[str, str]],
    spatial: list[dict[str, str]],
) -> dict[str, dict[tuple[str, str], tuple[int, int]]]:
    output: dict[str, dict[tuple[str, str], tuple[int, int]]] = defaultdict(dict)
    for row in attribute_classification:
        name = f"attribute_{row['view']}"
        output[name][(row["direction"], row["session_id"])] = (
            int(row["true_label"]), int(row["predicted_label"])
        )
    for row in attribute_retrieval:
        name = f"retrieval_{row['view']}"
        output[name][(row["direction"], row["session_id"])] = (
            int(row["true_label"]), int(row["predicted_label"])
        )
    for row in spatial:
        key = (row["view"], row["estimator"], row["ablation"], row["model"])
        if key == ("registered_position", "temporal_difference_energy", "dynamics", "logistic"):
            name = "spatial_registered_primary"
        elif key == ("invariant", "none", "fused", "logistic"):
            name = "spatial_invariant_fused"
        else:
            continue
        output[name][(row["direction"], row["session_id"])] = (
            int(row["true_label"]), int(row["predicted_label"])
        )
    return output


def analyze(
    *,
    morphology_path: Path,
    classical_path: Path,
    siamese_path: Path,
    spatial_predictions_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    morphology = _load_payload(morphology_path)
    classical = _load_payload(classical_path)
    siamese = _load_payload(siamese_path)
    morphology_dir = morphology_path.parent
    classification_path = morphology_dir / "classification_predictions.csv"
    retrieval_path = morphology_dir / "retrieval_predictions.csv"
    for filename, path in (
        ("classification_predictions_sha256", classification_path),
        ("retrieval_predictions_sha256", retrieval_path),
    ):
        if _sha256(path) != morphology["output_hashes"][filename]:
            raise ValueError(f"Prediction hash mismatch: {path}")

    maps = _prediction_maps(
        _read_csv(classification_path),
        _read_csv(retrieval_path),
        _read_csv(spatial_predictions_path),
    )
    classification_comparisons = (
        ("attribute_morphology_only", "spatial_registered_primary"),
        ("attribute_morphology_only", "spatial_invariant_fused"),
        ("attribute_morphology_plus_position", "attribute_morphology_only"),
        ("attribute_position_only", "attribute_morphology_only"),
        ("retrieval_morphology_only", "spatial_registered_primary"),
        ("retrieval_morphology_only", "attribute_morphology_only"),
    )
    classification_stats = []
    sequence = 0
    for direction in ("january_to_april_may", "april_may_to_january"):
        for left_name, right_name in classification_comparisons:
            left_keys = {key for key in maps[left_name] if key[0] == direction}
            right_keys = {key for key in maps[right_name] if key[0] == direction}
            if left_keys != right_keys:
                raise ValueError(f"Prediction cohorts differ: {left_name} vs {right_name} {direction}")
            keys = sorted(left_keys)
            labels = np.asarray([maps[left_name][key][0] for key in keys])
            if not np.array_equal(labels, [maps[right_name][key][0] for key in keys]):
                raise ValueError("Paired labels differ")
            left = np.asarray([maps[left_name][key][1] for key in keys])
            right = np.asarray([maps[right_name][key][1] for key in keys])
            for metric in ("macro_f1", "balanced_accuracy", "worst_class_recall"):
                result = stratified_paired_bootstrap(
                    labels,
                    left,
                    right,
                    metric=metric,
                    seed=20260808 + sequence,
                )
                sequence += 1
                result.update(
                    {
                        "direction": direction,
                        "left_method": left_name,
                        "right_method": right_name,
                        "comparison": f"{left_name}_minus_{right_name}",
                    }
                )
                classification_stats.append(result)
    for metric in ("macro_f1", "balanced_accuracy", "worst_class_recall"):
        positions = [i for i, row in enumerate(classification_stats) if row["metric"] == metric]
        qvalues = bh_qvalues([classification_stats[i]["bootstrap_two_sided_p"] for i in positions])
        for index, qvalue in zip(positions, qvalues, strict=True):
            classification_stats[index]["bh_qvalue_within_metric_family"] = qvalue

    methods = _method_rows(classical, siamese)
    for method in ATTRIBUTE_METHODS:
        rows = {}
        for raw in morphology["enrollment"]["episodes"]:
            if raw["method"] != method or raw["selector"] != "random":
                continue
            key = (str(raw["direction"]), str(raw["heldout_class"]), int(raw["shot"]), int(raw["draw"]))
            rows[key] = dict(raw)
        if len(rows) != 1080:
            raise ValueError(f"Expected 1080 random attribute episodes for {method}, found {len(rows)}")
        methods[method] = rows

    enrollment_pairs = []
    for attribute_method in ATTRIBUTE_METHODS:
        for control in (
            "class_prototype",
            "registered_distribution_hybrid",
            "sliced_wasserstein_session_gallery",
            "cuda_supervised_siamese_session_embedding",
        ):
            enrollment_pairs.append((attribute_method, control))
    enrollment_pairs.append((ATTRIBUTE_METHODS[1], ATTRIBUTE_METHODS[0]))
    enrollment_stats = []
    class_effects = []
    sequence = 0
    for metric in METRICS:
        for direction in ("january_to_april_may", "april_may_to_january"):
            for shot in (1, 3, 5):
                for left_name, right_name in enrollment_pairs:
                    result, rows = paired_cluster_comparison(
                        methods[left_name],
                        methods[right_name],
                        direction=direction,
                        shot=shot,
                        metric=metric,
                        seed=20261808 + sequence,
                    )
                    sequence += 1
                    comparison = f"{left_name}_minus_{right_name}"
                    result.update(
                        {"left_method": left_name, "right_method": right_name, "comparison": comparison}
                    )
                    for row in rows:
                        row.update(
                            {"left_method": left_name, "right_method": right_name, "comparison": comparison}
                        )
                    enrollment_stats.append(result)
                    class_effects.extend(rows)
    for metric in METRICS:
        positions = [i for i, row in enumerate(enrollment_stats) if row["metric"] == metric]
        qvalues = bh_qvalues(
            [enrollment_stats[i]["exact_sign_flip_two_sided_p"] for i in positions]
        )
        for index, qvalue in zip(positions, qvalues, strict=True):
            enrollment_stats[index]["bh_qvalue_within_metric_family"] = qvalue

    output_dir.mkdir(parents=True, exist_ok=True)
    classification_csv = output_dir / "classification_paired_statistics.csv"
    enrollment_csv = output_dir / "enrollment_paired_statistics.csv"
    class_effect_csv = output_dir / "enrollment_heldout_class_effects.csv"
    _write_csv(classification_csv, classification_stats)
    _write_csv(enrollment_csv, enrollment_stats)
    _write_csv(class_effect_csv, class_effects)
    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 morphology attribute paired statistical analysis",
        "evidence_status": "retrospective development; not independent confirmation",
        "inputs": {
            "morphology_payload_sha256": morphology["payload_sha256"],
            "classical_enrollment_payload_sha256": classical["payload_sha256"],
            "siamese_enrollment_payload_sha256": siamese["payload_sha256"],
            "spatial_predictions_sha256": _sha256(spatial_predictions_path),
        },
        "classification_statistics": classification_stats,
        "enrollment_statistics": enrollment_stats,
        "multiplicity": "Benjamini-Hochberg within metric families",
        "output_hashes": {
            "classification_paired_statistics_sha256": _sha256(classification_csv),
            "enrollment_paired_statistics_sha256": _sha256(enrollment_csv),
            "enrollment_heldout_class_effects_sha256": _sha256(class_effect_csv),
        },
        "limitations": [
            "Classification intervals are retrospective paired target-session bootstraps.",
            "Enrollment inference has only six independent held-out-class clusters and coarse exact p-values.",
            "The tested signal-derived attributes are interpretations, not independently labelled morphology targets.",
        ],
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "morphology_attribute_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--siamese", type=Path, required=True)
    parser.add_argument("--spatial-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        morphology_path=args.morphology,
        classical_path=args.classical,
        siamese_path=args.siamese,
        spatial_predictions_path=args.spatial_predictions,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "classification_comparisons": len(result["classification_statistics"]),
                "enrollment_comparisons": len(result["enrollment_statistics"]),
                "payload_sha256": result["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
