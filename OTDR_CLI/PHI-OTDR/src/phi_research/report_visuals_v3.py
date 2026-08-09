"""Generate uncertainty and error-analysis visuals from frozen PHI-OTDR v3 predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, log_loss

from .data_contract import CLASS_NAMES, canonical_json_hash


DISPLAY = {
    "spatial_registered_primary": "Registered morphology",
    "spatial_invariant_fused": "Invariant fused",
    "attribute_morphology_only": "Morphology attributes",
    "neural_deepsets_registered_3seed_ensemble": "DeepSets (3-seed ensemble)",
    "neural_attention_registered_3seed_ensemble": "Attention (3-seed ensemble)",
}
COLORS = {
    "spatial_registered_primary": "#2563eb",
    "spatial_invariant_fused": "#059669",
    "attribute_morphology_only": "#7c3aed",
    "neural_deepsets_registered_3seed_ensemble": "#dc2626",
    "neural_attention_registered_3seed_ensemble": "#ea580c",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _probabilities(row: dict[str, str]) -> np.ndarray:
    return np.asarray([float(row[f"prob_{name}"]) for name in CLASS_NAMES], dtype=np.float64)


def _datasets(
    spatial_path: Path,
    attribute_path: Path,
    neural_path: Path,
) -> dict[tuple[str, str], dict[str, object]]:
    rows: dict[tuple[str, str], list[tuple[str, int, np.ndarray]]] = defaultdict(list)
    for row in _read(spatial_path):
        key = (row["view"], row["estimator"], row["ablation"], row["model"])
        if key == ("registered_position", "temporal_difference_energy", "dynamics", "logistic"):
            method = "spatial_registered_primary"
        elif key == ("invariant", "none", "fused", "logistic"):
            method = "spatial_invariant_fused"
        else:
            continue
        rows[(row["direction"], method)].append(
            (row["session_id"], int(row["true_label"]), _probabilities(row))
        )
    for row in _read(attribute_path):
        if row["view"] != "morphology_only":
            continue
        rows[(row["direction"], "attribute_morphology_only")].append(
            (row["session_id"], int(row["true_label"]), _probabilities(row))
        )

    neural_group: dict[tuple[str, str, str], list[tuple[int, np.ndarray]]] = defaultdict(list)
    for row in _read(neural_path):
        if row["view"] != "registered_position_difference_dynamics":
            continue
        neural_group[(row["direction"], row["architecture"], row["session_id"])].append(
            (int(row["true_label"]), _probabilities(row))
        )
    for (direction, architecture, session), values in neural_group.items():
        labels = {label for label, _ in values}
        if len(values) != 3 or len(labels) != 1:
            raise ValueError("Neural ensemble requires three label-consistent seeds")
        method = f"neural_{architecture}_registered_3seed_ensemble"
        rows[(direction, method)].append(
            (session, labels.pop(), np.mean([probability for _, probability in values], axis=0))
        )

    output = {}
    for key, values in rows.items():
        values = sorted(values, key=lambda item: item[0])
        sessions = np.asarray([item[0] for item in values])
        labels = np.asarray([item[1] for item in values], dtype=np.int64)
        probs = np.stack([item[2] for item in values])
        if not np.allclose(np.sum(probs, axis=1), 1.0, atol=1e-5):
            raise ValueError(f"Probabilities do not sum to one for {key}")
        output[key] = {"sessions": sessions, "labels": labels, "probabilities": probs}
    return output


def _reliability(labels: np.ndarray, probs: np.ndarray, bins: int = 10) -> list[dict[str, object]]:
    confidence = np.max(probs, axis=1)
    correct = np.argmax(probs, axis=1) == labels
    output = []
    edges = np.linspace(0.0, 1.0, bins + 1)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        mask = (confidence > left) & (confidence <= right)
        output.append(
            {
                "bin": index,
                "left": float(left),
                "right": float(right),
                "count": int(np.sum(mask)),
                "mean_confidence": float(np.mean(confidence[mask])) if np.any(mask) else None,
                "accuracy": float(np.mean(correct[mask])) if np.any(mask) else None,
            }
        )
    return output


def _risk_curve(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    confidence = np.max(probs, axis=1)
    prediction = np.argmax(probs, axis=1)
    order = np.argsort(-confidence, kind="stable")
    correct = prediction[order] == labels[order]
    coverage = np.arange(1, len(labels) + 1) / len(labels)
    risk = 1.0 - np.cumsum(correct) / np.arange(1, len(labels) + 1)
    return coverage, risk, float(np.mean(risk))


def _summary(labels: np.ndarray, probs: np.ndarray) -> dict[str, object]:
    prediction = np.argmax(probs, axis=1)
    recalls = []
    for class_id in range(len(CLASS_NAMES)):
        mask = labels == class_id
        recalls.append(float(np.mean(prediction[mask] == class_id)))
    reliability = _reliability(labels, probs)
    ece = sum(
        row["count"] / len(labels) * abs(row["accuracy"] - row["mean_confidence"])
        for row in reliability
        if row["count"]
    )
    one_hot = np.eye(len(CLASS_NAMES))[labels]
    clipped = np.clip(probs, 1e-12, 1.0)
    clipped /= np.sum(clipped, axis=1, keepdims=True)
    _, _, aurc = _risk_curve(labels, probs)
    return {
        "sessions": int(len(labels)),
        "accuracy": float(np.mean(prediction == labels)),
        "macro_f1": float(
            f1_score(
                labels,
                prediction,
                labels=np.arange(len(CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        ),
        "balanced_accuracy": float(np.mean(recalls)),
        "worst_class_recall": float(np.min(recalls)),
        "per_class_recall": {name: recalls[i] for i, name in enumerate(CLASS_NAMES)},
        "negative_log_likelihood": float(
            log_loss(labels, clipped, labels=np.arange(len(CLASS_NAMES)))
        ),
        "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
        "ece_10": float(ece),
        "aurc": aurc,
        "reliability_bins": reliability,
        "confusion_matrix": confusion_matrix(
            labels, prediction, labels=np.arange(len(CLASS_NAMES))
        ).tolist(),
    }


def generate(
    *,
    spatial_path: Path,
    attribute_path: Path,
    neural_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    datasets = _datasets(spatial_path, attribute_path, neural_path)
    expected = 2 * len(DISPLAY)
    if len(datasets) != expected:
        raise ValueError(f"Expected {expected} direction-method datasets, found {len(datasets)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    summary_lookup = {}
    for (direction, method), dataset in sorted(datasets.items()):
        metrics = _summary(dataset["labels"], dataset["probabilities"])
        row = {"direction": direction, "method": method, **metrics}
        summaries.append(row)
        summary_lookup[(direction, method)] = row

    directions = ("january_to_april_may", "april_may_to_january")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True, constrained_layout=True)
    for axis, direction in zip(axes, directions, strict=True):
        axis.plot([0, 1], [0, 1], "--", color="#64748b", linewidth=1, label="Ideal")
        for method in DISPLAY:
            bins = summary_lookup[(direction, method)]["reliability_bins"]
            x = [row["mean_confidence"] for row in bins if row["count"]]
            y = [row["accuracy"] for row in bins if row["count"]]
            axis.plot(x, y, marker="o", linewidth=1.8, color=COLORS[method], label=DISPLAY[method])
        axis.set_title(direction.replace("_to_", " → ").replace("_", "/").title())
        axis.set_xlabel("Mean confidence")
        axis.set_ylabel("Observed accuracy")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="lower right")
    reliability_path = output_dir / "reliability_diagrams.png"
    fig.savefig(reliability_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True, constrained_layout=True)
    for axis, direction in zip(axes, directions, strict=True):
        for method in DISPLAY:
            dataset = datasets[(direction, method)]
            coverage, risk, _ = _risk_curve(dataset["labels"], dataset["probabilities"])
            axis.plot(coverage, risk, linewidth=1.8, color=COLORS[method], label=DISPLAY[method])
        axis.set_title(direction.replace("_to_", " → ").replace("_", "/").title())
        axis.set_xlabel("Coverage retained")
        axis.set_ylabel("Error risk")
        axis.grid(alpha=0.25)
    axes[1].legend(fontsize=8, loc="upper left")
    risk_path = output_dir / "risk_coverage_curves.png"
    fig.savefig(risk_path, dpi=180)
    plt.close(fig)

    confusion_paths = []
    selected_methods = (
        "spatial_registered_primary",
        "spatial_invariant_fused",
        "attribute_morphology_only",
        "neural_deepsets_registered_3seed_ensemble",
    )
    for direction in directions:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), constrained_layout=True)
        for axis, method in zip(axes, selected_methods, strict=True):
            matrix = np.asarray(summary_lookup[(direction, method)]["confusion_matrix"])
            row_sum = np.maximum(matrix.sum(axis=1, keepdims=True), 1)
            normalized = matrix / row_sum
            image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues")
            for row in range(6):
                for column in range(6):
                    axis.text(
                        column,
                        row,
                        f"{normalized[row, column]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if normalized[row, column] > 0.55 else "black",
                    )
            axis.set_title(DISPLAY[method], fontsize=10)
            axis.set_xticks(range(6), CLASS_NAMES, rotation=45, ha="right", fontsize=7)
            axis.set_yticks(range(6), CLASS_NAMES, fontsize=7)
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
        fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01, label="Row-normalized fraction")
        path = output_dir / f"confusion_{direction}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        confusion_paths.append(path)

    methods = list(DISPLAY)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True, constrained_layout=True)
    for axis, direction in zip(axes, directions, strict=True):
        matrix = np.asarray(
            [
                [summary_lookup[(direction, method)]["per_class_recall"][name] for name in CLASS_NAMES]
                for method in methods
            ]
        )
        image = axis.imshow(matrix, vmin=0, vmax=1, cmap="viridis")
        for row in range(len(methods)):
            for column in range(6):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center", fontsize=8,
                          color="white" if matrix[row, column] < 0.45 else "black")
        axis.set_xticks(range(6), CLASS_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(len(methods)), [DISPLAY[method] for method in methods], fontsize=8)
        axis.set_title(direction.replace("_to_", " → ").replace("_", "/").title())
    fig.colorbar(image, ax=axes, fraction=0.02, pad=0.01, label="Recall")
    recall_path = output_dir / "per_class_recall_heatmap.png"
    fig.savefig(recall_path, dpi=180)
    plt.close(fig)

    flat_rows = []
    for row in summaries:
        flat_rows.append(
            {
                key: value
                for key, value in row.items()
                if key not in {"per_class_recall", "reliability_bins", "confusion_matrix"}
            }
        )
    csv_path = output_dir / "calibration_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)

    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 retrospective calibration and selective-prediction synthesis",
        "evidence_status": "retrospective descriptive analysis",
        "inputs": {
            "spatial_predictions_sha256": _sha256(spatial_path),
            "attribute_predictions_sha256": _sha256(attribute_path),
            "neural_predictions_sha256": _sha256(neural_path),
        },
        "summaries": summaries,
        "neural_ensemble_policy": "Post-hoc descriptive average of three already calibrated seed probabilities; not a selected primary model.",
        "output_hashes": {
            "calibration_summary_csv_sha256": _sha256(csv_path),
            "reliability_diagrams_sha256": _sha256(reliability_path),
            "risk_coverage_curves_sha256": _sha256(risk_path),
            "per_class_recall_heatmap_sha256": _sha256(recall_path),
            **{path.name: _sha256(path) for path in confusion_paths},
        },
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "calibration_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--attribute", type=Path, required=True)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = generate(
        spatial_path=args.spatial,
        attribute_path=args.attribute,
        neural_path=args.neural,
        output_dir=args.output_dir,
    )
    print(json.dumps({"summaries": len(result["summaries"]), "payload_sha256": result["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
