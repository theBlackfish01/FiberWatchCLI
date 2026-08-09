"""Paired statistics for the PHI-OTDR v3 wavelet/rank missing-control analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .data_contract import canonical_json_hash
from .enrollment_analysis_v3 import bh_qvalues
from .morphology_attribute_analysis_v3 import stratified_paired_bootstrap


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


def _read(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze(
    *,
    extended_path: Path,
    morphology_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    extended = _load_payload(extended_path)
    morphology = _load_payload(morphology_path)
    extended_predictions = extended_path.parent / "classification_predictions.csv"
    morphology_predictions = morphology_path.parent / "classification_predictions.csv"
    if _sha256(extended_predictions) != extended["prediction_sha256"]:
        raise ValueError("Extended prediction hash mismatch")
    if _sha256(morphology_predictions) != morphology["output_hashes"]["classification_predictions_sha256"]:
        raise ValueError("Morphology prediction hash mismatch")

    maps: dict[str, dict[tuple[str, str], tuple[int, int]]] = {}
    for name, path in (
        ("morphology_only", morphology_predictions),
        ("wavelet_rank_only", extended_predictions),
        ("morphology_plus_wavelet_rank", extended_predictions),
    ):
        view = name
        maps[name] = {}
        for row in _read(path):
            if row["view"] != view:
                continue
            maps[name][(row["direction"], row["session_id"])] = (
                int(row["true_label"]),
                int(row["predicted_label"]),
            )

    rows = []
    comparisons = (
        ("wavelet_rank_only", "morphology_only"),
        ("morphology_plus_wavelet_rank", "morphology_only"),
    )
    for direction_index, direction in enumerate(("january_to_april_may", "april_may_to_january")):
        for comparison_index, (left, right) in enumerate(comparisons):
            keys = sorted(key for key in maps[left] if key[0] == direction)
            if keys != sorted(key for key in maps[right] if key[0] == direction):
                raise ValueError(f"Prediction sessions disagree for {direction}: {left}/{right}")
            labels = np.asarray([maps[left][key][0] for key in keys], dtype=np.int64)
            left_prediction = np.asarray([maps[left][key][1] for key in keys], dtype=np.int64)
            right_prediction = np.asarray([maps[right][key][1] for key in keys], dtype=np.int64)
            for metric_index, metric in enumerate(("macro_f1", "balanced_accuracy", "worst_class_recall")):
                result = stratified_paired_bootstrap(
                    labels,
                    left_prediction,
                    right_prediction,
                    metric=metric,
                    seed=20260808 + direction_index * 100 + comparison_index * 10 + metric_index,
                    draws=5000,
                )
                result.update(
                    {
                        "direction": direction,
                        "left": left,
                        "right": right,
                    }
                )
                rows.append(result)
    for metric in ("macro_f1", "balanced_accuracy", "worst_class_recall"):
        indices = [index for index, row in enumerate(rows) if row["metric"] == metric]
        qvalues = bh_qvalues([float(rows[index]["bootstrap_two_sided_p"]) for index in indices])
        for index, qvalue in zip(indices, qvalues, strict=True):
            rows[index]["bh_qvalue_within_metric"] = qvalue

    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "paired_statistics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 paired wavelet/rank missing-control statistics",
        "evidence_status": "retrospective development; not confirmatory",
        "inputs": {
            "extended_payload_sha256": extended["payload_sha256"],
            "morphology_payload_sha256": morphology["payload_sha256"],
            "extended_predictions_sha256": _sha256(extended_predictions),
            "morphology_predictions_sha256": _sha256(morphology_predictions),
        },
        "comparisons": rows,
        "paired_statistics_sha256": _sha256(csv_path),
        "multiplicity": "Benjamini-Hochberg within metric family over four comparisons",
        "limitations": [
            "The analysis is post-hoc retrospective evidence on previously exposed target cohorts.",
            "The two directions reuse the same corpus and do not constitute independent deployment domains.",
        ],
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "extended_morphology_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        extended_path=args.extended.resolve(),
        morphology_path=args.morphology.resolve(),
        output_dir=args.output_dir.resolve(),
    )
    print(json.dumps({"comparisons": len(result["comparisons"]), "payload_sha256": result["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
