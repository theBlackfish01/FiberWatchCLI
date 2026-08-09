"""Raw-session diagnostics for the frozen Phi-OTDR v3 registration rules."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from .data_contract import canonical_json_hash
from .dataset import SampleRef, build_sample_index, load_array
from .spatial_registration import activity_profile, profile_center, register_array


ESTIMATORS = (
    "temporal_difference_energy",
    "robust_variance",
    "spectral_energy",
    "multi_estimator_consensus",
)


def _representatives(samples: list[SampleRef]) -> list[SampleRef]:
    grouped: dict[str, list[SampleRef]] = defaultdict(list)
    for sample in samples:
        grouped[sample.session_id].append(sample)
    selected = []
    for session in sorted(grouped):
        local = sorted(grouped[session], key=lambda sample: sample.window_id)
        target = float(np.median([sample.window_id for sample in local]))
        selected.append(min(local, key=lambda sample: (abs(sample.window_id - target), sample.window_id)))
    return selected


def run(
    *,
    data_root: Path,
    manifest_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    samples = _representatives(build_sample_index(data_root, manifest_path))
    if len(samples) != 441:
        raise ValueError(f"Expected one representative for 441 sessions, received {len(samples)}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
    rows = []
    centers_by_session: dict[str, list[float]] = defaultdict(list)
    for index, sample in enumerate(samples, start=1):
        array = load_array(sample)
        metadata = session_rows[sample.session_id]
        for estimator in ESTIMATORS:
            result = register_array(array, estimator, temporal_stride=5)
            post_profile = activity_profile(result.values, estimator, temporal_stride=5)
            post_center, post_confidence = profile_center(post_profile)
            centers_by_session[sample.session_id].append(result.estimated_center)
            rows.append(
                {
                    "session_id": sample.session_id,
                    "class_name": sample.class_name,
                    "era": metadata["era"],
                    "rel_path": sample.rel_path,
                    "window_id": sample.window_id,
                    "estimator": estimator,
                    "center_before": result.estimated_center,
                    "confidence_before": result.confidence,
                    "applied_shift": result.applied_shift,
                    "center_after": post_center,
                    "confidence_after": post_confidence,
                    "center_error_after": abs(post_center - 5.5),
                    "retained_activity_fraction": result.retained_activity_fraction,
                    "clipped_channel_fraction": result.clipped_channel_fraction,
                }
            )
        if index % 50 == 0 or index == len(samples):
            print(f"[REGISTRATION] {index}/{len(samples)} sessions", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "raw_registration_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {}
    for estimator in ESTIMATORS:
        local = [row for row in rows if row["estimator"] == estimator]
        summary[estimator] = {
            "sessions": len(local),
            "median_abs_shift_channels": float(np.median([abs(row["applied_shift"]) for row in local])),
            "p90_abs_shift_channels": float(np.quantile([abs(row["applied_shift"]) for row in local], 0.90)),
            "median_post_center_error": float(np.median([row["center_error_after"] for row in local])),
            "p90_post_center_error": float(np.quantile([row["center_error_after"] for row in local], 0.90)),
            "median_retained_activity_fraction": float(np.median([row["retained_activity_fraction"] for row in local])),
            "p10_retained_activity_fraction": float(np.quantile([row["retained_activity_fraction"] for row in local], 0.10)),
            "fraction_any_channel_clipped": float(np.mean([row["clipped_channel_fraction"] > 0 for row in local])),
            "median_clipped_channel_fraction": float(np.median([row["clipped_channel_fraction"] for row in local])),
        }
    agreement = np.asarray([np.std(centers) for centers in centers_by_session.values()])
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "one raw median-window representative per complete-data session",
        "manifest_sha256": manifest["manifest_sha256"],
        "dataset_fingerprint_sha256": manifest["dataset_fingerprint_sha256"],
        "session_count": len(samples),
        "row_count": len(rows),
        "padding": "per-time baseline padding; no circular wrap",
        "summary": summary,
        "estimator_center_disagreement": {
            "median_within_session_center_std": float(np.median(agreement)),
            "p90_within_session_center_std": float(np.quantile(agreement, 0.90)),
            "fraction_std_gt_1_channel": float(np.mean(agreement > 1.0)),
        },
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "raw_registration_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(data_root=args.data_root, manifest_path=args.manifest, output_dir=args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
