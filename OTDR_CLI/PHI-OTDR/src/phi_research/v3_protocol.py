"""Freeze the complete-data Phi-OTDR v3 retrospective-development protocol.

V3 deliberately reuses the already-frozen v2 session assignments.  It updates
only the dataset fingerprint and per-session window counts after recovery of the
two upstream archive parts.  Historical target-query results have been viewed,
so v3 is explicitly not presented as a new confirmatory evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Mapping

from .data_contract import CLASS_NAMES, canonical_json_hash
from .era_contract import ERA_PARTITIONS, verify_acquisition_manifest


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _session_inventory(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    inventory = {str(row["session_id"]): row for row in rows}
    if not inventory or len(inventory) != len(rows):
        raise ValueError("Complete session inventory is empty or duplicated")
    return inventory


def _summary(sessions: list[dict[str, object]]) -> dict[str, dict[str, dict[str, int]]]:
    counts: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"sessions": 0, "windows": 0})
    )
    for row in sessions:
        cell = counts[str(row["partition"])][str(row["class_name"])]
        cell["sessions"] += 1
        cell["windows"] += int(row["window_count"])
    return {
        partition: {
            class_name: counts[partition][class_name]
            for class_name in CLASS_NAMES
        }
        for partition in ERA_PARTITIONS
    }


def upgrade_manifest(
    v2: Mapping[str, object],
    *,
    complete_fingerprint: str,
    inventory: Mapping[str, Mapping[str, str]],
) -> dict[str, object]:
    """Upgrade a v2 manifest without changing a single session assignment."""
    payload = deepcopy(dict(v2))
    old_hash = str(payload.pop("manifest_sha256"))
    sessions = list(payload["sessions"])
    ids = {str(row["session_id"]) for row in sessions}
    if ids != set(inventory):
        missing = sorted(set(inventory) - ids)
        extra = sorted(ids - set(inventory))
        raise ValueError(f"V2/complete session mismatch: missing={missing[:3]}, extra={extra[:3]}")
    for row in sessions:
        current = inventory[str(row["session_id"])]
        if int(row["class_id"]) != int(current["class_id"]):
            raise ValueError(f"Class mismatch for {row['session_id']}")
        row["window_count"] = int(current["window_count"])
    payload.update(
        {
            "schema_version": 3,
            "name": "phi_otdr_acquisition_era_split_v3",
            "dataset_fingerprint_sha256": complete_fingerprint,
            "derived_from_v2_manifest_sha256": old_hash,
            "evidence_status": "retrospective development; target-query outcomes were previously inspected",
            "external_confirmation_required": True,
            "model_selection_policy": (
                "Choose representations, architectures, preprocessing, and hyperparameters using "
                "source_train/source_validation only. Target calibration/support may be used only "
                "for their declared calibration or enrollment roles. Historical target_query "
                "correctness must not choose, stop, or revise a method."
            ),
            "target_query_access_policy": (
                "Target-query outcomes are already known from v2 and are retrospective evaluation "
                "evidence only. They cannot support a new confirmatory claim; a newly acquired "
                "domain is required for confirmation."
            ),
            "summary": _summary(sessions),
            "sessions": sorted(sessions, key=lambda row: str(row["session_id"])),
        }
    )
    payload["manifest_sha256"] = canonical_json_hash(payload)
    verify_acquisition_manifest(payload, expected_dataset_fingerprint=complete_fingerprint)
    return payload


def freeze_v3(
    *,
    v2_config_dir: Path,
    complete_audit: Path,
    complete_session_inventory: Path,
    output_dir: Path,
) -> dict[str, object]:
    audit = _read_json(complete_audit)
    fingerprint = str(audit["dataset_fingerprint_sha256"])
    if int(audit["readable_file_count"]) != 15418 or int(audit["session_count"]) != 441:
        raise ValueError("Complete-data audit no longer matches the reviewed v3 baseline")
    inventory = _session_inventory(complete_session_inventory)
    output_dir.mkdir(parents=True, exist_ok=True)
    directions = []
    specs = (
        ("january_to_april_may", "acquisition_january_to_april_may_v2.json"),
        ("april_may_to_january", "acquisition_april_may_to_january_v2.json"),
    )
    for name, filename in specs:
        manifest = upgrade_manifest(
            _read_json(v2_config_dir / filename),
            complete_fingerprint=fingerprint,
            inventory=inventory,
        )
        output_name = filename.replace("_v2.json", "_v3.json")
        output_path = output_dir / output_name
        output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        directions.append(
            {
                "name": name,
                "manifest": output_name,
                "manifest_sha256": manifest["manifest_sha256"],
                "derived_from_v2_manifest_sha256": manifest["derived_from_v2_manifest_sha256"],
            }
        )
    protocol: dict[str, object] = {
        "schema_version": 3,
        "name": "phi_otdr_spatial_morphology_retrospective_v3",
        "frozen_date": "2026-08-08",
        "dataset_fingerprint_sha256": fingerprint,
        "data_contract": {
            "listed_mat_files": int(audit["listed_file_count"]),
            "readable_mat_files": int(audit["readable_file_count"]),
            "sessions": int(audit["session_count"]),
            "known_zero_byte_files": 1,
            "known_exact_duplicate_groups": 1,
        },
        "directions": directions,
        "independent_unit": "recording session prefix before _single_data_<window>",
        "evidence_status": "retrospective development, not confirmatory",
        "external_confirmation_required": True,
        "selection_policy": (
            "All method selection and stop/go decisions use source development partitions only. "
            "Target support/calibration are restricted to declared enrollment/calibration. "
            "Historical target-query metrics are evaluated only after each comparison is frozen."
        ),
        "primary_question": (
            "Can spatially registered, position-aware morphology reduce acquisition-era sensitivity "
            "while preserving session-level activity discrimination?"
        ),
        "registration_estimators": [
            "temporal_difference_energy",
            "robust_variance",
            "spectral_energy",
            "multi_estimator_consensus",
        ],
        "required_views": [
            "absolute_channels",
            "translation_invariant_pooling",
            "registered_morphology",
            "registered_plus_position",
            "dual_branch_absolute_and_registered",
            "source_channel_shift_augmentation",
        ],
        "required_controls": [
            "label_permutation",
            "era_probe",
            "source_only_selection",
            "boundary_clipping_and_energy_retention",
            "amplitude_only",
            "dynamics_only",
            "fused",
        ],
        "neural_policy": {
            "cuda_required": True,
            "cpu_fallback_forbidden": True,
            "seeds": [20260808, 20260809, 20260810],
            "minimum_seeds_for_claim": 3,
            "mixed_precision": "allowed and logged",
        },
        "enrollment_policy": {
            "shots": [1, 3, 5],
            "draws": 30,
            "selectors": ["random", "medoid", "k_center", "pool_coverage"],
            "support_query_session_disjoint": True,
        },
        "primary_metrics": [
            "session_macro_f1",
            "session_balanced_accuracy",
            "per_class_recall",
            "worst_class_recall",
            "acquisition_generalization_gap",
        ],
        "secondary_metrics": [
            "negative_log_likelihood",
            "brier_score",
            "expected_calibration_error",
            "era_probe_balanced_accuracy",
            "risk_coverage",
            "runtime_and_peak_cuda_memory",
        ],
        "uncertainty": (
            "session-cluster bootstrap confidence intervals; paired direction/class comparisons; "
            "Benjamini-Hochberg correction for feature families"
        ),
        "stop_go": {
            "registration": (
                "Continue if a source-selected registered or invariant view improves either-direction "
                "target macro-F1 or worst-class recall without a material source-validation or "
                "information-retention penalty; interpret retrospective evidence cautiously."
            ),
            "neural": (
                "Continue only if interpretable morphology establishes transferable signal or a "
                "CUDA pilot improves source validation and at least one retrospective direction."
            ),
            "external": (
                "No paper-level robustness claim until a new session-separated acquisition domain "
                "is evaluated once under a preregistered frozen protocol."
            ),
        },
    }
    protocol_path = output_dir / "acquisition_protocol_v3.json"
    protocol_path.write_text(json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8")
    protocol_hash = canonical_json_hash(protocol)
    (output_dir / "acquisition_protocol_v3.sha256").write_text(
        f"{protocol_hash}  acquisition_protocol_v3.json (canonical JSON)\n", encoding="utf-8"
    )
    return {"protocol_sha256": protocol_hash, "directions": directions, "fingerprint": fingerprint}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v2-config-dir", type=Path, required=True)
    parser.add_argument("--complete-audit", type=Path, required=True)
    parser.add_argument("--complete-session-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = freeze_v3(
        v2_config_dir=args.v2_config_dir,
        complete_audit=args.complete_audit,
        complete_session_inventory=args.complete_session_inventory,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
