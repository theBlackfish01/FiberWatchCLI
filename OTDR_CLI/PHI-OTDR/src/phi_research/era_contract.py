"""Immutable acquisition-era protocols for Phi-OTDR research.

The filename date token supports a conservative January versus April--May
comparison.  It does not identify the physical cause of the shift, so this
module deliberately uses ``era`` rather than site, subject, fiber, or system.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

from .data_contract import CLASS_NAMES, canonical_json_hash


ERAS = ("january", "april_may")
ERA_PARTITIONS = (
    "source_train",
    "source_validation",
    "source_calibration",
    "target_calibration",
    "target_support",
    "target_query",
)
SOURCE_PARTITIONS = frozenset(ERA_PARTITIONS[:3])
TARGET_PARTITIONS = frozenset(ERA_PARTITIONS[3:])


def acquisition_era(date_token: str) -> str:
    """Map only the date ranges observed in the audited local dataset."""
    if date_token.startswith("2201"):
        return "january"
    if date_token.startswith(("2204", "2205")):
        return "april_may"
    raise ValueError(f"Unsupported acquisition date token: {date_token!r}")


def _stable_rank(seed: int, direction: str, class_id: int, session_id: str) -> str:
    import hashlib

    value = f"{seed}|{direction}|{class_id}|{session_id}"
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _source_counts(session_count: int) -> dict[str, int]:
    calibration = max(2, int(round(session_count * 0.10)))
    validation = max(2, int(round(session_count * 0.15)))
    training = session_count - calibration - validation
    if training < 5:
        raise ValueError(f"Source era needs at least five training sessions per class: {session_count}")
    return {
        "source_train": training,
        "source_validation": validation,
        "source_calibration": calibration,
    }


def _target_counts(session_count: int) -> dict[str, int]:
    calibration = max(2, int(round(session_count * 0.15)))
    # Seven candidates permit multiple deterministic five-session draws even
    # for the two target-era classes with only twelve sessions.
    support = 7
    query = session_count - calibration - support
    if query < 3:
        raise ValueError(
            "Target era needs at least three locked query sessions after "
            f"calibration and support allocation: {session_count}"
        )
    return {
        "target_calibration": calibration,
        "target_support": support,
        "target_query": query,
    }


def build_acquisition_manifest(
    session_rows: Sequence[Mapping[str, object]],
    *,
    dataset_fingerprint: str,
    source_era: str,
    target_era: str,
    seed: int = 20260805,
    legacy_partitions: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Create one cross-era protocol while keeping every session indivisible."""
    if source_era not in ERAS or target_era not in ERAS or source_era == target_era:
        raise ValueError(f"Invalid acquisition direction: {source_era!r} -> {target_era!r}")
    direction = f"{source_era}_to_{target_era}"
    legacy_partitions = legacy_partitions or {}
    grouped: dict[tuple[str, int], list[Mapping[str, object]]] = defaultdict(list)
    seen_ids: set[str] = set()
    for row in session_rows:
        session_id = str(row["session_id"])
        if session_id in seen_ids:
            raise ValueError(f"Duplicate session row: {session_id}")
        seen_ids.add(session_id)
        era = acquisition_era(str(row["date_token"]))
        class_id = int(row["class_id"])
        if class_id not in range(len(CLASS_NAMES)):
            raise ValueError(f"Invalid class id for {session_id}: {class_id}")
        grouped[(era, class_id)].append(row)

    assignments: list[dict[str, object]] = []
    summary: dict[str, dict[str, dict[str, int]]] = {
        partition: {} for partition in ERA_PARTITIONS
    }
    for era in ERAS:
        for class_id in range(len(CLASS_NAMES)):
            if not grouped[(era, class_id)]:
                raise ValueError(f"Era {era} is missing class {CLASS_NAMES[class_id]}")

    for era, partitions in (
        (source_era, _source_counts),
        (target_era, _target_counts),
    ):
        for class_id, class_name in enumerate(CLASS_NAMES):
            rows = sorted(
                grouped[(era, class_id)],
                key=lambda row: _stable_rank(seed, direction, class_id, str(row["session_id"])),
            )
            counts = partitions(len(rows))
            offset = 0
            # Protected/non-training subsets are selected first from the
            # deterministic ordering, making allocation rules easy to audit.
            allocation_order = (
                ("source_validation", "source_calibration", "source_train")
                if era == source_era
                else ("target_calibration", "target_support", "target_query")
            )
            for partition in allocation_order:
                selected = rows[offset : offset + counts[partition]]
                for row in selected:
                    session_id = str(row["session_id"])
                    assignments.append(
                        {
                            "session_id": session_id,
                            "class_id": class_id,
                            "class_name": class_name,
                            "era": era,
                            "role": "source" if era == source_era else "target",
                            "partition": partition,
                            "window_count": int(row["window_count"]),
                            "date_token": str(row["date_token"]),
                            "source_token": str(row["source_token"]),
                            "legacy_v1_partition": legacy_partitions.get(session_id, "unrecorded"),
                        }
                    )
                summary[partition][class_name] = {
                    "sessions": len(selected),
                    "windows": sum(int(row["window_count"]) for row in selected),
                }
                offset += len(selected)
            if offset != len(rows):
                raise AssertionError(f"Allocation failed for {era}/{class_name}: {offset} != {len(rows)}")

    # Fill structurally inapplicable partition/class cells with zeros so that
    # machine readers never need to infer missing keys.
    for partition in ERA_PARTITIONS:
        for class_name in CLASS_NAMES:
            summary[partition].setdefault(class_name, {"sessions": 0, "windows": 0})

    payload: dict[str, object] = {
        "schema_version": 2,
        "name": "phi_otdr_acquisition_era_split_v2",
        "seed": seed,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "direction": {"source": source_era, "target": target_era},
        "partition_order": list(ERA_PARTITIONS),
        "grouping_unit": "recording session prefix before _single_data_<window>",
        "era_definition": {
            "january": "date token begins 2201",
            "april_may": "date token begins 2204 or 2205",
        },
        "semantics_limit": (
            "The direction is a filename-derived acquisition-era shift. It is not evidence of a "
            "specific subject, site, fiber, interrogator, weather, or environment change."
        ),
        "model_selection_policy": (
            "Architecture, preprocessing, hyperparameters, and support-selection rules may use "
            "source_train and source_validation only. Calibration partitions may set frozen "
            "decision thresholds but may not select models."
        ),
        "target_query_access_policy": (
            "No fitting, model selection, early stopping, representation selection, threshold "
            "calibration, support selection, or stop/go decision may use target_query data."
        ),
        "open_world_policy": (
            "For each leave-one-class-out fold, the held-out class is absent from source fitting "
            "and both calibration partitions. Its target_support sessions are available only "
            "after pre-enrollment evaluation and remain disjoint from target_query."
        ),
        "legacy_exposure": (
            "All local files were inventoried in v1. The v1 final query was previously evaluated "
            "under a mixed-era protocol, and non-final v1 sessions informed exploratory cross-era "
            "diagnostics. Therefore target_query is prospectively locked for v2 method development "
            "but is not claimed to be globally untouched historical data."
        ),
        "summary": summary,
        "sessions": sorted(assignments, key=lambda row: str(row["session_id"])),
    }
    payload["manifest_sha256"] = canonical_json_hash(payload)
    return payload


def verify_acquisition_manifest(
    manifest: Mapping[str, object], *, expected_dataset_fingerprint: str | None = None
) -> dict[str, object]:
    """Validate the hash, direction, roles, partitions, and class coverage."""
    payload = dict(manifest)
    stored_hash = str(payload.pop("manifest_sha256", ""))
    calculated_hash = canonical_json_hash(payload)
    if stored_hash != calculated_hash:
        raise ValueError(f"Manifest hash mismatch: stored={stored_hash}, calculated={calculated_hash}")
    if payload.get("name") not in {
        "phi_otdr_acquisition_era_split_v2",
        "phi_otdr_acquisition_era_split_v3",
    }:
        raise ValueError(f"Unsupported acquisition manifest: {payload.get('name')!r}")
    if expected_dataset_fingerprint is not None:
        observed = str(payload.get("dataset_fingerprint_sha256", ""))
        if observed != expected_dataset_fingerprint:
            raise ValueError(
                f"Dataset fingerprint mismatch: expected={expected_dataset_fingerprint}, observed={observed}"
            )
    direction = dict(payload["direction"])
    source_era, target_era = str(direction["source"]), str(direction["target"])
    if source_era == target_era or {source_era, target_era} != set(ERAS):
        raise ValueError(f"Invalid direction: {direction}")
    sessions = list(payload.get("sessions", []))
    ids = [str(row["session_id"]) for row in sessions]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest sessions are empty or duplicated")
    observed_partitions = {str(row["partition"]) for row in sessions}
    if observed_partitions != set(ERA_PARTITIONS):
        raise ValueError(f"Unexpected partition coverage: {sorted(observed_partitions)}")
    class_coverage: dict[str, set[int]] = defaultdict(set)
    for row in sessions:
        partition = str(row["partition"])
        era = str(row["era"])
        role = str(row["role"])
        class_id = int(row["class_id"])
        expected_era = source_era if partition in SOURCE_PARTITIONS else target_era
        expected_role = "source" if partition in SOURCE_PARTITIONS else "target"
        if era != expected_era or role != expected_role:
            raise ValueError(f"Role/era mismatch for {row['session_id']}: {partition}/{era}/{role}")
        if acquisition_era(str(row["date_token"])) != era:
            raise ValueError(f"Date/era mismatch for {row['session_id']}")
        class_coverage[partition].add(class_id)
    expected_classes = set(range(len(CLASS_NAMES)))
    for partition in ERA_PARTITIONS:
        if class_coverage[partition] != expected_classes:
            raise ValueError(f"Partition {partition} lacks full class coverage")
    return {
        "valid": True,
        "manifest_sha256": stored_hash,
        "dataset_fingerprint_sha256": str(payload["dataset_fingerprint_sha256"]),
        "direction": direction,
        "session_count": len(sessions),
        "partitions": {
            partition: sum(str(row["partition"]) == partition for row in sessions)
            for partition in ERA_PARTITIONS
        },
    }


def verify_protocol_hash(protocol_path: Path, hash_path: Path) -> dict[str, object]:
    """Verify a protocol sidecar using canonical JSON, not raw line endings."""
    payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    calculated = canonical_json_hash(payload)
    expected = hash_path.read_text(encoding="utf-8").strip().split()[0].lower()
    if calculated != expected:
        raise ValueError(f"Protocol hash mismatch: expected={expected}, calculated={calculated}")
    return {
        "valid": True,
        "name": str(payload.get("name", "")),
        "canonical_json_sha256": calculated,
    }


def create_acquisition_manifest(
    audit_dir: Path,
    output_path: Path,
    *,
    source_era: str,
    target_era: str,
    seed: int = 20260805,
    legacy_manifest_path: Path | None = None,
) -> dict[str, object]:
    audit = json.loads((audit_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    with (audit_dir / "session_inventory.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    legacy: dict[str, str] = {}
    if legacy_manifest_path is not None:
        prior = json.loads(legacy_manifest_path.read_text(encoding="utf-8"))
        legacy = {str(row["session_id"]): str(row["partition"]) for row in prior["sessions"]}
    manifest = build_acquisition_manifest(
        rows,
        dataset_fingerprint=str(audit["dataset_fingerprint_sha256"]),
        source_era=source_era,
        target_era=target_era,
        seed=seed,
        legacy_partitions=legacy,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
