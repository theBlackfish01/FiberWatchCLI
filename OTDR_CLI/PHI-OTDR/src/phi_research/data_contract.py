"""Auditable data inventory and immutable session-level split construction.

The original dataset ships window-level ``train`` and ``test`` lists.  Those
lists are retained as provenance only: a recording prefix before
``_single_data_<n>`` is the smallest defensible independent unit exposed by
the available metadata, so all research partitions are assigned by that
session identifier.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import scipy.io as sio


CLASS_NAMES = ("background", "digging", "knocking", "watering", "shaking", "walking")
PARTITIONS = ("train", "validation", "calibration", "support", "final_query")
FOLDER_LABELS = {
    "01_background": 0,
    "02_dig": 1,
    "03_knock": 2,
    "04_water": 3,
    "05_shake": 4,
    "06_walk": 5,
}
SAMPLE_RE = re.compile(r"^(?P<session>.+)_single_data_(?P<window>\d+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedName:
    session_id: str
    window_id: int
    date_token: str
    source_token: str


@dataclass
class InventoryRecord:
    rel_path: str
    source_split: str
    class_id: int
    class_name: str
    folder_name: str
    session_id: str
    window_id: int
    date_token: str
    source_token: str
    exists: bool
    readable: bool = False
    error: str = ""
    data_key: str = ""
    shape: str = ""
    dtype: str = ""
    finite: bool = False
    minimum: float | None = None
    maximum: float | None = None
    mean: float | None = None
    standard_deviation: float | None = None
    file_size: int = 0
    file_sha256: str = ""
    array_sha256: str = ""


def parse_sample_name(filename: str) -> ParsedName:
    """Parse only metadata that is stable across all observed naming variants."""
    stem = Path(filename).stem
    match = SAMPLE_RE.fullmatch(stem)
    if match is None:
        raise ValueError(f"Unrecognized Phi-OTDR sample name: {filename}")
    session = match.group("session")
    tokens = session.split("_")
    if len(tokens) < 3 or not re.fullmatch(r"\d{6}", tokens[0]):
        raise ValueError(f"Session prefix lacks the expected date/source fields: {filename}")
    return ParsedName(
        session_id=session,
        window_id=int(match.group("window")),
        date_token=tokens[0],
        # This is deliberately not called a subject: later batches use event
        # names in the same position, so person identity is unresolved.
        source_token=tokens[1],
    )


def _normalized_rel_path(value: str) -> Path:
    value = value.strip().strip('"').strip("'").replace("\\", "/").lstrip("/")
    return Path(value)


def _read_labels(path: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) != 2:
            raise ValueError(f"Malformed label line {path}:{line_number}: {line!r}")
        rows.append((_normalized_rel_path(parts[0]), int(parts[1])))
    return rows


def _load_array(path: Path) -> tuple[str, np.ndarray]:
    raw = sio.loadmat(path.as_posix())
    keys = [key for key in raw if not key.startswith("__")]
    if not keys:
        raise KeyError("MAT file has no user array")
    key = "data" if "data" in raw else keys[0]
    array = np.asarray(raw[key])
    if array.ndim != 2:
        raise ValueError(f"expected a 2-D array, found {array.shape}")
    return key, array


def _scan_record(record: InventoryRecord, absolute_path: Path) -> None:
    if not record.exists:
        record.error = "missing"
        return
    record.file_size = absolute_path.stat().st_size
    try:
        key, array = _load_array(absolute_path)
        contiguous = np.ascontiguousarray(array)
        record.data_key = key
        record.shape = "x".join(str(value) for value in array.shape)
        record.dtype = str(array.dtype)
        # Integer sensor arrays are finite by construction. Avoid four full
        # arithmetic passes over 1.8 billion values during the contract audit;
        # distributional statistics are computed by the feature-analysis stage.
        record.finite = True if np.issubdtype(array.dtype, np.integer) else bool(np.isfinite(array).all())
        record.array_sha256 = hashlib.sha256(memoryview(contiguous)).hexdigest()
        record.readable = True
    except Exception as exc:  # inventory must preserve failures rather than abort
        record.error = f"{type(exc).__name__}: {exc}"


def canonical_json_hash(payload: object) -> str:
    """Hash a JSON-compatible payload independently of file formatting."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# Backward-compatible internal name used by the v1 split implementation.
_json_hash = canonical_json_hash


def _write_csv(path: Path, rows: Iterable[Mapping[str, object]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def audit_dataset(data_root: Path, output_dir: Path) -> dict[str, object]:
    """Scan every listed MAT file and emit machine-readable inventory evidence."""
    started = time.perf_counter()
    data_root = data_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[InventoryRecord] = []
    listed_paths: set[str] = set()

    for source_split in ("train", "test"):
        split_root = data_root / source_split
        label_path = split_root / "label.txt"
        for rel_path, class_id in _read_labels(label_path):
            absolute_path = split_root / rel_path
            parsed = parse_sample_name(rel_path.name)
            folder_name = rel_path.parts[0]
            if class_id not in range(len(CLASS_NAMES)):
                raise ValueError(f"Invalid class id {class_id} for {source_split}/{rel_path}")
            inferred = FOLDER_LABELS.get(folder_name)
            if inferred is None or inferred != class_id:
                raise ValueError(f"Folder/label disagreement for {source_split}/{rel_path}: {class_id}")
            canonical_rel = f"{source_split}/{rel_path.as_posix()}"
            if canonical_rel in listed_paths:
                raise ValueError(f"Duplicate label entry: {canonical_rel}")
            listed_paths.add(canonical_rel)
            record = InventoryRecord(
                rel_path=canonical_rel,
                source_split=source_split,
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                folder_name=folder_name,
                session_id=parsed.session_id,
                window_id=parsed.window_id,
                date_token=parsed.date_token,
                source_token=parsed.source_token,
                exists=absolute_path.is_file(),
            )
            _scan_record(record, absolute_path)
            records.append(record)
            if len(records) % 250 == 0:
                print(
                    f"[AUDIT] scanned {len(records)} files in {time.perf_counter() - started:.1f}s",
                    flush=True,
                )

    actual_paths = {
        path.relative_to(data_root).as_posix()
        for path in data_root.rglob("*.mat")
    }
    extra_paths = sorted(actual_paths - listed_paths)

    identity_counts = Counter((row.session_id, row.window_id) for row in records if row.exists)
    duplicate_identities = [
        {"session_id": session, "window_id": window, "count": count}
        for (session, window), count in identity_counts.items()
        if count > 1
    ]
    array_hash_groups: dict[str, list[str]] = defaultdict(list)
    for row in records:
        if row.array_sha256:
            array_hash_groups[row.array_sha256].append(row.rel_path)
    duplicate_arrays = [paths for paths in array_hash_groups.values() if len(paths) > 1]

    sessions: dict[str, list[InventoryRecord]] = defaultdict(list)
    for row in records:
        if row.readable:
            sessions[row.session_id].append(row)
    session_rows: list[dict[str, object]] = []
    for session_id, members in sorted(sessions.items()):
        class_ids = {member.class_id for member in members}
        if len(class_ids) != 1:
            raise ValueError(f"Session spans multiple classes: {session_id} -> {class_ids}")
        session_rows.append(
            {
                "session_id": session_id,
                "class_id": members[0].class_id,
                "class_name": members[0].class_name,
                "date_token": members[0].date_token,
                "source_token": members[0].source_token,
                "window_count": len(members),
                "source_splits": "+".join(sorted({member.source_split for member in members})),
                "minimum_window_id": min(member.window_id for member in members),
                "maximum_window_id": max(member.window_id for member in members),
            }
        )

    fingerprint_rows = [
        f"{row.rel_path}|{row.class_id}|{row.session_id}|{row.window_id}|{row.array_sha256 or row.error}"
        for row in sorted(records, key=lambda item: item.rel_path)
    ]
    dataset_fingerprint = hashlib.sha256("\n".join(fingerprint_rows).encode("utf-8")).hexdigest()
    source_split_sessions = {
        split: {row.session_id for row in records if row.readable and row.source_split == split}
        for split in ("train", "test")
    }
    class_file_counts = Counter(row.class_name for row in records if row.readable)
    class_session_counts = Counter(row["class_name"] for row in session_rows)
    shape_counts = Counter(row.shape for row in records if row.readable)
    dtype_counts = Counter(row.dtype for row in records if row.readable)

    summary: dict[str, object] = {
        "schema_version": 1,
        "data_root_name": data_root.name,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "listed_file_count": len(records),
        "actual_mat_file_count": len(actual_paths),
        "existing_listed_file_count": sum(row.exists for row in records),
        "readable_file_count": sum(row.readable for row in records),
        "missing_file_count": sum(not row.exists for row in records),
        "unreadable_file_count": sum(row.exists and not row.readable for row in records),
        "extra_unlisted_files": extra_paths,
        "session_count": len(session_rows),
        "class_file_counts": dict(sorted(class_file_counts.items())),
        "class_session_counts": dict(sorted(class_session_counts.items())),
        "shape_counts": dict(sorted(shape_counts.items())),
        "dtype_counts": dict(sorted(dtype_counts.items())),
        "nonfinite_file_count": sum(row.readable and not row.finite for row in records),
        "duplicate_session_window_identities": duplicate_identities,
        "duplicate_array_group_count": len(duplicate_arrays),
        "duplicate_array_file_count": sum(len(group) for group in duplicate_arrays),
        "original_train_session_count": len(source_split_sessions["train"]),
        "original_test_session_count": len(source_split_sessions["test"]),
        "original_session_overlap_count": len(
            source_split_sessions["train"] & source_split_sessions["test"]
        ),
        "source_token_semantics": (
            "Unresolved: the second filename token resembles participant initials in early batches "
            "but is an event word in several later batches; it is not treated as a verified subject ID."
        ),
        "independent_unit": "filename prefix before _single_data_<window>; interpreted as recording session",
        "elapsed_seconds": time.perf_counter() - started,
    }

    inventory_fields = list(asdict(records[0]).keys())
    _write_csv(output_dir / "dataset_inventory.csv", (asdict(row) for row in records), inventory_fields)
    _write_csv(output_dir / "session_inventory.csv", session_rows, list(session_rows[0].keys()))
    (output_dir / "duplicate_arrays.json").write_text(
        json.dumps(duplicate_arrays, indent=2), encoding="utf-8"
    )
    (output_dir / "dataset_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return summary


def _partition_counts(session_count: int) -> dict[str, int]:
    if session_count < 25:
        raise ValueError(f"At least 25 sessions per class are required, found {session_count}")
    held_out_each = max(5, int(round(session_count * 0.10)))
    counts = {name: held_out_each for name in PARTITIONS[1:]}
    counts["train"] = session_count - sum(counts.values())
    if counts["train"] <= max(counts[name] for name in PARTITIONS[1:]):
        raise ValueError(f"Insufficient training sessions after allocation: {counts}")
    return counts


def _stable_rank(seed: int, class_id: int, session_id: str) -> str:
    return hashlib.sha256(f"{seed}|{class_id}|{session_id}".encode("utf-8")).hexdigest()


def build_split_manifest(
    session_rows: Sequence[Mapping[str, object]],
    *,
    dataset_fingerprint: str,
    seed: int = 20260805,
) -> dict[str, object]:
    """Assign complete sessions to immutable, class-stratified partitions."""
    by_class: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in session_rows:
        by_class[int(row["class_id"])].append(row)
    if set(by_class) != set(range(len(CLASS_NAMES))):
        raise ValueError(f"Expected all six classes, found {sorted(by_class)}")

    assignments: list[dict[str, object]] = []
    for class_id in range(len(CLASS_NAMES)):
        rows = sorted(
            by_class[class_id],
            key=lambda row: _stable_rank(seed, class_id, str(row["session_id"])),
        )
        counts = _partition_counts(len(rows))
        offset = 0
        # Allocate protected partitions first so their minimum session counts
        # cannot be consumed by training.
        allocation_order = ("validation", "calibration", "support", "final_query", "train")
        for partition in allocation_order:
            count = counts[partition]
            for row in rows[offset : offset + count]:
                assignments.append(
                    {
                        "session_id": str(row["session_id"]),
                        "class_id": class_id,
                        "class_name": CLASS_NAMES[class_id],
                        "partition": partition,
                        "window_count": int(row["window_count"]),
                        "date_token": str(row["date_token"]),
                        "source_token": str(row["source_token"]),
                    }
                )
            offset += count
        if offset != len(rows):
            raise AssertionError(f"Allocation did not consume class {class_id}: {offset} != {len(rows)}")

    session_ids = [row["session_id"] for row in assignments]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("A session was assigned to more than one partition")

    summary: dict[str, dict[str, dict[str, int]]] = {}
    for partition in PARTITIONS:
        summary[partition] = {}
        for class_id, class_name in enumerate(CLASS_NAMES):
            selected = [
                row for row in assignments
                if row["partition"] == partition and row["class_id"] == class_id
            ]
            summary[partition][class_name] = {
                "sessions": len(selected),
                "windows": sum(int(row["window_count"]) for row in selected),
            }

    payload: dict[str, object] = {
        "schema_version": 1,
        "name": "phi_otdr_session_split_v1",
        "seed": seed,
        "dataset_fingerprint_sha256": dataset_fingerprint,
        "grouping_unit": "recording session (filename prefix before _single_data_<window>)",
        "partition_order": list(PARTITIONS),
        "ratios": {"train": 0.60, "validation": 0.10, "calibration": 0.10, "support": 0.10, "final_query": 0.10},
        "final_query_access_policy": (
            "No training, preprocessing fit, early stopping, architecture selection, hyperparameter "
            "selection, support selection, or threshold calibration may use final_query windows."
        ),
        "known_metadata_limitations": [
            "No authoritative subject, environment, sensor, or acquisition-condition fields are supplied.",
            "source_token is not treated as a subject because its naming semantics change across batches.",
            "The split is session-disjoint but cannot guarantee subject- or environment-disjointness.",
        ],
        "summary": summary,
        "sessions": sorted(assignments, key=lambda row: str(row["session_id"])),
    }
    payload["manifest_sha256"] = _json_hash(payload)
    return payload


def create_split_from_audit(audit_dir: Path, output_path: Path, *, seed: int = 20260805) -> dict[str, object]:
    audit = json.loads((audit_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    with (audit_dir / "session_inventory.csv").open(newline="", encoding="utf-8") as handle:
        session_rows = list(csv.DictReader(handle))
    manifest = build_split_manifest(
        session_rows,
        dataset_fingerprint=str(audit["dataset_fingerprint_sha256"]),
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def verify_split_manifest(
    manifest: Mapping[str, object], *, expected_dataset_fingerprint: str | None = None
) -> dict[str, object]:
    """Validate a frozen manifest using canonical JSON, independent of file line endings."""
    payload = dict(manifest)
    stored_hash = str(payload.pop("manifest_sha256", ""))
    calculated_hash = _json_hash(payload)
    if stored_hash != calculated_hash:
        raise ValueError(f"Manifest hash mismatch: stored={stored_hash}, calculated={calculated_hash}")
    if expected_dataset_fingerprint is not None:
        observed = str(payload.get("dataset_fingerprint_sha256", ""))
        if observed != expected_dataset_fingerprint:
            raise ValueError(
                f"Dataset fingerprint mismatch: expected={expected_dataset_fingerprint}, observed={observed}"
            )
    sessions = list(payload.get("sessions", []))
    session_ids = [str(row["session_id"]) for row in sessions]
    if len(session_ids) != len(set(session_ids)):
        raise ValueError("Manifest assigns at least one session more than once")
    invalid = sorted({str(row["partition"]) for row in sessions} - set(PARTITIONS))
    if invalid:
        raise ValueError(f"Manifest has invalid partitions: {invalid}")
    return {
        "manifest_sha256": stored_hash,
        "dataset_fingerprint_sha256": str(payload["dataset_fingerprint_sha256"]),
        "session_count": len(sessions),
        "partitions": {name: sum(str(row["partition"]) == name for row in sessions) for name in PARTITIONS},
        "valid": True,
    }
