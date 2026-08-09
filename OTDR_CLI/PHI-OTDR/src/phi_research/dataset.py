"""Manifest-backed sample indexing and strict MAT loading."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import scipy.io as sio

from .data_contract import CLASS_NAMES, FOLDER_LABELS, PARTITIONS, parse_sample_name


@dataclass(frozen=True)
class SampleRef:
    path: Path
    rel_path: str
    source_split: str
    class_id: int
    class_name: str
    session_id: str
    window_id: int
    partition: str


def load_manifest(path: Path) -> dict[str, object]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("name") not in {
        "phi_otdr_session_split_v1",
        "phi_otdr_acquisition_era_split_v2",
        "phi_otdr_acquisition_era_split_v3",
    }:
        raise ValueError(f"Unsupported split manifest: {manifest.get('name')!r}")
    sessions = manifest.get("sessions")
    if not isinstance(sessions, list) or not sessions:
        raise ValueError("Split manifest has no sessions")
    ids = [str(row["session_id"]) for row in sessions]
    if len(ids) != len(set(ids)):
        raise ValueError("Split manifest assigns a session more than once")
    return manifest


def _read_label_file(path: Path) -> list[tuple[Path, int]]:
    rows: list[tuple[Path, int]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.replace("\\", "/").split()
        if len(parts) != 2:
            raise ValueError(f"Malformed label line {path}:{line_number}")
        rows.append((Path(parts[0].lstrip("/")), int(parts[1])))
    return rows


def build_sample_index(
    data_root: Path,
    manifest_path: Path,
    *,
    partitions: Sequence[str] | None = None,
    class_ids: Iterable[int] | None = None,
) -> list[SampleRef]:
    manifest = load_manifest(manifest_path)
    available_partitions = set(manifest.get("partition_order", PARTITIONS))
    requested_partitions = available_partitions if partitions is None else set(partitions)
    unknown_partitions = requested_partitions - available_partitions
    if unknown_partitions:
        raise ValueError(f"Unknown partitions: {sorted(unknown_partitions)}")
    requested_classes = set(range(len(CLASS_NAMES))) if class_ids is None else set(class_ids)
    session_map = {
        str(row["session_id"]): (int(row["class_id"]), str(row["partition"]))
        for row in manifest["sessions"]
    }
    samples: list[SampleRef] = []
    identities: set[tuple[str, int]] = set()
    for source_split in ("train", "test"):
        split_root = data_root / source_split
        for rel_path, class_id in _read_label_file(split_root / "label.txt"):
            parsed = parse_sample_name(rel_path.name)
            if parsed.session_id not in session_map:
                raise ValueError(f"Session absent from split manifest: {parsed.session_id}")
            manifest_class, partition = session_map[parsed.session_id]
            if manifest_class != class_id:
                raise ValueError(f"Class disagreement for session {parsed.session_id}")
            folder_class = FOLDER_LABELS.get(rel_path.parts[0])
            if folder_class != class_id:
                raise ValueError(f"Folder/label disagreement: {source_split}/{rel_path}")
            identity = (parsed.session_id, parsed.window_id)
            if identity in identities:
                raise ValueError(f"Duplicate session/window identity: {identity}")
            identities.add(identity)
            if partition not in requested_partitions or class_id not in requested_classes:
                continue
            path = split_root / rel_path
            # The only currently unreadable entry is a zero-byte MAT file.
            # Unexpected nonempty corruption is a hard error in load_array.
            if not path.is_file() or path.stat().st_size == 0:
                continue
            samples.append(
                SampleRef(
                    path=path,
                    rel_path=f"{source_split}/{rel_path.as_posix()}",
                    source_split=source_split,
                    class_id=class_id,
                    class_name=CLASS_NAMES[class_id],
                    session_id=parsed.session_id,
                    window_id=parsed.window_id,
                    partition=partition,
                )
            )
    return sorted(samples, key=lambda row: (row.partition, row.class_id, row.session_id, row.window_id))


def load_array(sample: SampleRef) -> np.ndarray:
    raw = sio.loadmat(sample.path.as_posix())
    keys = [key for key in raw if not key.startswith("__")]
    if not keys:
        raise KeyError(f"No data array in {sample.path}")
    array = np.asarray(raw["data"] if "data" in raw else raw[keys[0]])
    if array.shape != (10000, 12):
        raise ValueError(f"Unexpected array shape {array.shape} in {sample.path}")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"Non-numeric data in {sample.path}: {array.dtype}")
    return array
