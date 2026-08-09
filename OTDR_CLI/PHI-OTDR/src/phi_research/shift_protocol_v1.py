"""Shared frozen-protocol and artifact helpers for the PHI shift study."""

from __future__ import annotations

import csv
import ctypes
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from .data_contract import canonical_json_hash


def process_memory_snapshot() -> dict[str, int | None]:
    """Return current and peak resident memory without an added dependency."""
    if hasattr(ctypes, "windll"):
        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        get_current_process = ctypes.windll.kernel32.GetCurrentProcess
        get_current_process.restype = ctypes.c_void_p
        get_process_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_process_memory.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(Counters),
            ctypes.c_ulong,
        ]
        get_process_memory.restype = ctypes.c_int
        process = get_current_process()
        success = get_process_memory(
            process, ctypes.byref(counters), counters.cb
        )
        if success:
            return {
                "working_set_bytes": int(counters.WorkingSetSize),
                "peak_working_set_bytes": int(counters.PeakWorkingSetSize),
            }
    return {"working_set_bytes": None, "peak_working_set_bytes": None}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_locked_config(config_path: Path, hash_path: Path) -> tuple[dict[str, object], str]:
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    expected = hash_path.read_text(encoding="utf-8").split()[0]
    observed = canonical_json_hash(payload)
    if observed != expected:
        raise ValueError(f"Frozen configuration hash mismatch: {observed} != {expected}")
    return payload, expected


def write_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"Refusing to create empty artifact: {path}")
    fieldnames = list(materialized[0])
    seen = set(fieldnames)
    for row in materialized[1:]:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(materialized)


def finalize_payload(payload: dict[str, object], path: Path) -> dict[str, object]:
    payload["payload_sha256"] = canonical_json_hash(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def verify_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = str(payload.pop("payload_sha256"))
    observed = canonical_json_hash(payload)
    if stored != observed:
        raise ValueError(f"Artifact payload hash mismatch: {observed} != {stored}")
    payload["payload_sha256"] = stored
    return payload
