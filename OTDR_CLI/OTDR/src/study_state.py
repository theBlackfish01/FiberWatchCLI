from __future__ import annotations

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import tempfile
import time
import traceback
from typing import Any, Iterator

import torch


SCHEMA_VERSION = 1


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_payload(value: Any) -> Any:
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): canonical_payload(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [canonical_payload(v) for v in value]
    return value


def config_hash(config: Any, length: int = 12) -> str:
    encoded = json.dumps(canonical_payload(config), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def stable_run_id(approach: str, holdout: tuple[int, int], seed: int, config: Any) -> str:
    return f"{approach.lower()}-{holdout[0]:02d}_{holdout[1]:02d}-s{seed}-{config_hash(config)}"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(canonical_payload(payload), indent=2, sort_keys=True, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def append_jsonl(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(canonical_payload(payload), sort_keys=True, allow_nan=False) + "\n")
        handle.flush()


def artifact_hashes(run_dir: str | Path) -> dict[str, str]:
    root = Path(run_dir)
    ignored = {"manifest.json", "stdout.log", "stderr.log"}
    return {
        path.relative_to(root).as_posix(): file_sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name not in ignored
    }


def write_manifest(run_dir: str | Path, payload: dict[str, Any]) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        **canonical_payload(payload),
        "completed_at": utc_now(),
        "artifact_sha256": artifact_hashes(root),
    }
    atomic_json(root / "manifest.json", manifest)
    return manifest


def validate_run(run_dir: str | Path, expected: dict[str, Any] | None = None) -> tuple[bool, str]:
    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return False, "manifest missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"manifest unreadable: {exc}"
    if manifest.get("schema_version") != SCHEMA_VERSION:
        return False, "manifest schema mismatch"
    if expected:
        for key, value in expected.items():
            if manifest.get(key) != canonical_payload(value):
                return False, f"manifest field mismatch: {key}"
    hashes = manifest.get("artifact_sha256", {})
    if not hashes:
        return False, "artifact hashes missing"
    for relative, expected_hash in hashes.items():
        path = root / relative
        if not path.is_file() or file_sha256(path) != expected_hash:
            return False, f"artifact mismatch: {relative}"
    return True, "valid"


def environment_metadata(device: torch.device) -> dict[str, Any]:
    index = 0 if device.index is None else device.index
    packages = {}
    for name in ("numpy", "pandas", "scikit-learn", "scipy", "matplotlib", "sentence-transformers"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(index),
        "gpu_total_memory_bytes": torch.cuda.get_device_properties(index).total_memory,
        "packages": packages,
    }


class StudyState:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.state_path = self.root / "RUN_STATE.json"
        self.registry_path = self.root / "experiment_registry.jsonl"
        self.failures_path = self.root / "failures.jsonl"

    def _state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"schema_version": SCHEMA_VERSION, "completed_runs": [], "failed_runs": [], "selected_configs": {}}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def update(self, *, status: str | None = None, completed: str | None = None, failed: str | None = None,
               selected_configs: dict[str, Any] | None = None, note: str | None = None) -> None:
        state = self._state()
        state["schema_version"] = SCHEMA_VERSION
        state["updated_at"] = utc_now()
        if status is not None:
            state["status"] = status
        for key, value in (("completed_runs", completed), ("failed_runs", failed)):
            values = state.setdefault(key, [])
            if value is not None and value not in values:
                values.append(value)
        if completed is not None and completed in state.setdefault("failed_runs", []):
            state["failed_runs"].remove(completed)
        if selected_configs:
            state.setdefault("selected_configs", {}).update(canonical_payload(selected_configs))
        notes = state.setdefault("notes", [])
        if note and note not in notes:
            notes.append(note)
        state["notes"] = list(dict.fromkeys(notes))
        atomic_json(self.state_path, state)

    @contextmanager
    def run(self, run_id: str, run_dir: str | Path, metadata: dict[str, Any]) -> Iterator[dict[str, Any]]:
        root = Path(run_dir)
        root.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        record = {"event": "started", "run_id": run_id, "timestamp": utc_now(), **canonical_payload(metadata)}
        append_jsonl(self.registry_path, record)
        self.update(status="running")
        with (root / "stdout.log").open("a", encoding="utf-8") as stdout, (root / "stderr.log").open("a", encoding="utf-8") as stderr:
            try:
                with redirect_stdout(stdout), redirect_stderr(stderr):
                    yield record
            except Exception as exc:
                failure = {
                    "event": "failed", "run_id": run_id, "timestamp": utc_now(),
                    "duration_seconds": time.perf_counter() - started,
                    "exception_type": type(exc).__name__, "exception": str(exc),
                    "traceback": traceback.format_exc(), **canonical_payload(metadata),
                }
                append_jsonl(self.failures_path, failure)
                append_jsonl(self.registry_path, failure)
                self.update(status="running_with_failures", failed=run_id)
                raise
        complete = {
            "event": "completed", "run_id": run_id, "timestamp": utc_now(),
            "duration_seconds": time.perf_counter() - started, **canonical_payload(metadata),
        }
        append_jsonl(self.registry_path, complete)
        self.update(status="running", completed=run_id)
