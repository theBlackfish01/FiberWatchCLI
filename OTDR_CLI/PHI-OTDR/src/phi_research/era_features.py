"""Repartition audited v1 feature evidence into locked acquisition-era bundles."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .data_contract import parse_sample_name
from .era_contract import verify_acquisition_manifest


CORE_KEYS = ("features", "labels", "sessions", "rel_paths", "partitions", "feature_names")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_legacy_bundles(paths: Sequence[Path]) -> dict[str, np.ndarray]:
    loaded = [np.load(path, allow_pickle=False) for path in paths]
    try:
        for bundle in loaded:
            missing = set(CORE_KEYS) - set(bundle.files)
            if missing:
                raise ValueError(f"Legacy feature bundle lacks keys: {sorted(missing)}")
        feature_names = loaded[0]["feature_names"].astype(str)
        for bundle in loaded[1:]:
            if not np.array_equal(feature_names, bundle["feature_names"].astype(str)):
                raise ValueError("Feature names differ between legacy bundles")
        arrays = {
            key: np.concatenate([bundle[key] for bundle in loaded], axis=0)
            for key in ("features", "labels", "sessions", "rel_paths", "partitions")
        }
        arrays["feature_names"] = feature_names
    finally:
        for bundle in loaded:
            bundle.close()
    rel_paths = arrays["rel_paths"].astype(str)
    if len(rel_paths) != len(set(rel_paths.tolist())):
        raise ValueError("Legacy bundles contain duplicate relative paths")
    return arrays


def _save_bundle(path: Path, arrays: Mapping[str, np.ndarray], mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=np.asarray(arrays["features"])[mask],
        labels=np.asarray(arrays["labels"])[mask],
        sessions=np.asarray(arrays["sessions"])[mask],
        rel_paths=np.asarray(arrays["rel_paths"])[mask],
        partitions=np.asarray(arrays["partitions"])[mask],
        window_ids=np.asarray(arrays["window_ids"])[mask],
        date_tokens=np.asarray(arrays["date_tokens"])[mask],
        source_tokens=np.asarray(arrays["source_tokens"])[mask],
        eras=np.asarray(arrays["eras"])[mask],
        feature_names=np.asarray(arrays["feature_names"]),
    )


def repartition_feature_bundles(
    legacy_paths: Sequence[Path],
    manifest: Mapping[str, object],
    output_dir: Path,
) -> dict[str, object]:
    """Create a development bundle and a physically separate locked query bundle."""
    verification = verify_acquisition_manifest(manifest)
    arrays = _load_legacy_bundles(legacy_paths)
    sessions = arrays["sessions"].astype(str)
    rel_paths = arrays["rel_paths"].astype(str)
    labels = arrays["labels"].astype(np.int64)
    session_map = {str(row["session_id"]): row for row in manifest["sessions"]}
    missing = sorted(set(sessions.tolist()) - set(session_map))
    if missing:
        raise ValueError(f"Feature sessions absent from acquisition manifest: {missing[:5]}")
    assigned_partitions: list[str] = []
    date_tokens: list[str] = []
    source_tokens: list[str] = []
    eras: list[str] = []
    window_ids: list[int] = []
    for session_id, rel_path, label in zip(sessions, rel_paths, labels, strict=True):
        row = session_map[session_id]
        if int(row["class_id"]) != int(label):
            raise ValueError(f"Manifest/feature class mismatch for {session_id}")
        parsed = parse_sample_name(Path(rel_path).name)
        if parsed.session_id != session_id:
            raise ValueError(f"Path/session mismatch: {rel_path} versus {session_id}")
        assigned_partitions.append(str(row["partition"]))
        date_tokens.append(str(row["date_token"]))
        source_tokens.append(str(row["source_token"]))
        eras.append(str(row["era"]))
        window_ids.append(parsed.window_id)
    arrays["partitions"] = np.asarray(assigned_partitions)
    arrays["date_tokens"] = np.asarray(date_tokens)
    arrays["source_tokens"] = np.asarray(source_tokens)
    arrays["eras"] = np.asarray(eras)
    arrays["window_ids"] = np.asarray(window_ids, dtype=np.int32)
    query = arrays["partitions"] == "target_query"
    development = ~query
    if not np.any(query) or not np.any(development):
        raise ValueError("Repartitioning produced an empty development or target-query bundle")
    development_path = output_dir / "development_features.npz"
    query_path = output_dir / "target_query_features.npz"
    _save_bundle(development_path, arrays, development)
    _save_bundle(query_path, arrays, query)

    output_rel_paths = np.concatenate((rel_paths[development], rel_paths[query]))
    if len(output_rel_paths) != len(rel_paths) or set(output_rel_paths) != set(rel_paths):
        raise AssertionError("Repartitioned bundles do not conserve the legacy feature rows")
    evidence: dict[str, object] = {
        "schema_version": 1,
        "protocol": "deterministic metadata-only repartition of audited v1 signal features",
        "manifest_sha256": verification["manifest_sha256"],
        "dataset_fingerprint_sha256": verification["dataset_fingerprint_sha256"],
        "direction": verification["direction"],
        "legacy_input_files": [
            {"path": path.as_posix(), "sha256": _file_sha256(path)} for path in legacy_paths
        ],
        "total_windows": int(len(rel_paths)),
        "total_sessions": int(len(set(sessions.tolist()))),
        "partition_windows": dict(sorted(Counter(assigned_partitions).items())),
        "partition_sessions": {
            partition: int(len(set(sessions[arrays["partitions"] == partition].tolist())))
            for partition in sorted(set(assigned_partitions))
        },
        "outputs": {
            "development": {
                "path": development_path.as_posix(),
                "sha256": _file_sha256(development_path),
                "windows": int(np.sum(development)),
                "sessions": int(len(set(sessions[development].tolist()))),
                "contains_target_query": False,
            },
            "target_query": {
                "path": query_path.as_posix(),
                "sha256": _file_sha256(query_path),
                "windows": int(np.sum(query)),
                "sessions": int(len(set(sessions[query].tolist()))),
                "partitions": ["target_query"],
            },
        },
    }
    (output_dir / "repartition_evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True), encoding="utf-8"
    )
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-features", type=Path, nargs="+", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    evidence = repartition_feature_bundles(args.legacy_features, manifest, args.output_dir)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
