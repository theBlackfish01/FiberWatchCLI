"""Build only the deterministic morphology-attribute caches needed by shift-v1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .morphology_attributes_v3 import (
    aggregate_attribute_sessions,
    derive_window_attributes,
)
from .shift_protocol_v1 import finalize_payload, sha256_file


def run(bundle_path: Path, output_dir: Path) -> dict[str, object]:
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    attributes, attribute_names = derive_window_attributes(
        bundle["features"], bundle["feature_names"]
    )
    session_attributes, sessions, session_attribute_names = aggregate_attribute_sessions(
        attributes, bundle["sessions"], bundle["window_ids"]
    )
    first = {
        session: int(np.flatnonzero(bundle["sessions"].astype(str) == session)[0])
        for session in sessions
    }
    labels = np.asarray([bundle["labels"][first[session]] for session in sessions], dtype=np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    window_path = output_dir / "window_attributes.npz"
    session_path = output_dir / "session_attributes.npz"
    np.savez_compressed(
        window_path,
        attributes=attributes,
        attribute_names=np.asarray(attribute_names),
        sessions=bundle["sessions"],
        window_ids=bundle["window_ids"],
        rel_paths=bundle["rel_paths"],
    )
    np.savez_compressed(
        session_path,
        attributes=session_attributes,
        attribute_names=np.asarray(session_attribute_names),
        sessions=sessions,
        labels=labels,
    )
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "PHI-OTDR shift-v1 minimal attribute cache",
        "bundle_sha256": sha256_file(bundle_path),
        "window_attributes_sha256": sha256_file(window_path),
        "session_attributes_sha256": sha256_file(session_path),
        "windows": len(attributes),
        "sessions": len(sessions),
        "window_features": attributes.shape[1],
        "session_features": session_attributes.shape[1],
    }
    return finalize_payload(payload, output_dir / "cache_build.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.bundle, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
