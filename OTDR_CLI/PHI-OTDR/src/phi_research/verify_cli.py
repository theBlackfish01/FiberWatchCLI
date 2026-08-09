"""Read-only verification for frozen Phi-OTDR research contracts and artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data_contract import verify_split_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--development-features", type=Path)
    parser.add_argument("--final-features", type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    expected_fingerprint = None
    if args.audit:
        audit = json.loads(args.audit.read_text(encoding="utf-8"))
        expected_fingerprint = str(audit["dataset_fingerprint_sha256"])
    result = {"manifest": verify_split_manifest(manifest, expected_dataset_fingerprint=expected_fingerprint)}
    feature_results: dict[str, object] = {}
    for label, path, expected_final in (
        ("development", args.development_features, False), ("final", args.final_features, True)
    ):
        if path is None:
            continue
        bundle = np.load(path, allow_pickle=False)
        partitions = set(bundle["partitions"].astype(str).tolist())
        has_final = "final_query" in partitions
        if has_final != expected_final:
            raise ValueError(f"{label} bundle final-query contract is invalid: {sorted(partitions)}")
        feature_results[label] = {
            "sample_count": int(len(bundle["labels"])),
            "feature_count": int(bundle["features"].shape[1]),
            "partitions": sorted(partitions),
        }
    result["features"] = feature_results
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
