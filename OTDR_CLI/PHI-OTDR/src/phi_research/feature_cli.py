"""Extract deterministic features without fitting on protected partitions."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .dataset import build_sample_index, load_array
from .features import extract_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partitions", nargs="+", required=True)
    args = parser.parse_args()

    samples = build_sample_index(args.data_root, args.manifest, partitions=args.partitions)
    started = time.perf_counter()
    rows: list[np.ndarray] = []
    feature_names: tuple[str, ...] | None = None
    for index, sample in enumerate(samples, start=1):
        feature = extract_features(load_array(sample))
        if feature_names is None:
            feature_names = feature.names
        elif feature.names != feature_names:
            raise AssertionError("Feature schema changed within one extraction run")
        rows.append(feature.values)
        if index % 500 == 0:
            print(f"[FEATURES] {index}/{len(samples)} in {time.perf_counter() - started:.1f}s", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=np.stack(rows),
        labels=np.asarray([sample.class_id for sample in samples], dtype=np.int64),
        sessions=np.asarray([sample.session_id for sample in samples]),
        rel_paths=np.asarray([sample.rel_path for sample in samples]),
        partitions=np.asarray([sample.partition for sample in samples]),
        feature_names=np.asarray(feature_names),
    )
    metadata = {
        "sample_count": len(samples),
        "feature_count": len(feature_names or ()),
        "partitions": args.partitions,
        "elapsed_seconds": time.perf_counter() - started,
        "final_query_included": "final_query" in args.partitions,
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
