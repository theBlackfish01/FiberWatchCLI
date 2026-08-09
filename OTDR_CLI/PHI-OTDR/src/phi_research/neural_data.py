"""PyTorch datasets bound to the immutable session manifest."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .data_contract import canonical_json_hash
from .dataset import SampleRef, build_sample_index, load_array, load_manifest


def normalize_window(array: np.ndarray, mode: str) -> np.ndarray:
    values = array.astype(np.float32, copy=False)
    if mode == "global_minmax":
        minimum = float(values.min())
        maximum = float(values.max())
        if maximum <= minimum:
            return np.zeros_like(values)
        return (values - minimum) / (maximum - minimum)
    if mode == "channel_zscore":
        mean = np.mean(values, axis=0, keepdims=True)
        std = np.std(values, axis=0, keepdims=True)
        return (values - mean) / np.maximum(std, 1e-6)
    if mode == "global_zscore":
        mean = float(np.mean(values))
        std = float(np.std(values))
        return (values - mean) / max(std, 1e-6)
    raise ValueError(f"Unknown normalization mode: {mode}")


class ManifestWindowDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        partitions: Sequence[str],
        *,
        normalization: str = "global_minmax",
        temporal_pool: int = 1,
        class_ids: Sequence[int] | None = None,
    ) -> None:
        self.samples: list[SampleRef] = build_sample_index(
            data_root, manifest_path, partitions=partitions, class_ids=class_ids
        )
        self.normalization = normalization
        if temporal_pool < 1:
            raise ValueError("temporal_pool must be positive")
        self.temporal_pool = temporal_pool

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        array = normalize_window(load_array(sample), self.normalization)
        if self.temporal_pool > 1:
            usable = len(array) - (len(array) % self.temporal_pool)
            array = array[:usable].reshape(-1, self.temporal_pool, array.shape[1]).mean(axis=1)
        return {
            "data": torch.from_numpy(np.ascontiguousarray(array)),
            "label": sample.class_id,
            "session": sample.session_id,
            "rel_path": sample.rel_path,
        }


class CachedWindowDataset(Dataset):
    """Read a deterministic pooled-window cache instead of reparsing MAT files."""

    def __init__(self, source: ManifestWindowDataset, data_path: Path, metadata: dict[str, object]) -> None:
        self.samples = source.samples
        self.normalization = source.normalization
        self.temporal_pool = source.temporal_pool
        self.metadata = metadata
        self._data = np.load(data_path, mmap_mode="r", allow_pickle=False)
        if len(self._data) != len(self.samples):
            raise ValueError("Cached array/sample count mismatch")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        sample = self.samples[index]
        # A writable contiguous batch copy avoids undefined behavior when
        # PyTorch wraps a read-only memory-mapped view.
        values = np.array(self._data[index], dtype=np.float32, copy=True, order="C")
        return {
            "data": torch.from_numpy(values),
            "label": sample.class_id,
            "session": sample.session_id,
            "rel_path": sample.rel_path,
        }


def cached_window_dataset(
    source: ManifestWindowDataset,
    manifest_path: Path,
    cache_dir: Path,
) -> CachedWindowDataset:
    """Build once using exact float32 normalization, then reuse across models."""
    manifest = load_manifest(manifest_path)
    identity_payload = {
        "schema_version": 1,
        "manifest_sha256": manifest.get("manifest_sha256", canonical_json_hash(manifest)),
        "normalization": source.normalization,
        "temporal_pool": source.temporal_pool,
        "samples": [sample.rel_path for sample in source.samples],
        "dtype": "float32",
    }
    cache_key = canonical_json_hash(identity_payload)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_path = cache_dir / "windows.npy"
    metadata_path = cache_dir / "cache.json"
    if data_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("cache_key_sha256") == cache_key:
            cached = CachedWindowDataset(source, data_path, metadata)
            print(
                f"[CACHE] reused {len(cached)} windows from {data_path} "
                f"({data_path.stat().st_size / 2**20:.1f} MiB)",
                flush=True,
            )
            return cached
    if not source.samples:
        raise ValueError("Cannot cache an empty window dataset")
    first = source[0]["data"].numpy()
    shape = (len(source), *first.shape)
    partial_path = cache_dir / "windows.partial.npy"
    mapped = np.lib.format.open_memmap(partial_path, mode="w+", dtype=np.float32, shape=shape)
    mapped[0] = first
    started = time.perf_counter()
    for index in range(1, len(source)):
        mapped[index] = source[index]["data"].numpy()
        if (index + 1) % 250 == 0:
            print(
                f"[CACHE] materialized {index + 1}/{len(source)} windows in "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    mapped.flush()
    del mapped
    partial_path.replace(data_path)
    metadata = {
        **identity_payload,
        "cache_key_sha256": cache_key,
        "shape": list(shape),
        "window_count": len(source),
        "session_count": len({sample.session_id for sample in source.samples}),
        "bytes": data_path.stat().st_size,
        "build_seconds": time.perf_counter() - started,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return CachedWindowDataset(source, data_path, metadata)
