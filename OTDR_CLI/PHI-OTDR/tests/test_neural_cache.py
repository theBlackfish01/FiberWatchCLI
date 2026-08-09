from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import Dataset


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.data_contract import canonical_json_hash
from phi_research.neural_data import cached_window_dataset


class _SyntheticWindows(Dataset):
    def __init__(self) -> None:
        self.samples = [
            SimpleNamespace(rel_path=f"train/x_{index}.mat", class_id=index % 2, session_id=f"s{index}")
            for index in range(4)
        ]
        self.normalization = "global_minmax"
        self.temporal_pool = 2

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        return {"data": torch.full((3, 2), float(index), dtype=torch.float32)}


def test_float32_window_cache_is_exact_and_reusable(tmp_path: Path) -> None:
    manifest = {
        "name": "phi_otdr_session_split_v1",
        "partition_order": ["train"],
        "sessions": [{"session_id": f"s{index}"} for index in range(4)],
    }
    manifest["manifest_sha256"] = canonical_json_hash(manifest)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    source = _SyntheticWindows()
    first = cached_window_dataset(source, manifest_path, tmp_path / "cache")
    second = cached_window_dataset(source, manifest_path, tmp_path / "cache")
    assert np.array_equal(first[3]["data"].numpy(), np.full((3, 2), 3.0, dtype=np.float32))
    assert np.array_equal(first[1]["data"].numpy(), second[1]["data"].numpy())
    assert first.metadata["cache_key_sha256"] == second.metadata["cache_key_sha256"]
