from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.data_contract import CLASS_NAMES
from phi_research.era_contract import build_acquisition_manifest
from phi_research.era_features import repartition_feature_bundles


def _manifest_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for date, count in (("220112", 12), ("220517", 12)):
        for class_id, class_name in enumerate(CLASS_NAMES):
            for index in range(count):
                rows.append(
                    {
                        "session_id": f"{date}_src_{class_name}_{index:02d}",
                        "class_id": class_id,
                        "class_name": class_name,
                        "date_token": date,
                        "source_token": "src",
                        "window_count": 1,
                    }
                )
    return rows


def test_repartition_conserves_rows_and_physically_separates_query(tmp_path: Path) -> None:
    rows = _manifest_rows()
    manifest = build_acquisition_manifest(
        rows,
        dataset_fingerprint="abc",
        source_era="january",
        target_era="april_may",
    )
    features = np.arange(len(rows) * 3, dtype=np.float32).reshape(len(rows), 3)
    labels = np.asarray([int(row["class_id"]) for row in rows])
    sessions = np.asarray([str(row["session_id"]) for row in rows])
    rel_paths = np.asarray(
        [f"train/01_background/{session}_single_data_1.mat" for session in sessions]
    )
    partitions = np.asarray(["train"] * len(rows))
    names = np.asarray(["raw_mean", "diff_std", "spectrum_bin_1"])
    first, second = tmp_path / "one.npz", tmp_path / "two.npz"
    midpoint = len(rows) // 2
    for path, selection in ((first, slice(0, midpoint)), (second, slice(midpoint, None))):
        np.savez_compressed(
            path,
            features=features[selection],
            labels=labels[selection],
            sessions=sessions[selection],
            rel_paths=rel_paths[selection],
            partitions=partitions[selection],
            feature_names=names,
        )
    output = tmp_path / "output"
    evidence = repartition_feature_bundles((first, second), manifest, output)
    development = np.load(output / "development_features.npz", allow_pickle=False)
    query = np.load(output / "target_query_features.npz", allow_pickle=False)
    try:
        assert "target_query" not in set(development["partitions"].astype(str))
        assert set(query["partitions"].astype(str)) == {"target_query"}
        assert set(development["rel_paths"].astype(str)).isdisjoint(
            set(query["rel_paths"].astype(str))
        )
        assert len(development["labels"]) + len(query["labels"]) == len(rows)
        assert "window_ids" in development.files
        assert evidence["total_windows"] == len(rows)
    finally:
        development.close()
        query.close()
