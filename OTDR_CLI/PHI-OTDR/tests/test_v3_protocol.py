from __future__ import annotations

import copy

import pytest

from phi_research.data_contract import canonical_json_hash
from phi_research.era_contract import verify_acquisition_manifest
from phi_research.v3_protocol import upgrade_manifest


def _v2_manifest() -> dict[str, object]:
    classes = ("background", "digging", "knocking", "watering", "shaking", "walking")
    sessions = []
    partitions = (
        "source_train",
        "source_validation",
        "source_calibration",
        "target_calibration",
        "target_support",
        "target_query",
    )
    for p_index, partition in enumerate(partitions):
        source = partition.startswith("source_")
        for class_id, class_name in enumerate(classes):
            session_id = f"{'220101' if source else '220401'}_x_{class_name}_{p_index}"
            sessions.append(
                {
                    "session_id": session_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "date_token": "220101" if source else "220401",
                    "source_token": "x",
                    "era": "january" if source else "april_may",
                    "role": "source" if source else "target",
                    "partition": partition,
                    "window_count": 1,
                }
            )
    payload = {
        "schema_version": 2,
        "name": "phi_otdr_acquisition_era_split_v2",
        "dataset_fingerprint_sha256": "old",
        "direction": {"source": "january", "target": "april_may"},
        "partition_order": list(partitions),
        "sessions": sessions,
    }
    payload["manifest_sha256"] = canonical_json_hash(payload)
    return payload


def test_upgrade_preserves_assignments_and_updates_counts() -> None:
    v2 = _v2_manifest()
    original = copy.deepcopy(v2)
    inventory = {
        row["session_id"]: {"session_id": row["session_id"], "class_id": str(row["class_id"]), "window_count": "3"}
        for row in v2["sessions"]
    }
    v3 = upgrade_manifest(v2, complete_fingerprint="complete", inventory=inventory)
    assert v2 == original
    assert v3["name"] == "phi_otdr_acquisition_era_split_v3"
    assert all(row["window_count"] == 3 for row in v3["sessions"])
    assert [
        (row["session_id"], row["partition"]) for row in v3["sessions"]
    ] == sorted(
        (row["session_id"], row["partition"]) for row in v2["sessions"]
    )
    assert verify_acquisition_manifest(v3, expected_dataset_fingerprint="complete")["valid"]


def test_upgrade_rejects_session_drift() -> None:
    v2 = _v2_manifest()
    with pytest.raises(ValueError, match="session mismatch"):
        upgrade_manifest(v2, complete_fingerprint="complete", inventory={})
