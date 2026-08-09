from __future__ import annotations

import copy
import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.data_contract import CLASS_NAMES
from phi_research.era_contract import (
    ERA_PARTITIONS,
    SOURCE_PARTITIONS,
    TARGET_PARTITIONS,
    acquisition_era,
    build_acquisition_manifest,
    verify_acquisition_manifest,
    verify_protocol_hash,
)


def _rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for era, date, count in (("january", "220112", 20), ("april_may", "220517", 12)):
        for class_id, class_name in enumerate(CLASS_NAMES):
            for index in range(count):
                rows.append(
                    {
                        "session_id": f"{date}_src_{class_name}_{index:02d}",
                        "class_id": class_id,
                        "class_name": class_name,
                        "date_token": date,
                        "source_token": "src",
                        "window_count": index + 1,
                    }
                )
    return rows


def test_acquisition_era_is_conservative() -> None:
    assert acquisition_era("220104") == "january"
    assert acquisition_era("220402") == "april_may"
    assert acquisition_era("220519") == "april_may"
    try:
        acquisition_era("220301")
    except ValueError as exc:
        assert "unsupported" in str(exc).lower()
    else:
        raise AssertionError("Unspecified date was silently assigned to an era")


def test_manifest_is_deterministic_complete_and_role_safe() -> None:
    rows = _rows()
    first = build_acquisition_manifest(
        rows,
        dataset_fingerprint="abc",
        source_era="january",
        target_era="april_may",
    )
    second = build_acquisition_manifest(
        copy.deepcopy(rows),
        dataset_fingerprint="abc",
        source_era="january",
        target_era="april_may",
    )
    assert first == second
    result = verify_acquisition_manifest(first, expected_dataset_fingerprint="abc")
    assert result["valid"] is True
    assert result["session_count"] == len(rows)
    assert set(result["partitions"]) == set(ERA_PARTITIONS)
    for row in first["sessions"]:
        if row["partition"] in SOURCE_PARTITIONS:
            assert row["era"] == "january"
        elif row["partition"] in TARGET_PARTITIONS:
            assert row["era"] == "april_may"


def test_target_support_has_seven_candidates_and_query_is_nonempty() -> None:
    manifest = build_acquisition_manifest(
        _rows(),
        dataset_fingerprint="abc",
        source_era="january",
        target_era="april_may",
    )
    for class_id in range(len(CLASS_NAMES)):
        support = [
            row for row in manifest["sessions"]
            if row["partition"] == "target_support" and row["class_id"] == class_id
        ]
        query = [
            row for row in manifest["sessions"]
            if row["partition"] == "target_query" and row["class_id"] == class_id
        ]
        assert len(support) == 7
        assert len(query) >= 3


def test_manifest_hash_survives_crlf_and_rejects_mutation() -> None:
    manifest = build_acquisition_manifest(
        _rows(),
        dataset_fingerprint="abc",
        source_era="april_may",
        target_era="january",
    )
    reconstructed = json.loads(json.dumps(manifest, indent=2).replace("\n", "\r\n"))
    assert verify_acquisition_manifest(reconstructed)["valid"] is True
    original = reconstructed["sessions"][0]["partition"]
    reconstructed["sessions"][0]["partition"] = (
        "source_train" if original != "source_train" else "target_query"
    )
    try:
        verify_acquisition_manifest(reconstructed)
    except ValueError as exc:
        assert "hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("Mutated manifest unexpectedly verified")


def test_protocol_hash_is_crlf_safe(tmp_path: Path) -> None:
    from phi_research.data_contract import canonical_json_hash

    payload = {"name": "test_protocol", "nested": {"answer": 42}}
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps(payload, indent=2).replace("\n", "\r\n"), encoding="utf-8")
    sidecar = tmp_path / "protocol.sha256"
    sidecar.write_text(f"{canonical_json_hash(payload)}  protocol.json (canonical JSON)\n", encoding="utf-8")
    assert verify_protocol_hash(protocol, sidecar)["valid"] is True
