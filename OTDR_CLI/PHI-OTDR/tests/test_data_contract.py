from __future__ import annotations

import copy
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.data_contract import (
    CLASS_NAMES, PARTITIONS, build_split_manifest, parse_sample_name, verify_split_manifest,
)


def test_parse_sample_name_preserves_complete_session_prefix() -> None:
    parsed = parse_sample_name("220112_cxm_background_01_single_data_32.mat")
    assert parsed.session_id == "220112_cxm_background_01"
    assert parsed.window_id == 32
    assert parsed.date_token == "220112"
    assert parsed.source_token == "cxm"


def test_parse_later_naming_variant_does_not_invent_subject() -> None:
    parsed = parse_sample_name("220517_shaking_03_single_data_2.mat")
    assert parsed.session_id == "220517_shaking_03"
    assert parsed.source_token == "shaking"


def _session_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        for index in range(60):
            rows.append(
                {
                    "session_id": f"220101_source_{class_name}_{index:02d}",
                    "class_id": class_id,
                    "class_name": class_name,
                    "date_token": "220101",
                    "source_token": "source",
                    "window_count": index + 1,
                }
            )
    return rows


def test_split_is_deterministic_and_session_disjoint() -> None:
    rows = _session_rows()
    first = build_split_manifest(rows, dataset_fingerprint="abc")
    second = build_split_manifest(copy.deepcopy(rows), dataset_fingerprint="abc")
    assert first == second
    session_ids = [row["session_id"] for row in first["sessions"]]
    assert len(session_ids) == len(set(session_ids))


def test_every_class_has_protected_sessions_and_all_windows_are_accounted_for() -> None:
    rows = _session_rows()
    manifest = build_split_manifest(rows, dataset_fingerprint="abc")
    for partition in PARTITIONS:
        for class_name in CLASS_NAMES:
            entry = manifest["summary"][partition][class_name]
            assert entry["sessions"] >= 5
            assert entry["windows"] > 0
    expected_windows = sum(int(row["window_count"]) for row in rows)
    observed_windows = sum(
        manifest["summary"][partition][class_name]["windows"]
        for partition in PARTITIONS
        for class_name in CLASS_NAMES
    )
    assert observed_windows == expected_windows


def test_changing_seed_changes_assignments_without_changing_membership() -> None:
    rows = _session_rows()
    first = build_split_manifest(rows, dataset_fingerprint="abc", seed=1)
    second = build_split_manifest(rows, dataset_fingerprint="abc", seed=2)
    first_map = {row["session_id"]: row["partition"] for row in first["sessions"]}
    second_map = {row["session_id"]: row["partition"] for row in second["sessions"]}
    assert set(first_map) == set(second_map)
    assert first_map != second_map


def test_manifest_hash_verifies_after_crlf_json_round_trip() -> None:
    import json

    manifest = build_split_manifest(_session_rows(), dataset_fingerprint="abc")
    windows_text = json.dumps(manifest, indent=2).replace("\n", "\r\n")
    reconstructed = json.loads(windows_text)
    result = verify_split_manifest(reconstructed, expected_dataset_fingerprint="abc")
    assert result["valid"] is True
    assert result["session_count"] == 360


def test_manifest_verifier_rejects_mutation() -> None:
    manifest = build_split_manifest(_session_rows(), dataset_fingerprint="abc")
    manifest["sessions"][0]["partition"] = "validation"
    try:
        verify_split_manifest(manifest)
    except ValueError as exc:
        assert "hash mismatch" in str(exc).lower()
    else:
        raise AssertionError("Mutated manifest unexpectedly verified")
