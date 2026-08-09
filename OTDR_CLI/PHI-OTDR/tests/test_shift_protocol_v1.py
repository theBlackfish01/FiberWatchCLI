import json

from phi_research.data_contract import canonical_json_hash
from phi_research.shift_protocol_v1 import (
    finalize_payload,
    load_locked_config,
    process_memory_snapshot,
    verify_payload,
    write_csv,
)


def test_locked_config_is_lf_crlf_invariant(tmp_path):
    payload = {"z": [3, 2, 1], "a": {"value": True}}
    expected = canonical_json_hash(payload)
    sidecar = tmp_path / "config.sha256"
    sidecar.write_text(f"{expected}  config.json (canonical JSON)\n", encoding="utf-8")
    for newline in ("\n", "\r\n"):
        path = tmp_path / "config.json"
        text = json.dumps(payload, indent=2).replace("\n", newline)
        path.write_bytes(text.encode("utf-8"))
        observed, digest = load_locked_config(path, sidecar)
        assert observed == payload
        assert digest == expected


def test_payload_round_trip_and_csv_union(tmp_path):
    payload_path = tmp_path / "result.json"
    result = finalize_payload({"schema_version": 1, "value": 4}, payload_path)
    assert verify_payload(payload_path) == result
    csv_path = tmp_path / "rows.csv"
    write_csv(csv_path, [{"a": 1}, {"a": 2, "b": 3}])
    assert csv_path.read_text(encoding="utf-8").splitlines()[0] == "a,b"


def test_process_memory_snapshot_has_nonnegative_values():
    snapshot = process_memory_snapshot()
    assert set(snapshot) == {"working_set_bytes", "peak_working_set_bytes"}
    for value in snapshot.values():
        assert value is None or value >= 0
