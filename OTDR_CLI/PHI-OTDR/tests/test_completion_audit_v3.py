from __future__ import annotations

import json
import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.completion_audit_v3 import _integrity, _sidecar_integrity
from phi_research.data_contract import canonical_json_hash


def test_sidecar_hash_is_crlf_safe(tmp_path: Path) -> None:
    payload = {"name": "registration", "nested": {"enabled": True}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    sidecar = tmp_path / "config.sha256"
    sidecar.write_text(f"{canonical_json_hash(payload)}  config.json (canonical JSON)\n", encoding="utf-8")
    result = _sidecar_integrity(config, sidecar)
    assert result["status"] == "pass"
    assert result["lf_hash"] == result["simulated_crlf_hash"]


def test_payload_integrity_detects_mutation(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = {"metric": 0.5}
    payload["payload_sha256"] = canonical_json_hash(payload)
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert _integrity(path, "payload_sha256")[0]
    payload["metric"] = 0.6
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not _integrity(path, "payload_sha256")[0]
