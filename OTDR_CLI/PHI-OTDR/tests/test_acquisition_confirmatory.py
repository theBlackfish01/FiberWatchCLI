import json

import numpy as np
import pytest

from phi_research.acquisition_confirmatory import class_conditional_conformal, verify_lock
from phi_research.data_contract import canonical_json_hash


def test_class_conditional_conformal_uses_only_matching_class() -> None:
    calibration_scores = np.asarray([[-1.0, -9.0], [-2.0, -8.0], [-7.0, -1.0], [-8.0, -2.0]])
    labels = np.asarray([0, 0, 1, 1])
    query = np.asarray([[-1.5, -1.5], [-20.0, -20.0]])
    result = class_conditional_conformal(calibration_scores, labels, query, [0, 1], alpha=0.40)
    assert result["calibration_counts"].tolist() == [2, 2]
    assert result["p_values"][0].tolist() == pytest.approx([2 / 3, 2 / 3])
    assert result["set_sizes"].tolist() == [2, 0]


def test_verify_lock_is_canonical_and_detects_change(tmp_path) -> None:
    config = tmp_path / "lock.json"
    sidecar = tmp_path / "lock.sha256"
    payload = {"final_query_used_at_freeze": False, "value": 3}
    config.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sidecar.write_text(f"{canonical_json_hash(payload)}  lock.json\n", encoding="utf-8")
    assert verify_lock(config, sidecar) == canonical_json_hash(payload)
    config.write_text(json.dumps({**payload, "value": 4}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_lock(config, sidecar)
