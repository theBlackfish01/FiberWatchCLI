from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.ontology_audit_v3 import filename_attributes


def test_background_nearby_activity_is_audit_only_subtype() -> None:
    attributes = filename_attributes("220509_bkg_13_heavy_steps_50cm_away", "background")
    assert attributes["nearby_non_target_activity"] is True
    assert attributes["background_subtype"] == "nearby_non_target_activity"
    assert attributes["distance_cm"] == 50


def test_filename_activity_attributes_are_conservative() -> None:
    attributes = filename_attributes("220518_knocking_edge_slow", "knocking")
    assert attributes["speed"] == "slow"
    assert attributes["spatial_contact"] == "edge"
    assert attributes["nearby_non_target_activity"] is False
    assert attributes["duration_seconds"] is None


def test_duration_and_locomotion_tokens_are_parsed() -> None:
    attributes = filename_attributes("220509_bkg_04_walk30s", "background")
    assert attributes["locomotion"] == "walk"
    assert attributes["duration_seconds"] == 30
