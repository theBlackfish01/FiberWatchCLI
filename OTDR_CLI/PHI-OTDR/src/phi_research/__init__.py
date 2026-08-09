"""Leakage-resistant research utilities for the local Phi-OTDR dataset."""

from .data_contract import CLASS_NAMES, PARTITIONS, build_split_manifest, parse_sample_name

__all__ = ["CLASS_NAMES", "PARTITIONS", "build_split_manifest", "parse_sample_name"]
