from __future__ import annotations

import numpy as np

from OTDR_CLI.OTDR.src.tabpfn_incremental_memory import (
    CONFIRMATORY_CONFIG_SHA256,
    CONFIG_SHA256,
    _analysis_cohorts,
    _base_memory_indices,
    _bounded_linear_svm_probability,
    _expected_manifest,
    activate_study,
    load_config,
)
import pandas as pd


def test_incremental_memory_config_is_frozen_and_cuda_only() -> None:
    config = load_config()
    assert len(config["pairs"]) == 21
    assert config["device"] == "cuda:0"
    assert config["feature_regime"] == "summary_only"
    assert config["base_context_per_class"] == 20
    assert CONFIG_SHA256 == "9ec2e53831c947e05aa35d45e233b1db4e93cf8e58b440e6ff18f65083bb4ee6"


def test_base_memory_selection_is_fixed_balanced_and_seed_sensitive() -> None:
    labels = np.repeat(np.arange(3), 25)
    groups = tuple(f"class-{label}-group-{index}" for index, label in enumerate(labels))
    first = _base_memory_indices(
        labels,
        groups,
        base_ids=(0, 1, 2),
        count=5,
        context_seed=42,
        namespace="test-fixed-memory",
    )
    repeated = _base_memory_indices(
        labels,
        groups,
        base_ids=(0, 1, 2),
        count=5,
        context_seed=42,
        namespace="test-fixed-memory",
    )
    alternative = _base_memory_indices(
        labels,
        groups,
        base_ids=(0, 1, 2),
        count=5,
        context_seed=123,
        namespace="test-fixed-memory",
    )
    assert np.array_equal(first, repeated)
    assert not np.array_equal(first, alternative)
    assert len(first) == len(set(first)) == 15
    assert np.array_equal(np.bincount(labels[first], minlength=3), np.array([5, 5, 5]))


def test_incremental_memory_manifest_contract_records_frozen_protocol() -> None:
    expected = _expected_manifest((1, 2), 42)
    assert expected["run_id"] == "incremental-memory-pair-1-2-seed-42"
    assert expected["evidence_schema"] == 1
    assert expected["pilot_config_sha256"] == CONFIG_SHA256
    assert expected["requested_draws"] == 20
    assert expected["requested_shots"] == [1, 3, 5]


def test_confirmatory_memory_protocol_is_frozen_cuda_only_and_uses_fresh_seeds() -> None:
    try:
        activate_study("confirmatory")
        config = load_config()
        assert CONFIRMATORY_CONFIG_SHA256 == (
            "d6cd76a9a85d644455b66bd9972fa614fc54fbe64ddb3646a7df48dee7e14f6f"
        )
        assert config["device"] == "cuda:0"
        assert config["seeds"] == [7, 42, 123, 2026, 31415]
        assert config["execution_seeds"] == [7, 123, 2026, 31415]
        assert config["previously_observed_seed"] == 42
        assert config["base_context_per_class"] == 20
        assert config["outer_query_tuning"] is False
    finally:
        activate_study("pilot")


def test_confirmatory_analysis_keeps_replication_and_combined_cohorts_distinct() -> None:
    frame = pd.DataFrame(
        {
            "pair": ["1-2"] * 5,
            "seed": [7, 42, 123, 2026, 31415],
            "harmonic_mean": [0.9, 0.99, 0.91, 0.92, 0.93],
        }
    )
    try:
        activate_study("confirmatory")
        cohorts = _analysis_cohorts(frame)
        assert set(cohorts) == {"primary_replication", "combined_five_seed"}
        assert set(cohorts["primary_replication"]["seed"]) == {7, 123, 2026, 31415}
        assert set(cohorts["combined_five_seed"]["seed"]) == {
            7,
            42,
            123,
            2026,
            31415,
        }
    finally:
        activate_study("pilot")


def test_bounded_linear_svm_control_returns_finite_aligned_probabilities() -> None:
    rng = np.random.default_rng(20260802)
    context_x = rng.normal(size=(32, 3))
    context_y = np.repeat(np.arange(8), 4)
    query_x = rng.normal(size=(11, 3))
    probability, diagnostic = _bounded_linear_svm_probability(
        context_x, context_y, query_x
    )
    assert probability.shape == (11, 8)
    assert np.isfinite(probability).all()
    assert np.allclose(probability.sum(axis=1), 1.0)
    assert diagnostic["max_iter"] == 1000
    assert diagnostic["maximum_binary_iterations"] <= 1000
