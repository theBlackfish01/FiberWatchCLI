from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.episodic_multiview import sample_episode


def test_episode_support_and_query_are_disjoint_and_balanced() -> None:
    labels = np.repeat(np.arange(5), 10)
    support, support_y, query, query_y = sample_episode(
        labels, list(range(5)), np.random.default_rng(7), support_per_class=3, query_per_class=4
    )
    assert set(support).isdisjoint(set(query))
    np.testing.assert_array_equal(np.bincount(support_y), np.repeat(3, 5))
    np.testing.assert_array_equal(np.bincount(query_y), np.repeat(4, 5))
