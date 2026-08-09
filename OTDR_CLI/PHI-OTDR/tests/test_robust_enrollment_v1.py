import numpy as np

from phi_research.robust_enrollment_v1 import (
    _add_support_scores,
    _base_scores,
    _distance_probabilities,
    deterministic_support_draw,
)


def test_support_draw_is_deterministic_unique_and_ordered():
    candidates = np.asarray(["s4", "s1", "s3", "s2", "s5"])
    first = deterministic_support_draw(candidates, shot=3, seed=17)
    second = deterministic_support_draw(candidates[::-1], shot=3, seed=17)
    assert np.array_equal(first, second)
    assert len(set(first.tolist())) == 3
    assert first.tolist() == sorted(first.tolist())


def test_support_scores_and_probabilities_are_valid():
    gallery = np.asarray([[0.0, 0.0], [0.2, 0.0], [3.0, 3.0], [3.2, 3.0]])
    labels = np.asarray([0, 0, 1, 1])
    query = np.asarray([[0.1, 0.0], [6.0, 6.0]])
    scores, disagreement = _base_scores(
        query, gallery, labels, holdout=2, mode="gallery", neighbors=2
    )
    _add_support_scores(
        scores,
        disagreement,
        query,
        np.asarray([[6.0, 6.0], [6.2, 6.0]]),
        2,
        mode="gallery",
        neighbors=2,
    )
    probabilities = _distance_probabilities(scores, scale=1.0, temperature=1.0)
    assert probabilities.shape == (2, 6)
    assert np.allclose(np.sum(probabilities, axis=1), 1.0)
    assert np.argmax(probabilities[0]) == 0
    assert np.argmax(probabilities[1]) == 2
