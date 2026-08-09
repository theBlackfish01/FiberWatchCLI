import numpy as np
from sklearn.metrics import confusion_matrix

from phi_research.shift_analysis_v1 import (
    _confusion_metrics,
    _macro_f1,
    _worst_recall,
)


def test_vectorized_confusion_metrics_match_direct_metrics():
    rng = np.random.default_rng(21)
    labels = rng.integers(0, 6, size=200)
    probabilities = rng.random((200, 6))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    matrix = confusion_matrix(
        labels, np.argmax(probabilities, axis=1), labels=np.arange(6)
    )
    macro_f1, worst = _confusion_metrics(matrix)
    assert np.isclose(macro_f1, _macro_f1(labels, probabilities))
    assert np.isclose(worst, _worst_recall(labels, probabilities))


def test_vectorized_confusion_metrics_accept_batches_and_absent_classes():
    matrices = np.zeros((2, 6, 6), dtype=int)
    matrices[0, 0, 0] = 4
    matrices[1, 1, 2] = 3
    macro_f1, worst = _confusion_metrics(matrices)
    assert macro_f1.shape == (2,)
    assert worst.tolist() == [1.0, 0.0]
