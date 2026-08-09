from __future__ import annotations

import numpy as np
import torch

from phi_research.session_neural_v3 import SessionDataset, SessionNet, collate_sessions


def test_session_dataset_orders_windows_and_normalizes_time() -> None:
    features = np.asarray([[2.0], [1.0], [3.0]], dtype=np.float32)
    dataset = SessionDataset(
        features,
        np.asarray([0, 0, 1]),
        np.asarray(["a", "a", "b"]),
        np.asarray([2, 1, 1]),
        ["a", "b"],
    )
    session, label, values = dataset[0]
    assert session == "a" and label == 0
    assert values[:, 0].tolist() == [1.0, 2.0]
    assert values[:, -1].tolist() == [-1.0, 1.0]


def test_masked_model_ignores_padded_values() -> None:
    torch.manual_seed(1)
    model = SessionNet(3, hidden_dim=8, dropout=0.0, architecture="attention").eval()
    values = torch.randn(2, 3, 3)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    first = model(values, mask)
    values[0, 2] = 10000.0
    second = model(values, mask)
    assert torch.allclose(first[0], second[0], atol=1e-6)


def test_collate_sessions_pads_and_masks() -> None:
    batch = [
        ("a", 0, np.ones((2, 3), dtype=np.float32)),
        ("b", 1, np.ones((1, 3), dtype=np.float32)),
    ]
    sessions, labels, values, mask = collate_sessions(batch)
    assert sessions == ["a", "b"]
    assert labels.tolist() == [0, 1]
    assert values.shape == (2, 2, 3)
    assert mask.tolist() == [[True, True], [True, False]]
