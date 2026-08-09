from __future__ import annotations

import torch

from phi_research.siamese_session_v3 import SiameseSessionEncoder, supervised_pair_loss


def test_siamese_embedding_is_normalized() -> None:
    model = SiameseSessionEncoder(5, 8, 4).eval()
    embedding, logits = model(torch.randn(6, 5))
    assert embedding.shape == (6, 4)
    assert logits.shape == (6, 6)
    assert torch.allclose(torch.linalg.norm(embedding, dim=1), torch.ones(6), atol=1e-6)


def test_pair_loss_prefers_same_class_similarity_and_different_class_separation() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    good = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    bad = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    assert supervised_pair_loss(good, labels, margin=0.2) < supervised_pair_loss(bad, labels, margin=0.2)
