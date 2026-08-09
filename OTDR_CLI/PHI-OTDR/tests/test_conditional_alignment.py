from __future__ import annotations

import sys
from pathlib import Path

import torch


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from phi_research.conditional_alignment import conditional_center_alignment, choose_alignment_weight


def test_conditional_alignment_ignores_cross_class_domain_confounds() -> None:
    embedding = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    domains = torch.tensor([0, 0, 1, 1])
    assert float(conditional_center_alignment(embedding, labels, domains)) == 0.0


def test_conditional_alignment_penalizes_within_class_domain_shift() -> None:
    embedding = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], requires_grad=True
    )
    labels = torch.tensor([0, 0, 0, 0])
    domains = torch.tensor([0, 0, 1, 1])
    loss = conditional_center_alignment(embedding, labels, domains)
    assert float(loss.detach()) > 0
    loss.backward()
    assert embedding.grad is not None


def _result(weight: float, class_f1: float, date_f1: float) -> dict[str, object]:
    return {
        "alignment_weight": weight,
        "validation": {"session_metrics": {"macro_f1": class_f1}},
        "metadata_probe": {"date_token": {"validation": {"macro_f1": date_f1}}},
    }


def test_alignment_gate_requires_nuisance_reduction_without_class_collapse() -> None:
    selection = choose_alignment_weight(
        [_result(0.0, 0.95, 0.80), _result(0.01, 0.94, 0.70), _result(0.20, 0.80, 0.20)]
    )
    assert selection["decision"] == "continue"
    assert selection["selected_alignment_weight"] == 0.01
