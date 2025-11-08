from __future__ import annotations

"""TabNet multitask model for OTDR traces.

Uses **pytorch‑tabnet** as a backbone and adds two simple heads so the forward
signature matches the existing `OTDR_TCN` and `TimeSeriesTransformer` models.

Public API
----------
* ``OTDR_TabNet`` – backbone + heads returning ``(logits, pos_pred)``.
* ``TrainConfig`` – hyper‑parameters mirroring the other models.
* ``train_tabnet`` – early‑stopping training loop.
* ``predict`` – batched inference helper.

Installation note
-----------------
This model requires the *pytorch‑tabnet* package:

```bash
pip install pytorch-tabnet
```
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

try:
    from pytorch_tabnet.tab_network import TabNet  # type: ignore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "pytorch-tabnet is required for models.tabnet; install with\n"
        "    pip install pytorch-tabnet"
    ) from e

__all__ = [
    "OTDR_TabNet",
    "TrainConfig",
    "train_tabnet",
    "predict",
]


# ---------------------------------------------------------------------------
# Model                                                                       |
# ---------------------------------------------------------------------------


class OTDR_TabNet(nn.Module):
    """TabNet backbone with dual heads (classification + localisation)."""

    def __init__(
            self,
            *,
            input_dim: int = 31,
            n_classes: int = 8,
            n_d: int = 64,
            n_a: int = 64,
            n_steps: int = 5,
            gamma: float = 1.5,
            n_independent: int = 2,
            n_shared: int = 2,
            momentum: float = 0.02,
            lambda_sparse: float = 1e-4,
    ) -> None:
        super().__init__()
        self.backbone = TabNet(
            input_dim=input_dim,
            output_dim=n_d,
            n_d=n_d,
            n_a=n_a,
            n_steps=n_steps,
            gamma=gamma,
            n_independent=n_independent,
            n_shared=n_shared,
            momentum=momentum
        )
        self.class_head = nn.Linear(n_d, n_classes)
        self.loc_head = nn.Linear(n_d, 1)

    # -------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input shape ``(B, F)`` or ``(B, 1, F)``. Returns (logits, pos_pred)."""
        if x.dim() == 3:  # (B, 1, F)
            x = x.squeeze(1)
        features, _ = self.backbone(x)  # features: (B, n_d)
        return self.class_head(features), self.loc_head(features).squeeze(-1)


# ---------------------------------------------------------------------------
# Training                                                                    |
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-5
    patience: int = 10
    step_size: int = 20
    gamma: float = 0.5
    lambda_loc: float = 0.5
    device: torch.device | str | None = None
    save_path: str | Path | None = None


def _val_metrics(
        model: OTDR_TabNet,
        loader: DataLoader,
        *,
        loss_cls: nn.Module,
        loss_loc: nn.Module,
        lambda_loc: float,
        device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    v_loss = 0.0
    v_correct = 0
    v_samples = 0
    mse_sum = 0.0
    with torch.no_grad():
        for xb, y_cls, y_pos in loader:
            xb = xb.to(device)
            y_cls = y_cls.to(device)
            y_pos = y_pos.to(device)
            logits, pos_hat = model(xb)
            loss = loss_cls(logits, y_cls) + lambda_loc * loss_loc(pos_hat, y_pos)
            v_loss += loss.item() * xb.size(0)
            v_correct += (logits.argmax(1) == y_cls).sum().item()
            mse_sum += loss_loc(pos_hat, y_pos).item() * xb.size(0)
            v_samples += xb.size(0)
    v_loss /= v_samples
    acc = v_correct / v_samples
    rmse = (mse_sum / v_samples) ** 0.5
    return v_loss, acc, rmse


def train_tabnet(
        model: OTDR_TabNet,
        train_X: torch.Tensor,
        train_y_cls: torch.Tensor,
        train_y_pos: torch.Tensor,
        val_X: torch.Tensor,
        val_y_cls: torch.Tensor,
        val_y_pos: torch.Tensor,
        cfg: TrainConfig | None = None,
) -> OTDR_TabNet:
    cfg = cfg or TrainConfig()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device is None else torch.device(cfg.device)
    )
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(train_X, train_y_cls, train_y_pos.view(-1)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_X, val_y_cls, val_y_pos.view(-1)),
        batch_size=cfg.batch_size,
    )

    optim = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.gamma)

    loss_cls = nn.CrossEntropyLoss()
    loss_loc = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss_sum = 0.0
        for xb, y_cls, y_pos in train_loader:
            xb = xb.to(device)
            y_cls = y_cls.to(device)
            y_pos = y_pos.to(device)
            logits, pos_hat = model(xb)
            loss = loss_cls(logits, y_cls) + cfg.lambda_loc * loss_loc(pos_hat, y_pos)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            tr_loss_sum += loss.item() * xb.size(0)

        avg_tr_loss = tr_loss_sum / len(train_loader.dataset)
        val_loss, val_acc, val_rmse = _val_metrics(
            model,
            val_loader,
            loss_cls=loss_cls,
            loss_loc=loss_loc,
            lambda_loc=cfg.lambda_loc,
            device=device,
        )
        print(
            f"E{epoch + 1:02d} | trL={avg_tr_loss:.4f} | valL={val_loss:.4f} | "
            f"Acc={val_acc:.3f} | RMSE={val_rmse:.3f}"  # noqa: T201
        )
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            if cfg.save_path:
                torch.save(model.state_dict(), cfg.save_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.patience:
                print("Early stopping")  # noqa: T201
                break

    if cfg.save_path and Path(cfg.save_path).exists():
        model.load_state_dict(torch.load(cfg.save_path, map_location=device))

    return model


# ---------------------------------------------------------------------------
# Inference                                                                   |
# ---------------------------------------------------------------------------


def predict(
        model: OTDR_TabNet,
        data: torch.Tensor,
        *,
        batch_size: int = 512,
        device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    device = device or next(model.parameters()).device
    model.eval()
    logits_list, pos_list = [], []
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            xb = data[i: i + batch_size].to(device)
            logits, pos_hat = model(xb)
            logits_list.append(logits.cpu())
            pos_list.append(pos_hat.cpu())
    return torch.cat(logits_list, 0), torch.cat(pos_list, 0)
