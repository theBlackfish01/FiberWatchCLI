from __future__ import annotations

"""Temporal Convolutional Network (TCN) multitask classifier for OTDR traces.

Implements the lightweight dilated‑TCN you trained in the notebook.  It predicts
both **fault class** (categorical) and **fault position** (regression).

Public API
----------
* ``OTDR_TCN`` – the model definition.
* ``TrainConfig`` – hyper‑parameters for supervised multitask training.
* ``train_tcn`` – full training loop with early‑stopping & best‑weights save.
* ``predict`` – batched inference that returns *(cls_logits, pos_pred)*.

All helpers mirror the GRU‑AE module’s style for consistency.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "OTDR_TCN",
    "TrainConfig",
    "train_tcn",
    "predict",
]


# ---------------------------------------------------------------------------
# Building blocks                                                             |
# ---------------------------------------------------------------------------


class Chomp1d(nn.Module):
    """Remove extra padding on the right produced by dilation."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L+pad)
        return x[:, :, :-self.chomp].contiguous()


class TemporalBlock(nn.Module):
    """Residual dilated causal convolutional block (à la WaveNet)."""

    def __init__(self, in_ch: int, out_ch: int, k: int, d: int):
        super().__init__()
        pad = (k - 1) * d
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad),
            nn.ReLU(),
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        return self.act(self.net(x) + self.down(x))


# ---------------------------------------------------------------------------
# Model                                                                       |
# ---------------------------------------------------------------------------


class OTDR_TCN(nn.Module):
    """Dilated TCN multitask model.

    Parameters
    ----------
    in_ch : int
        Number of input channels (= 1 for plain feature vector interpreted as length‑31 sequence).
    mid_ch : int
        Channel width for hidden layers.
    n_blocks : int
        Depth (each block doubles the receptive field).
    k : int
        Kernel width.
    n_classes : int
        Number of categorical fault classes (include class 0 = healthy).
    """

    def __init__(
        self,
        *,
        in_ch: int = 1,
        mid_ch: int = 64,
        n_blocks: int = 6,
        k: int = 3,
        n_classes: int = 8,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_ch
        for b in range(n_blocks):
            layers.append(TemporalBlock(ch, mid_ch, k, 2 ** b))
            ch = mid_ch
        self.tcn = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool1d(1)  # (B, C, 1)

        # heads
        self.class_head = nn.Linear(mid_ch, n_classes)
        self.loc_head = nn.Linear(mid_ch, 1)

    # -------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:  # (B, 1, L)
        h = self.tcn(x)
        h = self.gap(h).squeeze(-1)  # (B, mid_ch)
        return self.class_head(h), self.loc_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training                                                                    |
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 10
    lambda_loc: float = 0.5  # weight of localisation MSE
    step_size: int = 15
    gamma: float = 0.5
    device: torch.device | str | None = None
    save_path: str | Path | None = None


def _val_metrics(
    model: OTDR_TCN,
    loader: DataLoader,
    *,
    loss_fn_cls: nn.Module,
    loss_fn_loc: nn.Module,
    lambda_loc: float,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Return (loss, accuracy, RMSE)."""

    model.eval()
    v_loss = 0.0
    v_correct = 0
    v_samples = 0
    mse_sum = 0.0
    with torch.no_grad():
        for xb, y_cls, y_loc in loader:
            xb = xb.unsqueeze(1).to(device)  # (B, 1, L)
            y_cls = y_cls.to(device)
            y_loc = y_loc.to(device)
            logits, pos_hat = model(xb)
            loss = loss_fn_cls(logits, y_cls) + lambda_loc * loss_fn_loc(pos_hat, y_loc)
            v_loss += loss.item() * xb.size(0)
            v_correct += (logits.argmax(1) == y_cls).sum().item()
            mse_sum += loss_fn_loc(pos_hat, y_loc).item() * xb.size(0)
            v_samples += xb.size(0)

    v_loss /= v_samples
    acc = v_correct / v_samples
    rmse = (mse_sum / v_samples) ** 0.5
    return v_loss, acc, rmse


def train_tcn(
    model: OTDR_TCN,
    train_tensor: torch.Tensor,
    train_y_cls: torch.Tensor,
    train_y_pos: torch.Tensor,
    val_tensor: torch.Tensor,
    val_y_cls: torch.Tensor,
    val_y_pos: torch.Tensor,
    cfg: TrainConfig | None = None,
):
    """Standard supervised training loop with early‑stopping."""

    cfg = cfg or TrainConfig()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device is None
        else torch.device(cfg.device)
    )
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_y_cls, train_y_pos.squeeze(-1)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor, val_y_cls, val_y_pos.squeeze(-1)),
        batch_size=cfg.batch_size,
    )

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.gamma)

    loss_cls = nn.CrossEntropyLoss()
    loss_loc = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        # ------------------ train ------------------ #
        model.train()
        train_loss_sum = 0.0
        for xb, y_cls, y_loc in train_loader:
            xb = xb.unsqueeze(1).to(device)
            y_cls = y_cls.to(device)
            y_loc = y_loc.to(device)
            logits, pos_hat = model(xb)
            loss = loss_cls(logits, y_cls) + cfg.lambda_loc * loss_loc(pos_hat, y_loc)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss_sum += loss.item() * xb.size(0)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)

        # ------------------ val -------------------- #
        val_loss, val_acc, val_rmse = _val_metrics(
            model,
            val_loader,
            loss_fn_cls=loss_cls,
            loss_fn_loc=loss_loc,
            lambda_loc=cfg.lambda_loc,
            device=device,
        )

        print(
            f"E{epoch+1:02d} | trainL={avg_train_loss:.4f} | valL={val_loss:.4f} | "
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

    # reload best
    if cfg.save_path and Path(cfg.save_path).exists():
        model.load_state_dict(torch.load(cfg.save_path, map_location=device))

    return model


# ---------------------------------------------------------------------------
# Inference                                                                   |
# ---------------------------------------------------------------------------


def predict(
    model: OTDR_TCN,
    data: torch.Tensor,
    *,
    batch_size: int = 512,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (class_logits, pos_pred) concatenated over input rows."""

    device = device or next(model.parameters()).device
    model.eval()
    logits_list, pos_list = [], []
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            xb = data[i : i + batch_size].unsqueeze(1).to(device)
            logits, pos_hat = model(xb)
            logits_list.append(logits.cpu())
            pos_list.append(pos_hat.cpu())
    return torch.cat(logits_list, 0), torch.cat(pos_list, 0)
