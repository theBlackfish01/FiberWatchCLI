from __future__ import annotations

"""Temporal Convolutional Network (TCN) multitask classifier for OTDR traces.

Implements the lightweight dilated-TCN you trained in the notebook. It predicts
both **fault class** (categorical) and **fault position** (regression).

This version treats the first feature (SNR) as a **global scalar** and injects it
as a **second channel** by broadcasting it across the 30 positional steps:

    raw row:  [SNR, P1, P2, ... , P30]  ->  (B, 31)
    model in: (B, 2, 30)
        chan 0 = positions (P1..P30)
        chan 1 = SNR repeated 30 times

Public API
----------
* ``OTDR_TCN`` – the model definition.
* ``TrainConfig`` – hyper-parameters for supervised multitask training.
* ``train_tcn`` – full training loop with early-stopping & best-weights save.
* ``predict`` – batched inference that returns *(cls_logits, pos_pred)*.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from sklearn.metrics import root_mean_squared_error
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "OTDR_TCN",
    "TrainConfig",
    "train_tcn",
    "predict",
]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------


class Chomp1d(nn.Module):
    """Remove extra padding on the right produced by dilation."""

    def __init__(self, chomp: int):
        super().__init__()
        self.chomp = chomp

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L+pad)
        return x[:, :, :-self.chomp].contiguous()


class TemporalBlock(nn.Module):
    """Residual dilated causal convolutional block."""

    def __init__(self, in_ch: int, out_ch: int, k: int, d: int, dropout: float = 0.0):
        super().__init__()
        pad = (k - 1) * d
        drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad),
            nn.ReLU(),
            drop1,
            nn.Conv1d(out_ch, out_ch, k, padding=pad, dilation=d),
            Chomp1d(pad),
            nn.ReLU(),
            drop2,
        )
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        return self.act(self.net(x) + self.down(x))


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class AttentionPooling(nn.Module):
    """Additive attention pooling over temporal dimension."""

    def __init__(self, in_ch: int, hidden_ch: int | None = None) -> None:
        super().__init__()
        hidden_ch = hidden_ch or in_ch
        self.score = nn.Sequential(
            nn.Conv1d(in_ch, hidden_ch, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_ch, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, L)
        attn = self.score(x)  # (B, 1, L)
        weights = torch.softmax(attn, dim=-1)
        return torch.sum(x * weights, dim=-1)  # (B, C)


class OTDR_TCN(nn.Module):
    """Dilated TCN multitask model.

    Parameters
    ----------
    in_ch : int
        Number of input channels (= 2 here: positions + broadcast SNR).
    mid_ch : int
        Channel width for hidden layers.
    n_blocks : int
        Depth (each block doubles the receptive field).
    k : int
        Kernel width.
    n_classes : int
        Number of categorical fault classes.
    """

    def __init__(
        self,
        *,
        in_ch: int = 2,  # <-- 2 channels: positions + SNR
        mid_ch: int = 64,
        n_blocks: int = 4,
        k: int = 3,
        n_classes: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        ch = in_ch
        for b in range(n_blocks):
            layers.append(TemporalBlock(ch, mid_ch, k, 2 ** b, dropout=dropout))
            ch = mid_ch
        self.tcn = nn.Sequential(*layers)
        self.attn_pool = AttentionPooling(mid_ch)

        # heads
        self.class_head = nn.Linear(mid_ch, n_classes)
        self.loc_head = nn.Linear(mid_ch, 1)
        self._init_weights()

    # -------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """x is expected to be (B, 2, L) already."""
        h = self.tcn(x)              # (B, mid_ch, L)
        h = self.attn_pool(h)        # (B, mid_ch)
        return self.class_head(h), self.loc_head(h).squeeze(-1)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 25
    lambda_loc: float = 0  # weight of localisation MSE
    step_size: int = 15
    gamma: float = 0.5
    device: torch.device | str | None = None
    save_path: str | Path | None = None


def _to_two_channel(xb: torch.Tensor) -> torch.Tensor:
    """Convert (B, 31) = [snr, p1..p30] to (B, 2, 30)."""
    # xb: (B, 31)
    snr = xb[:, 0]                 # (B,)
    pos = xb[:, 1:]                # (B, 30)
    # broadcast SNR across sequence length
    snr_seq = snr.unsqueeze(1).repeat(1, pos.size(1))  # (B, 30)
    # stack as channels
    x2 = torch.stack([pos, snr_seq], dim=1)            # (B, 2, 30)
    return x2


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
    y_true: list[np.ndarray] = []
    y_pred: list[np.ndarray] = []
    with torch.no_grad():
        for xb, y_cls, y_loc in loader:
            # xb: (B, 31) -> (B, 2, 30)
            xb = _to_two_channel(xb).to(device)
            y_cls = y_cls.to(device)
            y_loc = y_loc.to(device)
            logits, pos_hat = model(xb)
            loss = loss_fn_cls(logits, y_cls) + lambda_loc * loss_fn_loc(pos_hat, y_loc)
            v_loss += loss.item() * xb.size(0)
            v_correct += (logits.argmax(1) == y_cls).sum().item()
            y_true.append(y_loc.detach().cpu().numpy())
            y_pred.append(pos_hat.detach().cpu().numpy())
            v_samples += xb.size(0)

    v_loss /= v_samples
    acc = v_correct / v_samples
    rmse = float("nan")
    if y_true:
        rmse = root_mean_squared_error(np.concatenate(y_true), np.concatenate(y_pred))
    return v_loss, acc, rmse


def train_tcn(
    model: OTDR_TCN,
    train_tensor: torch.Tensor,  # (N, 31) = [snr, p1..p30]
    train_y_cls: torch.Tensor,
    train_y_pos: torch.Tensor,
    val_tensor: torch.Tensor,
    val_y_cls: torch.Tensor,
    val_y_pos: torch.Tensor,
    cfg: TrainConfig | None = None,
):
    """Standard supervised training loop with early-stopping."""

    cfg = cfg or TrainConfig()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device is None
        else torch.device(cfg.device)
    )
    print("[INFO] Using device:", device)
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_y_cls, train_y_pos.view(-1)),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor, val_y_cls, val_y_pos.view(-1)),
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
            # xb: (B, 31) -> (B, 2, 30)
            xb = _to_two_channel(xb).to(device)
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
            f"Acc={val_acc:.3f} | RMSE={val_rmse:.3f}"
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
                print("Early stopping")
                break

    # reload best
    if cfg.save_path and Path(cfg.save_path).exists():
        model.load_state_dict(torch.load(cfg.save_path, map_location=device))

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def predict(
    model: OTDR_TCN,
    data: torch.Tensor,  # (N, 31)
    *,
    batch_size: int = 512,
    device: torch.device | str | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (class_logits, pos_pred) concatenated over input rows."""

    device = device or next(model.parameters()).device
    model.eval()
    logits_list: list[torch.Tensor] = []
    pos_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            xb = data[i : i + batch_size]          # (B, 31)
            xb = _to_two_channel(xb).to(device)     # (B, 2, 30)
            logits, pos_hat = model(xb)
            logits_list.append(logits.cpu())
            pos_list.append(pos_hat.cpu())
    return torch.cat(logits_list, 0), torch.cat(pos_list, 0)
