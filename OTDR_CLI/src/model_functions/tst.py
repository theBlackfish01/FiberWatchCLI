from __future__ import annotations

"""Time‑Series Transformer (TST) multitask model for OTDR traces.

Public API
----------
* ``TimeSeriesTransformer`` – model definition.
* ``TrainConfig`` – hyper‑parameters for joint class + position training.
* ``train_tst`` – early‑stopping training loop that mirrors the GRU‑AE/TCN helpers.
* ``predict`` – batched inference returning ``(class_logits, pos_pred)``.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "TimeSeriesTransformer",
    "TrainConfig",
    "train_tst",
    "predict",
]


# ---------------------------------------------------------------------------
# Model                                                                       |
# ---------------------------------------------------------------------------


class TimeSeriesTransformer(nn.Module):
    """Transformer encoder with learnable positional encodings + global pooling."""

    def __init__(
        self,
        *,
        seq_len: int = 31,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        n_classes: int = 8,
    ) -> None:
        super().__init__()
        self.seq_len = seq_len

        # project scalar amplitude → d_model
        self.input_proj = nn.Linear(1, d_model)

        # learnable [L, d_model] positional embeddings
        self.pos_embed = nn.Parameter(torch.randn(seq_len, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # multitask heads (classification & localisation)
        self.cls_head = nn.Linear(d_model, n_classes)
        self.loc_head = nn.Linear(d_model, 1)

        self._init_weights()

    # -------------------------------------------- #
    # Weight init                                  #
    # -------------------------------------------- #

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # -------------------------------------------- #

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Input ``x`` can be (B, L) or (B, 1, L). Returns (logits, pos_pred)."""
        if x.dim() == 3:  # (B, 1, L)
            x = x.squeeze(1)
        assert x.dim() == 2 and x.size(1) == self.seq_len, "Unexpected input shape"

        B, L = x.shape
        x = x.unsqueeze(-1)  # (B, L, 1)
        h = self.input_proj(x)  # (B, L, d_model)
        h = h + self.pos_embed[None, :, :]  # broadcast add positional encodings

        h = self.encoder(h)  # (B, L, d_model)
        h = h.mean(dim=1)  # (B, d_model) – global pooling

        return self.cls_head(h), self.loc_head(h).squeeze(-1)


# ---------------------------------------------------------------------------
# Training                                                                    |
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 2e-4
    weight_decay: float = 1e-2
    step_size: int = 15
    gamma: float = 0.5
    patience: int = 10
    lambda_loc: float = 0.5
    device: torch.device | str | None = None
    save_path: str | Path | None = None


def _val_metrics(
    model: TimeSeriesTransformer,
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
        for xb, y_cls, y_loc in loader:
            xb = xb.to(device)
            y_cls = y_cls.to(device)
            y_loc = y_loc.to(device)
            logits, pos_hat = model(xb)
            loss = loss_cls(logits, y_cls) + lambda_loc * loss_loc(pos_hat, y_loc)
            v_loss += loss.item() * xb.size(0)
            v_correct += (logits.argmax(1) == y_cls).sum().item()
            mse_sum += loss_loc(pos_hat, y_loc).item() * xb.size(0)
            v_samples += xb.size(0)

    v_loss /= v_samples
    acc = v_correct / v_samples
    rmse = (mse_sum / v_samples) ** 0.5
    return v_loss, acc, rmse


def train_tst(
    model: TimeSeriesTransformer,
    train_tensor: torch.Tensor,
    train_y_cls: torch.Tensor,
    train_y_pos: torch.Tensor,
    val_tensor: torch.Tensor,
    val_y_cls: torch.Tensor,
    val_y_pos: torch.Tensor,
    cfg: TrainConfig | None = None,
) -> TimeSeriesTransformer:
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
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor, val_y_cls, val_y_pos.squeeze(-1)),
        batch_size=cfg.batch_size,
    )

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.gamma)

    loss_cls = nn.CrossEntropyLoss()
    loss_loc = nn.MSELoss()

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        # ----------- training ------------ #
        model.train()
        train_loss_sum = 0.0
        for xb, y_cls, y_loc in train_loader:
            xb = xb.to(device)
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

        # ----------- validation ---------- #
        val_loss, val_acc, val_rmse = _val_metrics(
            model,
            val_loader,
            loss_cls=loss_cls,
            loss_loc=loss_loc,
            lambda_loc=cfg.lambda_loc,
            device=device,
        )
        print(
            f"E{epoch+1:02d} | trL={avg_train_loss:.4f} | valL={val_loss:.4f} | "
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
    model: TimeSeriesTransformer,
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
            xb = data[i : i + batch_size].to(device)
            logits, pos_hat = model(xb)
            logits_list.append(logits.cpu())
            pos_list.append(pos_hat.cpu())
    return torch.cat(logits_list, 0), torch.cat(pos_list, 0)
