from __future__ import annotations

"""Binary Temporal Convolutional Network (TCN) classifier for OTDR traces."""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .tcn import AttentionPooling, TemporalBlock, _to_two_channel

__all__ = [
    "OTDR_TCNBinary",
    "TrainConfig",
    "train_tcn_binary",
    "predict",
]


class OTDR_TCNBinary(nn.Module):
    """Dilated TCN for binary (normal vs anomaly) classification."""

    def __init__(
        self,
        *,
        in_ch: int = 2,
        mid_ch: int = 64,
        n_blocks: int = 4,
        k: int = 3,
        n_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if n_classes != 2:
            raise ValueError("Binary TCN expects exactly two classes (normal vs anomaly).")
        self.n_classes = n_classes

        layers: list[nn.Module] = []
        ch = in_ch
        for b in range(n_blocks):
            layers.append(TemporalBlock(ch, mid_ch, k, 2 ** b, dropout=dropout))
            ch = mid_ch
        self.tcn = nn.Sequential(*layers)
        self.attn_pool = AttentionPooling(mid_ch)
        self.class_head = nn.Linear(mid_ch, n_classes)
        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.tcn(x)
        h = self.attn_pool(h)
        return self.class_head(h)

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


@dataclass
class TrainConfig:
    epochs: int = 150
    batch_size: int = 128
    lr: float = 1e-3
    patience: int = 25
    step_size: int = 15
    gamma: float = 0.5
    device: torch.device | str | None = None
    save_path: str | Path | None = None
    pos_count: int = 30


def _compute_class_weights(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.to(dtype=torch.long), minlength=n_classes).float()
    if counts.min() <= 0:
        return torch.ones(n_classes)
    total = counts.sum()
    return total / (n_classes * counts)


def _val_metrics(
    model: OTDR_TCNBinary,
    loader: DataLoader,
    *,
    loss_fn: nn.Module,
    device: torch.device,
    pos_count: int,
) -> Tuple[float, float]:
    model.eval()
    v_loss = 0.0
    v_correct = 0
    v_samples = 0
    with torch.no_grad():
        for xb, y_cls in loader:
            xb = _to_two_channel(xb, pos_count=pos_count).to(device)
            y_cls = y_cls.to(device)
            logits = model(xb)
            loss = loss_fn(logits, y_cls)
            v_loss += loss.item() * xb.size(0)
            v_correct += (logits.argmax(1) == y_cls).sum().item()
            v_samples += xb.size(0)
    v_loss /= max(v_samples, 1)
    acc = v_correct / max(v_samples, 1)
    return v_loss, acc


def train_tcn_binary(
    model: OTDR_TCNBinary,
    train_tensor: torch.Tensor,
    train_y_cls: torch.Tensor,
    val_tensor: torch.Tensor,
    val_y_cls: torch.Tensor,
    cfg: TrainConfig | None = None,
) -> OTDR_TCNBinary:
    cfg = cfg or TrainConfig()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device is None
        else torch.device(cfg.device)
    )
    print("[INFO] Using device:", device)
    model = model.to(device)

    train_loader = DataLoader(
        TensorDataset(train_tensor, train_y_cls),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_tensor, val_y_cls),
        batch_size=cfg.batch_size,
    )

    class_weights = _compute_class_weights(train_y_cls, model.n_classes).to(device)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.gamma)

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        train_loss_sum = 0.0
        for xb, y_cls in train_loader:
            xb = _to_two_channel(xb, pos_count=cfg.pos_count).to(device)
            y_cls = y_cls.to(device)
            logits = model(xb)
            loss = loss_fn(logits, y_cls)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            train_loss_sum += loss.item() * xb.size(0)

        avg_train_loss = train_loss_sum / len(train_loader.dataset)
        val_loss, val_acc = _val_metrics(
            model,
            val_loader,
            loss_fn=loss_fn,
            device=device,
            pos_count=cfg.pos_count,
        )

        print(
            f"E{epoch + 1:02d} | trainL={avg_train_loss:.4f} | valL={val_loss:.4f} | Acc={val_acc:.3f}"
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

    if cfg.save_path and Path(cfg.save_path).exists():
        model.load_state_dict(torch.load(cfg.save_path, map_location=device))

    return model


def predict(
    model: OTDR_TCNBinary,
    data: torch.Tensor,
    *,
    batch_size: int = 512,
    device: torch.device | str | None = None,
    pos_count: int = 30,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    logits_list: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            xb = data[i : i + batch_size]
            xb = _to_two_channel(xb, pos_count=pos_count).to(device)
            logits = model(xb)
            logits_list.append(logits.cpu())
    return torch.cat(logits_list, 0)
