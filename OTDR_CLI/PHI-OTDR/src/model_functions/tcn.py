from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import torch
import torch.nn as nn


# ------------------------- TCN building blocks ------------------------- #

class Chomp1d(nn.Module):
    """Remove the extra right padding introduced by causal padding."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size] if self.chomp_size > 0 else x


class TemporalBlock(nn.Module):
    """Causal dilated residual block: Conv1d -> ReLU -> Dropout (x2) + residual."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # left padding for causal conv
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU(inplace=True)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=pad,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU(inplace=True)
        self.drop2 = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.final_relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x)
        y = self.chomp1(y)
        y = self.relu1(y)
        y = self.drop1(y)

        y = self.conv2(y)
        y = self.chomp2(y)
        y = self.relu2(y)
        y = self.drop2(y)

        res = self.downsample(x)
        return self.final_relu(y + res)


class TemporalConvNet(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilations."""
    def __init__(
        self,
        in_channels: int,
        channels: Sequence[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_c = in_channels
        for i, c in enumerate(channels):
            dilation = 2 ** i
            layers.append(
                TemporalBlock(
                    prev_c, c, kernel_size=kernel_size, dilation=dilation, dropout=dropout
                )
            )
            prev_c = c
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)  # (B, channels[-1], T)


# ------------------------------- Model -------------------------------- #

class TCN(nn.Module):
    """
    Temporal Convolutional Network for Φ-OTDR classification.

    Expected input:
      - (B, 1, T, C)  from your dataloader + unsqueeze(1)  (like CNN path), OR
      - (B, T, C)     raw batch from dataset, OR
      - (B, C, T)     already prepared for Conv1d

    We internally convert to (B, C, T) for Conv1d.
    """
    def __init__(
        self,
        in_channels: int,         # number of DAS channels (e.g., 12)
        n_classes: int = 6,
        channels: Sequence[int] | None = None,
        kernel_size: int = 3,
        dropout: float = 0.2,
        pool: str = "avg",        # "avg" or "max"
    ):
        super().__init__()
        if channels is None:
            channels = (64, 64, 128, 128)

        self.in_channels = in_channels
        self.tcn = TemporalConvNet(in_channels, channels, kernel_size=kernel_size, dropout=dropout)
        self.pool = nn.AdaptiveAvgPool1d(1) if pool == "avg" else nn.AdaptiveMaxPool1d(1)
        self.classifier = nn.Linear(channels[-1], n_classes)

    @staticmethod
    def _to_bct(x: torch.Tensor, in_channels: int) -> torch.Tensor:
        """Convert input into (B, C, T) for Conv1d."""
        if x.dim() == 4:        # (B,1,T,C) -> (B,C,T)
            x = x.squeeze(1).permute(0, 2, 1)
        elif x.dim() == 3:
            # If second dim != channels, assume (B,T,C) and permute
            if x.shape[1] != in_channels and x.shape[2] == in_channels:
                x = x.permute(0, 2, 1)
        else:
            raise ValueError(f"Unexpected input shape for TCN: {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._to_bct(x, self.in_channels)       # (B, C, T)
        y = self.tcn(x)                              # (B, F, T)
        feats = self.pool(y).squeeze(-1)             # (B, F)
        logits = self.classifier(feats)              # (B, n_classes)
        return feats, logits


# --------------------------- Inference helper -------------------------- #

def predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Return logits for input x (any accepted shape)."""
    model.eval()
    with torch.no_grad():
        _, logits = model(x)
    return logits


# ------------------------------ Training ------------------------------- #

@dataclass
class TrainConfig:
    save_path: Path
    device: torch.device
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-5
    kernel_size: int = 3
    dropout: float = 0.2
    channels: Iterable[int] = (64, 64, 128, 128)
    in_channels: int = 12          # Φ-OTDR default (update if different)


def train_tcn(model: TCN, train_loader, val_loader, cfg: TrainConfig) -> TCN:
    """
    Training loop mirrors your CNN trainer:
      - batches can be None (skipped safely)
      - inputs are taken as dataset tensors (B, T, C) but we call unsqueeze(1)
        so the same dataloader works for both CNN and TCN paths.
    """
    model = model.to(cfg.device)
    print(f"[TCN] Training on {cfg.device} with {cfg.epochs} epochs")
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        # ---------------------- Train ---------------------- #
        model.train()
        tr_correct, tr_total, tr_loss, tr_batches = 0, 0, 0.0, 0
        for batch in train_loader:
            if batch is None:
                continue
            x = batch["data"].unsqueeze(1).to(cfg.device, dtype=torch.float32)  # (B,1,T,C)
            y = batch["label"].to(cfg.device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            _, logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            opt.step()

            tr_loss += loss.item()
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.numel()
            tr_batches += 1

        # ----------------------- Val ----------------------- #
        model.eval()
        va_correct, va_total, va_loss, va_batches = 0, 0, 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x = batch["data"].unsqueeze(1).to(cfg.device, dtype=torch.float32)
                y = batch["label"].to(cfg.device, dtype=torch.long)
                _, logits = model(x)
                loss = criterion(logits, y)
                va_loss += loss.item()
                va_correct += (logits.argmax(1) == y).sum().item()
                va_total += y.numel()
                va_batches += 1

        tr_acc = tr_correct / max(tr_total, 1)
        va_acc = va_correct / max(va_total, 1)
        print(f"[TCN] Epoch {epoch:03d}  train_acc={tr_acc:.3f}  "
              f"train_loss={tr_loss/max(tr_batches,1):.4f}  "
              f"val_acc={va_acc:.3f}  val_loss={va_loss/max(va_batches,1):.4f}")

        if va_acc > best_acc and va_batches > 0:
            best_acc = va_acc
            torch.save(model.state_dict(), cfg.save_path)

    print(f"[TCN] Best val acc={best_acc:.4f} (saved to {cfg.save_path})")
    return model
