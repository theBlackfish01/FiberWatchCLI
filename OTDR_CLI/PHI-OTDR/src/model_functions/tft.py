from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext


# ---- AMP compatibility (torch.amp vs torch.cuda.amp) ----
try:
    # New API (PyTorch ≥ 2.0)
    from torch.amp import autocast as _autocast_new, GradScaler as _GradScalerNew

    def _autocast(device: torch.device):
        return _autocast_new('cuda') if device.type == 'cuda' else nullcontext()

    def _make_scaler(device: torch.device):
        return _GradScalerNew('cuda') if device.type == 'cuda' else None

except Exception:
    # Old API (PyTorch ≤ 1.x)
    from torch.cuda.amp import autocast as _autocast_old, GradScaler as _GradScalerOld

    def _autocast(device: torch.device):
        return _autocast_old(enabled=(device.type == 'cuda'))

    def _make_scaler(device: torch.device):
        return _GradScalerOld(enabled=(device.type == 'cuda'))


# ------------------------------ utils ------------------------------ #

def _to_btc(x: torch.Tensor, in_channels: int) -> torch.Tensor:
    """
    Convert input to (B, T, C).
    Accepts (B,1,T,C) or (B,T,C) or (B,C,T).
    """
    if x.dim() == 4:          # (B,1,T,C)
        x = x.squeeze(1)
    if x.dim() != 3:
        raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")
    # (B, ?, ?) -> ensure last dim is C
    if x.shape[-1] == in_channels:
        return x
    elif x.shape[1] == in_channels:
        return x.transpose(1, 2)  # (B,C,T) -> (B,T,C)
    else:
        raise ValueError(f"Cannot infer (T,C) from shape {tuple(x.shape)} with in_channels={in_channels}")


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard Transformer sinusoidal position encodings.
    """
    def __init__(self, d_model: int, max_len: int = 20000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)  # (max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        T = x.size(1)
        return x + self.pe[:T, :]


# --------------------------- TFT components -------------------------- #

class VariableSelection(nn.Module):
    """
    Lightweight variable-selection module:
      - Computes per-timestep softmax weights over C channels.
      - Each channel has its own small embedding (1 -> d_model).
      - Fuses to a single (B, T, d_model) representation via weighted sum.

    Input:  x in (B, T, C)
    Output: fused in (B, T, d_model), weights in (B, T, C)
    """
    def __init__(self, in_channels: int, d_model: int):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.var_projs = nn.ModuleList([nn.Linear(1, d_model) for _ in range(in_channels)])
        self.weight_net = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        if C != self.in_channels:
            raise ValueError(f"VariableSelection expected C={self.in_channels}, got {C}")

        # (B,T,C) -> (B,T,C) weights
        w = self.weight_net(x)  # softmax across features at each timestep

        # Per-variable embeddings, then fuse with weights
        fused = 0.0
        for c in range(C):
            xc = x[..., c:c+1]                       # (B,T,1)
            ec = self.var_projs[c](xc)               # (B,T,d_model)
            wc = w[..., c:c+1]                       # (B,T,1)
            fused = fused + ec * wc

        return fused, w  # (B,T,d_model), (B,T,C)


class AttentionPooling(nn.Module):
    """
    Single-query attention pooling over time for sequence classification.
    Returns (B, d_model) pooled representation.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.query = nn.Parameter(torch.randn(d_model))
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, d_model)
        """
        q = self.query  # (d_model,)
        k = torch.tanh(self.key(x))           # (B,T,d_model)
        v = self.value(x)                     # (B,T,d_model)
        scores = torch.einsum("btd,d->bt", k, q)  # (B,T)
        attn = torch.softmax(scores, dim=1)        # (B,T)
        pooled = torch.einsum("bt,btd->bd", attn, v)
        return pooled


# ------------------------------ Model -------------------------------- #

class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer (classification variant).

    Pipeline:
      1) Variable selection & per-feature embedding (timestep-wise).
      2) Add sinusoidal positional encoding.
      3) Transformer-Encoder stack over time.
      4) Attention pooling over time.
      5) Linear classifier.

    Expected input:
      - (B, 1, T, C)  OR (B, T, C) OR (B, C, T)
    """
    def __init__(
        self,
        in_channels: int,
        n_classes: int,
        d_model: int = 96,        # lighter than 128
        n_heads: int = 3,         # d_model % n_heads == 0
        num_layers: int = 2,      # fewer layers
        d_ff: int = 192,
        dropout: float = 0.1,
        max_len: int = 20000,
        max_tokens: int = 1024,   # cap seq length seen by attention
    ):
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        self.max_tokens = max_tokens

        self.var_sel = VariableSelection(in_channels, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        self.pool = AttentionPooling(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Convert to (B,T,C)
        x = _to_btc(x, self.in_channels)  # (B,T,C)

        # --- downsample long sequences along time (avg pool) ---
        B, T, C = x.shape
        if T > self.max_tokens:
            k = math.ceil(T / self.max_tokens)  # pooling stride/kernel
            x = F.avg_pool1d(x.transpose(1, 2), kernel_size=k, stride=k, ceil_mode=True).transpose(1, 2)
            T = x.size(1)
        # -------------------------------------------------------

        # 1) variable selection
        z, _weights = self.var_sel(x)
        # 2) positional encodings
        z = self.pos_enc(z)
        # 3) transformer encoder
        z = self.encoder(z)
        # 4) attention pooling
        feats = self.pool(z)
        # 5) classifier
        logits = self.classifier(feats)
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
    in_channels: int = 12
    d_model: int = 96
    n_heads: int = 3
    num_layers: int = 2
    d_ff: int = 192
    dropout: float = 0.1
    max_tokens: int = 1024


def train_tft(model: TemporalFusionTransformer, train_loader, val_loader, cfg: TrainConfig) -> TemporalFusionTransformer:
    model = model.to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scaler = _make_scaler(cfg.device)

    # propagate max_tokens to model if provided via cfg
    if hasattr(cfg, "max_tokens") and hasattr(model, "max_tokens"):
        model.max_tokens = cfg.max_tokens

    best_acc = -1.0
    for epoch in range(1, cfg.epochs + 1):
        # ---------------------- Train ---------------------- #
        model.train()
        tr_correct = tr_total = 0
        tr_loss = 0.0
        tr_batches = 0
        for batch in train_loader:
            if batch is None:
                continue
            x = batch["data"].unsqueeze(1).to(cfg.device, dtype=torch.float32)
            y = batch["label"].to(cfg.device, dtype=torch.long)

            opt.zero_grad(set_to_none=True)
            with _autocast(cfg.device):
                _, logits = model(x)
                loss = criterion(logits, y)

            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                opt.step()

            tr_loss += float(loss.detach())
            tr_correct += (logits.argmax(1) == y).sum().item()
            tr_total += y.numel()
            tr_batches += 1

        # ----------------------- Val ----------------------- #
        model.eval()
        va_correct = va_total = 0
        va_loss = 0.0
        va_batches = 0
        with torch.no_grad():
            for batch in val_loader:
                if batch is None:
                    continue
                x = batch["data"].unsqueeze(1).to(cfg.device, dtype=torch.float32)
                y = batch["label"].to(cfg.device, dtype=torch.long)
                with _autocast(cfg.device):
                    _, logits = model(x)
                    loss = criterion(logits, y)
                va_loss += float(loss)
                va_correct += (logits.argmax(1) == y).sum().item()
                va_total += y.numel()
                va_batches += 1

        tr_acc = tr_correct / max(tr_total, 1)
        va_acc = va_correct / max(va_total, 1)
        print(f"[TFT] Epoch {epoch:03d}  train_acc={tr_acc:.3f}  "
              f"train_loss={tr_loss/max(tr_batches,1):.4f}  "
              f"val_acc={va_acc:.3f}  val_loss={va_loss/max(va_batches,1):.4f}")

        if va_acc > best_acc and va_batches > 0:
            best_acc = va_acc
            torch.save(model.state_dict(), cfg.save_path)

    print(f"[TFT] Best val acc={best_acc:.4f} (saved to {cfg.save_path})")
    return model
