from __future__ import annotations

"""GRU-based Auto-Encoder (GRU-AE) for anomaly detection.

This module exposes:
    * ``VectorGRUAE`` – the model class (encoder‑decoder GRU over *vector* inputs).
    * ``train_gru_ae`` – util that trains on **normal** samples only and performs
      early‑stopping, returning the _best_ model and reconstruction‑error
      threshold (quantile over training errors).
    * ``reconstruction_error`` – batched inference utility that yields mean‑
      squared error per sample.
    * ``detect`` – convenience wrapper that flags anomalies above the threshold.

The encoder is bidirectional by default. When bidirectional, we concatenate the
last hidden states from the forward **and** backward directions before feeding
into the latent bottleneck.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "VectorGRUAE",
    "TrainConfig",
    "train_gru_ae",
    "reconstruction_error",
    "determine_threshold",
    "detect",
]


# ---------------------------------------------------------------------------
# Model                                                                       |
# ---------------------------------------------------------------------------


class VectorGRUAE(nn.Module):
    """Symmetric GRU auto‑encoder over *vector* inputs."""

    def __init__(
        self,
        feat_dim: int,
        *,
        hidden: int = 256,
        latent: int = 128,
        layers: int = 5,
        bidir: bool = True,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.latent = latent
        self.layers = layers
        self.bidir = bidir

        # ---------------- encoder ---------------- #
        self.encoder = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
        )
        dir_mult = 2 if bidir else 1
        self.fc_mu = nn.Linear(hidden * dir_mult, latent)

        # ---------------- decoder ---------------- #
        self.fc_init = nn.Linear(latent, hidden * layers)
        self.decoder = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
        )
        self.out = nn.Linear(hidden, feat_dim)

    # --------------------------------------------------------------------- #
    # Forward                                                               #
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(B, 1, feat_dim)``.
        """
        assert x.dim() == 3, "Input must be (batch, seq_len, feat_dim)"

        _, h_enc = self.encoder(x)  # (layers*dir_mult, B, hidden)

        if self.bidir:
            # Concatenate last layer's forward and backward hidden states
            h_forward = h_enc[-2]  # (B, hidden)
            h_backward = h_enc[-1]  # (B, hidden)
            h_cat = torch.cat([h_forward, h_backward], dim=1)  # (B, hidden*2)
        else:
            h_cat = h_enc[-1]  # (B, hidden)

        z = self.fc_mu(h_cat)  # (B, latent)
        h0 = (
            self.fc_init(z)
            .view(self.layers, x.size(0), self.hidden)
            .contiguous()
        )  # (layers, B, hidden)
        dec_out, _ = self.decoder(x, h0)
        return self.out(dec_out)  # (B, 1, feat_dim)


# ---------------------------------------------------------------------------
# Training                                                                    |
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 120
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 10
    step_size: int = 10
    gamma: float = 0.5
    quantile: float = 0.95
    device: torch.device | str | None = None
    save_path: str | Path | None = None


@torch.no_grad()
def _quick_val_loss(model: "VectorGRUAE", val_tensor: torch.Tensor, loss_fn: nn.Module) -> float:
    sample = val_tensor[:5000].unsqueeze(1)
    return loss_fn(model(sample), sample).item()


def train_gru_ae(
    model: "VectorGRUAE",
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
    cfg: TrainConfig | None = None,
) -> Tuple["VectorGRUAE", float]:
    cfg = cfg or TrainConfig()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device is None
        else torch.device(cfg.device)
    )
    model = model.to(device)

    loader = DataLoader(TensorDataset(train_tensor), batch_size=cfg.batch_size, shuffle=True, drop_last=True)
    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=cfg.step_size, gamma=cfg.gamma)
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.unsqueeze(1).to(device)
            loss = loss_fn(model(xb), xb)
            optim.zero_grad()
            loss.backward()
            optim.step()
            epoch_loss += loss.item() * xb.size(0)

        avg_train_loss = epoch_loss / len(loader.dataset)
        val_loss = _quick_val_loss(model, val_tensor.to(device), loss_fn)
        print(f"E{epoch:02d}  trainMSE={avg_train_loss:.5f}  valMSE={val_loss:.5f}")  # noqa: T201
        scheduler.step()

        if val_loss < best_val:
            best_val = val_loss
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

    errs = reconstruction_error(model, train_tensor, batch_size=cfg.batch_size, device=device)
    thresh = determine_threshold(errs, cfg.quantile)
    return model, thresh


# ---------------------------------------------------------------------------
# Inference & helpers                                                         |
# ---------------------------------------------------------------------------


def reconstruction_error(
    model: "VectorGRUAE",
    data: torch.Tensor,
    *,
    batch_size: int = 512,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    model.eval()
    device = device or next(model.parameters()).device
    errs = []
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            xb = data[i : i + batch_size].unsqueeze(1).to(device)
            mse = (model(xb) - xb).pow(2).mean(dim=(1, 2))
            errs.append(mse.cpu())
    return torch.cat(errs, 0)


def determine_threshold(errs: torch.Tensor, quantile: float = 0.95) -> float:
    return torch.quantile(errs, quantile).item()


def detect(
    model: "VectorGRUAE",
    data: torch.Tensor,
    *,
    threshold: float,
    batch_size: int = 512,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    errs = reconstruction_error(model, data, batch_size=batch_size, device=device)
    return errs > threshold
