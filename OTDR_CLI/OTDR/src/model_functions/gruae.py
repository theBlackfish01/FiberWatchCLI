from __future__ import annotations

"""GRU-based Auto-Encoder (GRU-AE) for anomaly detection.
----------
* ``VectorGRUAE`` – model class
* ``TrainConfig`` – training hyperparams
* ``train_gru_ae`` – trains on normal samples and returns (best_model, threshold)
* ``reconstruction_error`` – batched MSE per sample
* ``determine_threshold`` – quantile-based threshold
* ``detect`` – flags anomalies
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
# Model
# ---------------------------------------------------------------------------


class VectorGRUAE(nn.Module):
    """GRU auto-encoder for *vector* inputs (shape (B, 1, feat_dim)).

    Encoder: GRU → take last hidden → linear → latent
    Decoder: latent → linear → initial hidden for GRU → GRU fed with zeros →
             linear → reconstructed vector
    """

    def __init__(
        self,
        feat_dim: int,
        *,
        hidden: int = 128,
        latent: int = 64,
        layers: int = 1,
        bidir: bool = True,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.feat_dim = feat_dim
        self.hidden = hidden
        self.latent = latent
        self.layers = layers
        self.bidir = bidir
        self.dropout_p = float(dropout)

        # ---------------- encoder ---------------- #
        gru_dropout = dropout if layers > 1 else 0.0
        self.encoder = nn.GRU(
            input_size=feat_dim,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidir,
            dropout=gru_dropout,
        )
        dir_mult = 2 if bidir else 1
        enc_dim = hidden * dir_mult
        self.fc_mu = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            nn.LayerNorm(enc_dim),
            nn.Linear(enc_dim, latent),
        )
        self.latent_norm = nn.LayerNorm(latent)
        self.latent_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # ---------------- decoder ---------------- #
        # map latent to initial hidden state of decoder GRU
        self.fc_init = nn.Sequential(
            nn.Linear(latent, latent),
            nn.GELU(),
            nn.Linear(latent, hidden * layers),
        )
        self.decoder = nn.GRU(
            input_size=feat_dim,  # we'll feed zeros of this size
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.decoder_dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, feat_dim),
        )
        self.out_norm = nn.LayerNorm(feat_dim)

        self._init_weights()

    # --------------------------------------------------------------------- #
    # Forward                                                               #
    # --------------------------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, 1, feat_dim).
        """
        assert x.dim() == 3, "Input must be (batch, seq_len=1, feat_dim)"
        B = x.size(0)

        # ----- encode ----- #
        _, h_enc = self.encoder(x)  # h_enc: (layers * dir_mult, B, hidden)
        if self.bidir:
            # concatenate last layer's forward and backward states
            h_fwd = h_enc[-2]  # (B, hidden)
            h_bwd = h_enc[-1]  # (B, hidden)
            h_cat = torch.cat([h_fwd, h_bwd], dim=1)  # (B, hidden*2)
        else:
            h_cat = h_enc[-1]  # (B, hidden)

        z = self.fc_mu(h_cat)  # (B, latent)
        z = self.latent_norm(z)
        z = self.latent_dropout(z)

        # ----- prepare decoder init ----- #
        h0 = self.fc_init(z).view(self.layers, B, self.hidden).contiguous()  # (layers, B, hidden)

        # ----- decode from zeros, not from x ----- #
        # force model to use latent
        decoder_input = x.new_zeros(B, 1, self.feat_dim)  # (B, 1, feat_dim)
        dec_out, _ = self.decoder(decoder_input, h0)  # (B, 1, hidden)
        dec_out = self.decoder_dropout(dec_out)
        recon = self.out(dec_out)  # (B, 1, feat_dim)
        recon = self.out_norm(recon)
        return recon

    # ------------------------------------------------------------------ #
    # Weight initialisation                                              #
    # ------------------------------------------------------------------ #

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
            elif name.endswith("weight"):
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.ones_(param)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    epochs: int = 180
    batch_size: int = 256
    lr: float = 1e-3
    patience: int = 20
    lr_patience: int = 5
    lr_factor: float = 0.5
    min_lr: float = 1e-5
    quantile: float = 0.95
    device: torch.device | str | None = None
    save_path: str | Path | None = None
    weight_decay: float = 1e-5
    grad_clip: float = 1.0


@torch.no_grad()
def _quick_val_loss(
    model: "VectorGRUAE",
    val_tensor: torch.Tensor,
    loss_fn: nn.Module,
    device: torch.device,
    batch_size: int,
) -> float:
    total = 0.0
    count = 0
    for i in range(0, val_tensor.size(0), batch_size):
        xb = val_tensor[i : i + batch_size].unsqueeze(1).to(device)  # (B, 1, feat_dim)
        recon = model(xb)
        loss = loss_fn(recon, xb)
        total += loss.item() * xb.size(0)
        count += xb.size(0)
    return total / count if count else float("nan")


def train_gru_ae(
    model: "VectorGRUAE",
    train_tensor: torch.Tensor,
    val_tensor: torch.Tensor,
    cfg: TrainConfig | None = None,
) -> Tuple["VectorGRUAE", float]:
    """Train AE on *normal* data and return (best_model, threshold)."""
    cfg = cfg or TrainConfig()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if cfg.device is None
        else torch.device(cfg.device)
    )
    model = model.to(device)

    loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
    )
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optim,
        mode="min",
        patience=cfg.lr_patience,
        factor=cfg.lr_factor,
        min_lr=cfg.min_lr,
    )
    loss_fn = nn.MSELoss()

    best_val = float("inf")
    epochs_no_improve = 0
    best_state_dict: dict[str, torch.Tensor] | None = None

    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss = 0.0
        for (xb,) in loader:
            xb = xb.unsqueeze(1).to(device)  # (B, 1, feat_dim)
            recon = model(xb)
            loss = loss_fn(recon, xb)
            optim.zero_grad()
            loss.backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.grad_clip)
            optim.step()
            epoch_loss += loss.item() * xb.size(0)

        avg_train_loss = epoch_loss / len(loader.dataset)
        val_loss = _quick_val_loss(model, val_tensor, loss_fn, device, cfg.batch_size)
        current_lr = optim.param_groups[0]["lr"]
        print(
            f"E{epoch:02d}  trainMSE={avg_train_loss:.5f}  valMSE={val_loss:.5f}  lr={current_lr:.2e}"  # noqa: T201
        )
        scheduler.step(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            epochs_no_improve = 0
            best_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
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
    elif best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    # determine threshold on training errors
    errs = reconstruction_error(model, train_tensor, batch_size=cfg.batch_size, device=device)
    thresh = determine_threshold(errs, cfg.quantile)
    return model, thresh


# ---------------------------------------------------------------------------
# Inference & helpers
# ---------------------------------------------------------------------------


def reconstruction_error(
    model: "VectorGRUAE",
    data: torch.Tensor,
    *,
    batch_size: int = 512,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return per-sample MSE reconstruction error."""
    model.eval()
    device = device or next(model.parameters()).device
    errs = []
    with torch.no_grad():
        for i in range(0, data.size(0), batch_size):
            xb = data[i : i + batch_size].unsqueeze(1).to(device)  # (B, 1, feat_dim)
            recon = model(xb)
            mse = (recon - xb).pow(2).mean(dim=(1, 2))  # mean over seq and features
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
    """Return boolean mask (is_anomaly) of shape (N,)."""
    errs = reconstruction_error(model, data, batch_size=batch_size, device=device)
    return errs > threshold
