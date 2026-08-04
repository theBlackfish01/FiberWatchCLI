from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from .model_functions.multi_similarity_siamese import MultiSimilaritySiamese
from .model_functions.tcn import _to_two_channel
from .model_functions.zero_shot import require_cuda
from .one_shot_data import build_balanced_pair_indices
from .zero_shot_training import seed_everything


@dataclass(frozen=True)
class OneShotTrainingConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 7
    pair_count: int = 16384
    validation_pair_count: int = 4096
    embedding_dim: int = 128
    dropout: float = 0.1
    similarity_mode: str = "multi"
    calibration_epochs: int = 4
    calibration_pair_count: int = 4096
    noise_std: float = 0.02
    seed: int = 42


def _pair_loader(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    pair_count: int,
    seed: int,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    pairs = build_balanced_pair_indices(labels.numpy(), pair_count=pair_count, seed=seed)
    dataset = TensorDataset(
        features[torch.from_numpy(pairs.left)],
        features[torch.from_numpy(pairs.right)],
        torch.from_numpy(pairs.targets),
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )


@torch.no_grad()
def _validation_loss(
    model: MultiSimilaritySiamese,
    loader: DataLoader,
    *,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for left, right, targets in loader:
        left = _to_two_channel(left.to(device, non_blocking=True), pos_count=30)
        right = _to_two_channel(right.to(device, non_blocking=True), pos_count=30)
        targets = targets.to(device, non_blocking=True)
        loss = F.binary_cross_entropy_with_logits(model(left, right), targets)
        total += float(loss) * len(targets)
        count += len(targets)
    return total / max(count, 1)


def train_multi_similarity_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    validation_x: torch.Tensor,
    validation_y: torch.Tensor,
    *,
    device: torch.device,
    config: OneShotTrainingConfig,
) -> tuple[MultiSimilaritySiamese, dict[str, object]]:
    """Fit the BCE pair objective on CUDA and return the best checkpoint."""

    device = require_cuda(str(device))
    seed_everything(config.seed)
    model = MultiSimilaritySiamese(
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
        similarity_mode=config.similarity_mode,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    grad_scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    validation_loader = _pair_loader(
        validation_x,
        validation_y,
        pair_count=min(config.validation_pair_count, max(2, len(validation_x) * 4 // 2 * 2)),
        seed=config.seed + 100_000,
        batch_size=config.batch_size,
        shuffle=False,
    )
    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(1, config.epochs + 1):
        train_loader = _pair_loader(
            train_x,
            train_y,
            pair_count=config.pair_count,
            seed=config.seed + epoch,
            batch_size=config.batch_size,
            shuffle=True,
        )
        model.train()
        train_total = 0.0
        train_count = 0
        for left, right, targets in train_loader:
            left = left.to(device, non_blocking=True)
            right = right.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if config.noise_std:
                left = left + torch.randn_like(left) * config.noise_std
                right = right + torch.randn_like(right) * config.noise_std
            left = _to_two_channel(left, pos_count=30)
            right = _to_two_channel(right, pos_count=30)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                loss = F.binary_cross_entropy_with_logits(model(left, right), targets)
            grad_scaler.scale(loss).backward()
            grad_scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            train_total += float(loss.detach()) * len(targets)
            train_count += len(targets)
        scheduler.step()
        validation_loss = _validation_loss(model, validation_loader, device=device)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_total / max(train_count, 1),
                "validation_loss": validation_loss,
            }
        )
        if np.isfinite(validation_loss) and validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            stale = 0
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Siamese training did not produce a finite checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "duration_seconds": time.perf_counter() - started,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "history": history,
    }


@torch.no_grad()
def encode_traces(
    model: MultiSimilaritySiamese,
    features: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    device = require_cuda(str(device))
    model.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size].pin_memory().to(device, non_blocking=True)
        rows.append(model.encode(_to_two_channel(batch, pos_count=30)).float().cpu())
    return torch.cat(rows) if rows else torch.empty((0, model.embedding_dim))


def detection_metrics(
    *,
    is_known: np.ndarray,
    confidence: np.ndarray,
    accepted: np.ndarray,
    true_labels: np.ndarray | None = None,
) -> dict[str, float]:
    known = np.asarray(is_known, dtype=bool)
    scores = np.asarray(confidence, dtype=float)
    accepted = np.asarray(accepted, dtype=bool)
    result = {
        "known_unknown_auroc": float(roc_auc_score(known, scores)),
        "known_unknown_aupr": float(average_precision_score(known, scores)),
        "known_acceptance": float(accepted[known].mean()),
        "unknown_recall": float((~accepted[~known]).mean()),
        "unknown_false_acceptance": float(accepted[~known].mean()),
    }
    order = np.argsort(scores[known])[::-1]
    known_sorted = scores[known][order]
    threshold_95 = known_sorted[min(len(known_sorted) - 1, max(0, int(np.ceil(0.95 * len(known_sorted))) - 1))]
    result["fpr_at_95_tpr"] = float((scores[~known] >= threshold_95).mean())
    if true_labels is not None:
        normal = np.asarray(true_labels) == 0
        result["normal_rejection_rate"] = float((~accepted[normal]).mean()) if normal.any() else 0.0
    return result


def one_shot_classification_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seen_class_ids: Sequence[int],
    unseen_class_ids: Sequence[int],
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    def macro_accuracy(class_ids: Sequence[int]) -> float:
        values = [float((y_pred[y_true == value] == value).mean()) for value in class_ids if np.any(y_true == value)]
        return float(np.mean(values)) if values else 0.0

    seen = macro_accuracy(seen_class_ids)
    unseen = macro_accuracy(unseen_class_ids)
    harmonic = 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": macro_accuracy(sorted(np.unique(y_true))),
        "seen_accuracy": seen,
        "unseen_accuracy": unseen,
        "harmonic_mean": harmonic,
        "rejection_rate": float((y_pred == -1).mean()),
        "per_class_accuracy": {
            str(value): float((y_pred[y_true == value] == value).mean())
            for value in sorted(np.unique(y_true))
        },
    }


def config_dict(config: OneShotTrainingConfig) -> dict[str, object]:
    return asdict(config)
