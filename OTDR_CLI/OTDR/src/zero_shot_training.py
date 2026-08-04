from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .model_functions.tcn import _to_two_channel
from .model_functions.zero_shot import ZeroShotClassifier
from .zero_shot_data import FaultPrototype, INPUT_COLUMNS, OuterFold


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 40
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 7
    noise_std: float = 0.02
    supcon_weight: float = 0.1
    seed: int = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def fit_seen_scaler(fold: OuterFold) -> StandardScaler:
    return StandardScaler().fit(fold.train[INPUT_COLUMNS].to_numpy(dtype=np.float32))


def transform_frame(frame: pd.DataFrame, scaler: StandardScaler) -> tuple[torch.Tensor, torch.Tensor]:
    values = scaler.transform(frame[INPUT_COLUMNS].to_numpy(dtype=np.float32)).astype(np.float32, copy=True)
    labels = frame["Class"].to_numpy(dtype=np.int64, copy=True)
    return torch.from_numpy(values), torch.from_numpy(labels)


def _load_sentence_encoder(model_name: str, device: torch.device, *, factory=None):
    if factory is None:
        from sentence_transformers import SentenceTransformer

        factory = SentenceTransformer
    try:
        return factory(model_name, device=str(device), local_files_only=True)
    except OSError:
        return factory(model_name, device=str(device))


def encode_fault_prototypes(
    prototypes: Sequence[FaultPrototype],
    *,
    model_name: str,
    device: torch.device,
) -> torch.Tensor:
    encoder = _load_sentence_encoder(model_name, device)
    encoder.eval()
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)
    descriptions = [text for item in prototypes for text in item.descriptions]
    embeddings = encoder.encode(
        descriptions,
        batch_size=32,
        convert_to_tensor=True,
        normalize_embeddings=True,
        device=str(device),
        show_progress_bar=False,
    )
    if not embeddings.is_cuda:
        raise RuntimeError("Text prototype embeddings were not computed on CUDA.")
    return embeddings.reshape(8, 5, -1).to(device=device, dtype=torch.float32)


def supervised_contrastive_loss(embeddings: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    similarities = embeddings @ embeddings.T / temperature
    eye = torch.eye(labels.numel(), dtype=torch.bool, device=labels.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    logits = similarities.masked_fill(eye, -torch.inf)
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positives.sum(1)
    valid = counts > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    return -(log_prob.masked_fill(~positives, 0.0).sum(1)[valid] / counts[valid]).mean()


def _loader(x: torch.Tensor, y: torch.Tensor, *, config: TrainingConfig, train: bool) -> DataLoader:
    if train:
        counts = torch.bincount(y)
        weights = torch.tensor([1.0 / float(counts[int(label)]) for label in y], dtype=torch.double)
        sampler = WeightedRandomSampler(weights, num_samples=len(y), replacement=True)
    else:
        sampler = None
    return DataLoader(
        TensorDataset(x, y),
        batch_size=config.batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )


def _mapped_targets(labels: torch.Tensor, class_ids: Sequence[int]) -> torch.Tensor:
    mapping = torch.full((8,), -1, device=labels.device, dtype=torch.long)
    mapping[torch.tensor(class_ids, device=labels.device)] = torch.arange(len(class_ids), device=labels.device)
    mapped = mapping[labels]
    if (mapped < 0).any():
        raise ValueError("Training labels include a class outside the active seen classes.")
    return mapped


@torch.no_grad()
def predict_scores(
    model: ZeroShotClassifier,
    x: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int = 512,
) -> torch.Tensor:
    model.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(x), batch_size):
        batch = x[start : start + batch_size].pin_memory().to(device, non_blocking=True)
        model_input = _to_two_channel(batch, pos_count=30)
        scores, _ = model(model_input, prototypes)
        rows.append(scores.float().cpu())
    return torch.cat(rows) if rows else torch.empty((0, 8))


def train_zero_shot_model(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    prototypes: torch.Tensor,
    *,
    seen_class_ids: Sequence[int],
    device: torch.device,
    config: TrainingConfig,
) -> tuple[ZeroShotClassifier, dict[str, object]]:
    seed_everything(config.seed)
    model = ZeroShotClassifier(embedding_dim=prototypes.shape[-1]).to(device)
    if next(model.parameters()).device.type != "cuda":
        raise RuntimeError("Zero-shot sensor model is not on CUDA.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(config.epochs, 1))
    use_bf16 = torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=not use_bf16)
    train_loader = _loader(train_x, train_y, config=config, train=True)
    active = torch.tensor(list(seen_class_ids), device=device)
    active_prototypes = prototypes[active]
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    best_epoch = 0
    stale = 0
    started = time.perf_counter()
    history: list[dict[str, float]] = []
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            if config.noise_std:
                xb = xb + torch.randn_like(xb) * config.noise_std
            model_input = _to_two_channel(xb, pos_count=30)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                scores, embeddings = model(model_input, active_prototypes)
                targets = _mapped_targets(yb, seen_class_ids)
                loss = F.cross_entropy(scores, targets)
                loss = loss + config.supcon_weight * supervised_contrastive_loss(embeddings, yb)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach()) * len(xb)
            count += len(xb)
        scheduler.step()
        val_scores = predict_scores(model, val_x, prototypes, device=device)
        val_targets = _mapped_targets(val_y, seen_class_ids).cpu()
        val_loss = float(F.cross_entropy(val_scores[:, list(seen_class_ids)], val_targets))
        history.append({"epoch": float(epoch), "train_loss": total / max(count, 1), "val_loss": val_loss})
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            stale = 0
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training did not produce a finite checkpoint.")
    model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "duration_seconds": time.perf_counter() - started,
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "history": history,
    }


def apply_seen_penalty(
    scores: torch.Tensor,
    gamma: float,
    *,
    seen_class_ids: set[int],
    candidate_class_ids: Sequence[int],
) -> torch.Tensor:
    adjusted = scores.clone()
    for column, class_id in enumerate(candidate_class_ids):
        if class_id in seen_class_ids:
            adjusted[:, column] -= gamma
    return adjusted


def _macro_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray, classes: set[int]) -> float:
    values = [float((y_pred[y_true == cls] == cls).mean()) for cls in sorted(classes) if np.any(y_true == cls)]
    return float(np.mean(values)) if values else 0.0


def compute_gzsl_metrics(
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    seen_class_ids: set[int],
    unseen_class_ids: set[int],
) -> dict[str, float]:
    seen = _macro_class_accuracy(y_true, y_pred, seen_class_ids)
    unseen = _macro_class_accuracy(y_true, y_pred, unseen_class_ids)
    harmonic = 0.0 if seen + unseen == 0 else 2 * seen * unseen / (seen + unseen)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    seen_mask = np.isin(y_true, list(seen_class_ids))
    unseen_mask = np.isin(y_true, list(unseen_class_ids))
    seen_to_unseen = float(np.isin(y_pred[seen_mask], list(unseen_class_ids)).mean()) if seen_mask.any() else 0.0
    unseen_to_seen = float(np.isin(y_pred[unseen_mask], list(seen_class_ids)).mean()) if unseen_mask.any() else 0.0
    per_class = {
        str(class_id): float((y_pred[y_true == class_id] == class_id).mean())
        for class_id in sorted(set(y_true.tolist()))
    }
    return {
        "seen_accuracy": seen,
        "unseen_accuracy": unseen,
        "harmonic_mean": harmonic,
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "seen_to_unseen_error_rate": seen_to_unseen,
        "unseen_to_seen_error_rate": unseen_to_seen,
        "per_class_accuracy": per_class,
    }


def choose_seen_penalty(
    scores: torch.Tensor,
    labels: np.ndarray,
    *,
    seen_class_ids: set[int],
    candidate_class_ids: Sequence[int],
) -> tuple[float, list[dict[str, float]]]:
    rows: list[dict[str, float]] = []
    unseen = set(candidate_class_ids) - seen_class_ids
    for gamma in np.round(np.arange(-1.0, 1.0001, 0.05), 2):
        adjusted = apply_seen_penalty(scores, float(gamma), seen_class_ids=seen_class_ids, candidate_class_ids=candidate_class_ids)
        predictions = np.asarray([candidate_class_ids[index] for index in adjusted.argmax(1).tolist()])
        metrics = compute_gzsl_metrics(y_true=labels, y_pred=predictions, seen_class_ids=seen_class_ids, unseen_class_ids=unseen)
        rows.append({"gamma": float(gamma), **metrics})
    best = max(rows, key=lambda row: (row["harmonic_mean"], -abs(row["gamma"])))
    return float(best["gamma"]), rows


def gpu_metadata(device: torch.device) -> dict[str, object]:
    index = 0 if device.index is None else device.index
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": f"cuda:{index}",
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(index),
        "compute_capability": list(torch.cuda.get_device_capability(index)),
        "total_memory_bytes": int(properties.total_memory),
        "torch_version": torch.__version__,
        "compiled_cuda_version": torch.version.cuda,
    }


def save_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def config_dict(config: TrainingConfig) -> dict[str, object]:
    return asdict(config)
