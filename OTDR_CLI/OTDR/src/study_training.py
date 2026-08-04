from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .model_functions.study_models import (
    EpisodicMetricModel,
    PhysicsSemanticModel,
    SelfSupervisedTraceModel,
    nt_xent_loss,
    supervised_contrastive_loss,
)
from .model_functions.tcn import _to_two_channel
from .model_functions.zero_shot import require_cuda
from .zero_shot_training import seed_everything


@dataclass(frozen=True)
class ApproachAConfig:
    epochs: int = 16
    steps_per_epoch: int = 96
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embedding_dim: int = 128
    dropout: float = 0.1
    temperature: float = 0.1
    supcon_weight: float = 0.25
    hard_negative_weight: float = 0.1
    hard_negative_margin: float = 0.15
    noise_std: float = 0.02
    aggregation: str = "prototype"
    seed: int = 42


@dataclass(frozen=True)
class ApproachBConfig:
    epochs: int = 16
    steps_per_epoch: int = 96
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    latent_dim: int = 128
    dropout: float = 0.1
    temperature: float = 0.08
    prototype_weight: float = 1.0
    attribute_weight: float = 0.5
    supcon_weight: float = 0.1
    description_consistency_weight: float = 0.0
    prototype_mode: str = "physics"
    seen_penalty_grid_max: float = 3.0
    seed: int = 42


@dataclass(frozen=True)
class ApproachCConfig:
    epochs: int = 16
    steps_per_epoch: int = 96
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    embedding_dim: int = 128
    dropout: float = 0.1
    temperature: float = 0.1
    mask_ratio: float = 0.15
    reconstruction_weight: float = 1.0
    contrastive_weight: float = 1.0
    noise_std: float = 0.02
    scale_std: float = 0.03
    offset_std: float = 0.01
    density: str = "mahalanobis"
    knn_k: int = 10
    covariance_shrinkage: float = 0.1
    seed: int = 42


def _amp() -> tuple[torch.dtype, torch.amp.GradScaler]:
    bf16 = torch.cuda.is_bf16_supported()
    return (torch.bfloat16 if bf16 else torch.float16, torch.amp.GradScaler("cuda", enabled=not bf16))


def _balanced_batch(labels: torch.Tensor, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
    classes = sorted(int(value) for value in labels.unique())
    per_class = max(2, batch_size // len(classes))
    indices: list[np.ndarray] = []
    labels_np = labels.numpy()
    for class_id in classes:
        candidates = np.flatnonzero(labels_np == class_id)
        indices.append(rng.choice(candidates, size=per_class, replace=len(candidates) < per_class))
    result = np.concatenate(indices)
    if len(result) < batch_size:
        result = np.concatenate([result, rng.choice(len(labels), size=batch_size - len(result), replace=False)])
    rng.shuffle(result)
    return torch.from_numpy(result[:batch_size].astype(np.int64))


def _training_metadata(model: nn.Module, started: float, history: list[dict[str, float]], device: torch.device, amp_dtype: torch.dtype) -> dict[str, object]:
    return {
        "duration_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "amp_dtype": str(amp_dtype).replace("torch.", ""),
        "history": history,
    }


def train_approach_a(train_x: torch.Tensor, train_y: torch.Tensor, validation_x: torch.Tensor, validation_y: torch.Tensor,
                     *, device: torch.device, config: ApproachAConfig) -> tuple[EpisodicMetricModel, dict[str, object]]:
    device = require_cuda(str(device))
    seed_everything(config.seed)
    seen_ids = sorted(int(value) for value in train_y.unique())
    class_map = {value: index for index, value in enumerate(seen_ids)}
    mapped = torch.tensor([class_map[int(value)] for value in train_y], dtype=torch.long)
    model = EpisodicMetricModel(class_count=len(seen_ids), embedding_dim=config.embedding_dim, dropout=config.dropout, temperature=config.temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_dtype, scaler = _amp()
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(config.steps_per_epoch):
            idx = _balanced_batch(train_y, config.batch_size, rng)
            x = train_x[idx].to(device, non_blocking=True)
            target = mapped[idx].to(device, non_blocking=True)
            if config.noise_std:
                x = x + torch.randn_like(x) * config.noise_std
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits, embeddings = model(_to_two_channel(x, pos_count=30))
                ce = F.cross_entropy(logits, target)
                supcon = supervised_contrastive_loss(embeddings, target, temperature=config.temperature)
                true = logits.gather(1, target[:, None]).squeeze(1)
                wrong = logits.masked_fill(F.one_hot(target, logits.shape[1]).bool(), -torch.inf).max(1).values
                hard = F.relu(wrong - true + config.hard_negative_margin).mean()
                loss = ce + config.supcon_weight * supcon + config.hard_negative_weight * hard
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        history.append({"epoch": float(epoch), "train_loss": total / config.steps_per_epoch})
    metadata = _training_metadata(model, started, history, device, amp_dtype)
    metadata["seen_class_ids"] = seen_ids
    metadata["config"] = asdict(config)
    return model, metadata


def train_approach_b(train_x: torch.Tensor, train_y: torch.Tensor, validation_x: torch.Tensor, validation_y: torch.Tensor,
                     prototypes: torch.Tensor, *, device: torch.device, config: ApproachBConfig) -> tuple[PhysicsSemanticModel, dict[str, object]]:
    device = require_cuda(str(device))
    seed_everything(config.seed)
    seen_ids = sorted(int(value) for value in train_y.unique())
    class_map = {value: index for index, value in enumerate(seen_ids)}
    mapped = torch.tensor([class_map[int(value)] for value in train_y], dtype=torch.long)
    seen_prototypes = prototypes[seen_ids].to(device)
    model = PhysicsSemanticModel(attribute_dim=prototypes.shape[1], latent_dim=config.latent_dim, dropout=config.dropout, temperature=config.temperature).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_dtype, scaler = _amp()
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(config.steps_per_epoch):
            idx = _balanced_batch(train_y, config.batch_size, rng)
            x = train_x[idx].to(device, non_blocking=True)
            original_labels = train_y[idx].to(device)
            target = mapped[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                logits, attributes, latent = model(_to_two_channel(x, pos_count=30), seen_prototypes)
                ce = F.cross_entropy(logits, target)
                attr = F.smooth_l1_loss(attributes, prototypes.to(device)[original_labels])
                supcon = supervised_contrastive_loss(latent, target, temperature=config.temperature)
                loss = config.prototype_weight * ce + config.attribute_weight * attr + config.supcon_weight * supcon
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        history.append({"epoch": float(epoch), "train_loss": total / config.steps_per_epoch})
    metadata = _training_metadata(model, started, history, device, amp_dtype)
    metadata["seen_class_ids"] = seen_ids
    metadata["config"] = asdict(config)
    return model, metadata


def _augment_ssl(x: torch.Tensor, config: ApproachCConfig) -> tuple[torch.Tensor, torch.Tensor]:
    view = x.clone()
    positions = view[:, 1:]
    if config.scale_std:
        positions.mul_(1.0 + torch.randn((len(x), 1), device=x.device) * config.scale_std)
    if config.offset_std:
        positions.add_(torch.randn((len(x), 1), device=x.device) * config.offset_std)
    if config.noise_std:
        view.add_(torch.randn_like(view) * config.noise_std)
    mask = torch.rand_like(positions) < config.mask_ratio
    positions.masked_fill_(mask, 0.0)
    return view, mask


def train_approach_c(train_x: torch.Tensor, *, device: torch.device, config: ApproachCConfig) -> tuple[SelfSupervisedTraceModel, dict[str, object]]:
    device = require_cuda(str(device))
    seed_everything(config.seed)
    model = SelfSupervisedTraceModel(embedding_dim=config.embedding_dim, dropout=config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_dtype, scaler = _amp()
    rng = np.random.default_rng(config.seed)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(1, config.epochs + 1):
        model.train()
        total = 0.0
        for _ in range(config.steps_per_epoch):
            indices = torch.from_numpy(rng.choice(len(train_x), size=config.batch_size, replace=len(train_x) < config.batch_size).astype(np.int64))
            target = train_x[indices].to(device, non_blocking=True)
            left, left_mask = _augment_ssl(target, config)
            right, right_mask = _augment_ssl(target, config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                left_z, left_reconstruction = model(_to_two_channel(left, pos_count=30))
                right_z, right_reconstruction = model(_to_two_channel(right, pos_count=30))
                reconstruct = (F.mse_loss(left_reconstruction[:, 1:][left_mask], target[:, 1:][left_mask]) +
                               F.mse_loss(right_reconstruction[:, 1:][right_mask], target[:, 1:][right_mask])) / 2
                contrast = nt_xent_loss(left_z, right_z, temperature=config.temperature)
                loss = config.reconstruction_weight * reconstruct + config.contrastive_weight * contrast
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        history.append({"epoch": float(epoch), "train_loss": total / config.steps_per_epoch})
    metadata = _training_metadata(model, started, history, device, amp_dtype)
    metadata["config"] = asdict(config)
    return model, metadata


@torch.no_grad()
def encode(model: nn.Module, features: torch.Tensor, *, device: torch.device, kind: str, batch_size: int = 1024) -> torch.Tensor:
    device = require_cuda(str(device))
    model.eval()
    rows: list[torch.Tensor] = []
    for start in range(0, len(features), batch_size):
        x = features[start:start + batch_size].pin_memory().to(device, non_blocking=True)
        channels = _to_two_channel(x, pos_count=30)
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            if kind == "a":
                value = model.encoder(channels)
            elif kind == "b":
                _, value, _ = model(channels, None)
            elif kind == "c":
                value = model.encoder(channels)
            else:
                raise ValueError(f"Unknown approach kind: {kind}")
        rows.append(value.float().cpu())
    dim = getattr(getattr(model, "encoder", None), "embedding_dim", 0)
    return torch.cat(rows) if rows else torch.empty((0, dim))
