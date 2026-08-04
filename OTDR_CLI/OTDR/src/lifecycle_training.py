from __future__ import annotations

"""CUDA-only training and inference for the shared lifecycle backbone."""

from dataclasses import asdict, dataclass, replace
import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lifecycle_data import LifecycleBatch
from .model_functions.lifecycle import FeatureAssistedOTDR, LifecycleModelConfig
from .model_functions.zero_shot import require_cuda
from .zero_shot_training import seed_everything


@dataclass(frozen=True)
class LifecycleTrainingConfig:
    epochs: int = 30
    steps_per_epoch: int = 80
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    supcon_weight: float = 0.1
    localization_weight: float = 0.05
    competence_weight: float = 0.05
    scalar_dropout: float = 0.1
    trace_noise_std: float = 0.015
    context_noise_std: float = 0.02
    patience: int = 6
    temperature: float = 0.1
    seed: int = 42


def _balanced_indices(labels: torch.Tensor, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
    y = labels.numpy()
    ids = np.unique(y)
    per_class = max(2, batch_size // len(ids))
    pieces = []
    for class_id in ids:
        candidates = np.flatnonzero(y == class_id)
        pieces.append(rng.choice(candidates, per_class, replace=len(candidates) < per_class))
    result = np.concatenate(pieces)
    if len(result) < batch_size:
        result = np.concatenate((result, rng.choice(len(y), batch_size - len(result), replace=True)))
    rng.shuffle(result)
    return torch.from_numpy(result[:batch_size].astype(np.int64))


def supervised_contrastive_loss(embedding: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    if len(embedding) < 2:
        return embedding.new_zeros(())
    similarity = embedding @ embedding.T / temperature
    diagonal = torch.eye(len(embedding), dtype=torch.bool, device=embedding.device)
    positive = labels[:, None].eq(labels[None, :]) & (~diagonal)
    logits = similarity.masked_fill(diagonal, -torch.inf)
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    counts = positive.sum(1)
    valid = counts > 0
    if not valid.any():
        return embedding.new_zeros(())
    return -((log_probability.masked_fill(~positive, 0).sum(1) / counts.clamp_min(1))[valid]).mean()


def _augment(
    trace: torch.Tensor,
    context: torch.Tensor,
    missing: torch.Tensor,
    config: LifecycleTrainingConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if config.trace_noise_std:
        trace = trace + torch.randn_like(trace) * config.trace_noise_std
    if config.context_noise_std:
        context = context + torch.randn_like(context) * config.context_noise_std * (1 - missing)
    if config.scalar_dropout:
        # SNR stays available; operational summaries are dropped as a branch-robustness regularizer.
        dropped = (torch.rand_like(missing[:, 1:]) < config.scalar_dropout) & (missing[:, 1:] == 0)
        context = context.clone()
        missing = missing.clone()
        context[:, 1:][dropped] = 0
        missing[:, 1:][dropped] = 1
    return trace, context, missing


@torch.no_grad()
def _validation_loss(
    model: FeatureAssistedOTDR,
    batch: LifecycleBatch,
    *,
    device: torch.device,
    batch_size: int = 2048,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    for start in range(0, len(batch), batch_size):
        stop = start + batch_size
        trace = batch.trace[start:stop].pin_memory().to(device, non_blocking=True)
        context = batch.context[start:stop].pin_memory().to(device, non_blocking=True)
        missing = batch.context_missing[start:stop].pin_memory().to(device, non_blocking=True)
        labels = batch.labels[start:stop].pin_memory().to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16):
            output = model(trace, context, missing)
            loss = F.cross_entropy(output["logits"], labels)
        total_loss += float(loss) * len(labels)
        correct += int((output["logits"].argmax(1) == labels).sum())
    return total_loss / len(batch), correct / len(batch)


def train_lifecycle_model(
    train: LifecycleBatch,
    validation: LifecycleBatch,
    *,
    device: torch.device | str,
    model_config: LifecycleModelConfig | None = None,
    training_config: LifecycleTrainingConfig | None = None,
) -> tuple[FeatureAssistedOTDR, dict[str, Any]]:
    device = require_cuda(str(device))
    config = training_config or LifecycleTrainingConfig()
    seed_everything(config.seed)
    model = FeatureAssistedOTDR(model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    rng = np.random.default_rng(config.seed)
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(config.epochs):
        model.train()
        totals = np.zeros(5, dtype=float)
        for _ in range(config.steps_per_epoch):
            indices = _balanced_indices(train.labels, config.batch_size, rng)
            trace = train.trace[indices].pin_memory().to(device, non_blocking=True)
            context = train.context[indices].pin_memory().to(device, non_blocking=True)
            missing = train.context_missing[indices].pin_memory().to(device, non_blocking=True)
            labels = train.labels[indices].pin_memory().to(device, non_blocking=True)
            positions = train.position[indices].pin_memory().to(device, non_blocking=True)
            trace, context, missing = _augment(trace, context, missing, config)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                output = model(trace, context, missing)
                ce = F.cross_entropy(output["logits"], labels)
                supcon = supervised_contrastive_loss(output["embedding"], labels, config.temperature)
                finite_position = torch.isfinite(positions)
                localization = (
                    F.smooth_l1_loss(output["position"][finite_position], positions[finite_position])
                    if finite_position.any()
                    else ce.new_zeros(())
                )
                correct_target = output["logits"].detach().argmax(1).eq(labels).to(output["competence"].dtype)
                competence = F.binary_cross_entropy_with_logits(output["competence"], correct_target)
                loss = (
                    ce
                    + config.supcon_weight * supcon
                    + config.localization_weight * localization
                    + config.competence_weight * competence
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            totals += [float(loss.detach()), float(ce.detach()), float(supcon.detach()),
                       float(localization.detach()), float(competence.detach())]
        validation_loss, validation_accuracy = _validation_loss(model, validation, device=device)
        history.append({
            "epoch": epoch + 1,
            "loss": totals[0] / config.steps_per_epoch,
            "classification": totals[1] / config.steps_per_epoch,
            "supervised_contrastive": totals[2] / config.steps_per_epoch,
            "localization": totals[3] / config.steps_per_epoch,
            "competence": totals[4] / config.steps_per_epoch,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
        })
        if validation_loss < best_loss - 1e-5:
            best_loss = validation_loss
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= config.patience:
                break
    if best_state is None:
        raise RuntimeError("Training failed to produce a finite validation checkpoint.")
    model.load_state_dict(best_state)
    metadata = {
        "training_config": asdict(config),
        "model_config": asdict(model.config),
        "history": history,
        "best_validation_loss": best_loss,
        "duration_seconds": time.perf_counter() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "amp_dtype": str(amp_dtype),
        "deterministic_algorithms": False,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }
    return model, metadata


@torch.no_grad()
def infer_lifecycle_model(
    model: FeatureAssistedOTDR,
    batch: LifecycleBatch,
    *,
    device: torch.device | str,
    batch_size: int = 2048,
) -> dict[str, torch.Tensor]:
    device = require_cuda(str(device))
    model = model.to(device).eval()
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    rows: dict[str, list[torch.Tensor]] = {}
    for start in range(0, len(batch), batch_size):
        stop = start + batch_size
        trace = batch.trace[start:stop].pin_memory().to(device, non_blocking=True)
        context = batch.context[start:stop].pin_memory().to(device, non_blocking=True)
        missing = batch.context_missing[start:stop].pin_memory().to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=amp_dtype):
            output = model(trace, context, missing)
        for name, value in output.items():
            rows.setdefault(name, []).append(value.detach().float().cpu())
    return {name: torch.cat(values) for name, values in rows.items()}
