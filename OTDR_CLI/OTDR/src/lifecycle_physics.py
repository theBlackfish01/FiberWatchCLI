from __future__ import annotations

"""Coherent OTDR counterfactual exposure for KPSC ablations."""

from dataclasses import asdict, dataclass
import copy
import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .lifecycle_data import LifecycleBatch
from .model_functions.lifecycle import FeatureAssistedOTDR
from .model_functions.zero_shot import require_cuda
from .zero_shot_training import seed_everything
from .event_openworld_baselines import heuristic_physics_factors


@dataclass(frozen=True)
class PhysicsOEConfig:
    mode: str = "diverse_physics"
    epochs: int = 3
    steps_per_epoch: int = 20
    batch_size: int = 256
    learning_rate: float = 1e-4
    margin: float = 1.0
    oe_weight: float = 0.5
    preservation_weight: float = 1.0
    seed: int = 42


class CoherentOTDRCounterfactuals:
    """Jointly alter waveform, loss, and reflectance in normalized feature space."""

    MODES = ("reflection", "attenuation", "terminal", "broad", "multi_event", "mixed")

    def render(
        self,
        trace: torch.Tensor,
        context: torch.Tensor,
        missing: torch.Tensor,
        *,
        generator: torch.Generator,
        diverse: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if trace.ndim != 2 or trace.shape[1] != 30 or context.shape != (len(trace), 3):
            raise ValueError("Counterfactual inputs must be [N,30] trace and [N,3] context.")
        count = len(trace)
        clusters = (
            torch.arange(count, device=trace.device) % len(self.MODES)
            if diverse
            else torch.zeros(count, dtype=torch.long, device=trace.device)
        )
        clusters = clusters[torch.randperm(count, generator=generator, device=trace.device)]
        x = trace.clone()
        c = context.clone()
        m = missing.clone()
        axis = torch.arange(30, device=x.device, dtype=x.dtype)[None, :]
        centers = torch.randint(6, 24, (count, 1), generator=generator, device=x.device)
        amplitude = 0.8 + 1.2 * torch.rand((count, 1), generator=generator, device=x.device)
        gaussian = torch.exp(-0.5 * ((axis - centers) / 1.2).square())
        for cluster in range(len(self.MODES)):
            mask = clusters == cluster
            if not mask.any():
                continue
            if cluster == 0:  # reflection
                x[mask] += amplitude[mask] * gaussian[mask]
                c[mask, 2] += amplitude[mask, 0]
                m[mask, 2] = 0
            elif cluster == 1:  # non-reflective attenuation
                step = axis >= centers
                x[mask] -= amplitude[mask] * step[mask]
                c[mask, 1] += amplitude[mask, 0]
                m[mask, 1] = 0
            elif cluster == 2:  # terminal drop
                step = axis >= centers
                x[mask] -= 1.5 * amplitude[mask] * step[mask]
                c[mask, 1] += 1.5 * amplitude[mask, 0]
                m[mask, 1] = 0
            elif cluster == 3:  # broad loss
                broad = torch.exp(-0.5 * ((axis - centers) / 4.0).square())
                x[mask] -= amplitude[mask] * broad[mask]
                c[mask, 1] += 0.7 * amplitude[mask, 0]
                m[mask, 1] = 0
            elif cluster == 4:  # multiple reflective events
                second = torch.exp(-0.5 * ((axis - (centers + 5).clamp_max(27)) / 1.2).square())
                x[mask] += amplitude[mask] * (gaussian[mask] + 0.7 * second[mask])
                c[mask, 2] += 1.2 * amplitude[mask, 0]
                c[mask, 1] += 0.3 * amplitude[mask, 0]
                m[mask, 1:] = 0
            else:  # coherent reflection plus following attenuation
                step = axis >= (centers + 2)
                x[mask] += amplitude[mask] * gaussian[mask] - 0.7 * amplitude[mask] * step[mask]
                c[mask, 2] += amplitude[mask, 0]
                c[mask, 1] += 0.7 * amplitude[mask, 0]
                m[mask, 1:] = 0
        return x, c, m, clusters


@torch.no_grad()
def event_grammar_residual(
    trace: torch.Tensor,
    context: torch.Tensor,
    recipe_means: torch.Tensor,
    recipe_stds: torch.Tensor,
    *,
    known_class_ids: tuple[int, ...],
) -> np.ndarray:
    """Deterministic physics-factor residual to the nearest known event recipe."""
    if trace.ndim != 2 or trace.shape[1] != 30 or context.shape != (len(trace), 3):
        raise ValueError("Physics residual requires trace [N,30] and context [N,3].")
    features = torch.cat((context[:, :1].float(), trace.float()), dim=1)
    factors = heuristic_physics_factors(features)
    means = recipe_means.float().cpu()
    stds = recipe_stds.float().cpu()
    distance = (
        (factors[:, None, :] - means[None, :, :]).square()
        / stds[None, :, :].square().clamp_min(1e-6)
    ).mean(2)
    return distance[:, list(known_class_ids)].amin(1).numpy()


def finetune_physics_oe(
    model: FeatureAssistedOTDR,
    train: LifecycleBatch,
    *,
    device: torch.device | str,
    config: PhysicsOEConfig,
) -> tuple[FeatureAssistedOTDR, dict[str, Any]]:
    device = require_cuda(str(device))
    if config.mode not in {"generic", "pc2_physics", "diverse_physics", "diverse_anchor"}:
        raise ValueError("Unknown physics OE mode.")
    seed_everything(config.seed)
    model = model.to(device)
    teacher = copy.deepcopy(model).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    renderer = CoherentOTDRCounterfactuals()
    generator = torch.Generator(device=device).manual_seed(config.seed + 991)
    rng = np.random.default_rng(config.seed)
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)
    history = []
    started = time.perf_counter()
    for epoch in range(config.epochs):
        totals = np.zeros(4)
        model.train()
        for _ in range(config.steps_per_epoch):
            indices = torch.from_numpy(rng.choice(len(train), config.batch_size, replace=len(train) < config.batch_size))
            trace = train.trace[indices].pin_memory().to(device, non_blocking=True)
            context = train.context[indices].pin_memory().to(device, non_blocking=True)
            missing = train.context_missing[indices].pin_memory().to(device, non_blocking=True)
            labels = train.labels[indices].pin_memory().to(device, non_blocking=True)
            if config.mode == "generic":
                synthetic_trace = torch.flip(trace, dims=[1]) + torch.randn_like(trace) * 0.3
                synthetic_context = context[torch.randperm(len(context), generator=generator, device=device)]
                synthetic_missing = missing
            else:
                synthetic_trace, synthetic_context, synthetic_missing, _ = renderer.render(
                    trace, context, missing, generator=generator,
                    diverse=config.mode in {"diverse_physics", "diverse_anchor"},
                )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype):
                real = model(trace, context, missing)
                synthetic = model(synthetic_trace, synthetic_context, synthetic_missing)
                ce = F.cross_entropy(real["logits"], labels)
                normal_energy = -torch.logsumexp(real["logits"], dim=1)
                outlier_energy = -torch.logsumexp(synthetic["logits"], dim=1)
                oe = F.relu(config.margin + normal_energy - outlier_energy).mean()
                if config.mode == "diverse_anchor":
                    with torch.no_grad():
                        teacher_logits = teacher(trace, context, missing)["logits"]
                    preservation = F.kl_div(
                        F.log_softmax(real["logits"], dim=1),
                        F.softmax(teacher_logits, dim=1),
                        reduction="batchmean",
                    )
                else:
                    preservation = ce.new_zeros(())
                loss = ce + config.oe_weight * oe + config.preservation_weight * preservation
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            totals += [float(loss.detach()), float(ce.detach()), float(oe.detach()), float(preservation.detach())]
        history.append({
            "epoch": epoch + 1, "loss": totals[0] / config.steps_per_epoch,
            "classification": totals[1] / config.steps_per_epoch,
            "outlier_exposure": totals[2] / config.steps_per_epoch,
            "preservation_kl": totals[3] / config.steps_per_epoch,
        })
    return model, {
        "config": asdict(config),
        "history": history,
        "duration_seconds": time.perf_counter() - started,
        "device": str(device),
        "counterfactual_clusters": list(CoherentOTDRCounterfactuals.MODES),
        "outer_heldout_used": False,
    }
