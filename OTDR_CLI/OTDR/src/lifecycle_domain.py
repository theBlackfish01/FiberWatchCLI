from __future__ import annotations

"""Label-independent morphology canonicalization and acquisition stress tests."""

from dataclasses import asdict, dataclass
import copy
import time
from typing import Literal

import numpy as np
import torch
from torch.nn import functional as F

from .lifecycle_data import LifecycleBatch
from .model_functions.lifecycle import FeatureAssistedOTDR, coral_loss, mmd_rbf
from .model_functions.zero_shot import require_cuda
from .zero_shot_training import seed_everything


@dataclass(frozen=True)
class EventProposal:
    center: float
    width: float
    confidence: float


def propose_event(trace: np.ndarray) -> EventProposal:
    """Propose an event from waveform derivatives without labels or Position."""
    values = np.asarray(trace, dtype=np.float64)
    if values.shape != (30,) or not np.isfinite(values).all():
        raise ValueError("Event proposal requires one finite 30-sample trace.")
    x = np.arange(30, dtype=float)
    trend = np.polyval(np.polyfit(x, values, 1), x)
    residual = values - trend
    derivative = np.gradient(residual)
    curvature = np.gradient(derivative)
    saliency = np.abs(derivative) + 0.5 * np.abs(curvature)
    baseline = np.median(saliency)
    scale = 1.4826 * np.median(np.abs(saliency - baseline)) + 1e-8
    weights = np.exp(np.clip((saliency - saliency.max()) / scale, -30, 0))
    center = float((weights * x).sum() / weights.sum())
    spread = float(np.sqrt((weights * (x - center) ** 2).sum() / weights.sum()))
    confidence = float((saliency.max() - baseline) / scale)
    return EventProposal(center=center, width=max(1.0, 2.355 * spread), confidence=confidence)


def canonicalize_trace(trace: np.ndarray, *, target_center: float = 14.5) -> tuple[np.ndarray, EventProposal]:
    """Shift/resample around a label-independent event proposal to fixed support."""
    values = np.asarray(trace, dtype=np.float64)
    proposal = propose_event(values)
    source = np.arange(30, dtype=float) + proposal.center - target_center
    canonical = np.interp(source, np.arange(30, dtype=float), values, left=values[0], right=values[-1])
    return canonical.astype(np.float32), proposal


def canonicalize_batch(traces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(traces, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 30:
        raise ValueError("Trace batch must have shape [samples, 30].")
    canonical, context = [], []
    for trace in values:
        transformed, proposal = canonicalize_trace(trace)
        canonical.append(transformed)
        context.append((proposal.center / 29.0, proposal.width / 30.0, proposal.confidence))
    return np.asarray(canonical, dtype=np.float32), np.asarray(context, dtype=np.float32)


StressKind = Literal[
    "snr_noise", "amplitude_scale", "amplitude_offset", "event_width", "position_shift",
    "resampling", "structured_noise", "loss_noise", "reflectance_noise",
    "missing_loss", "missing_reflectance", "scalar_quantization",
]


@dataclass(frozen=True)
class DomainAlignmentConfig:
    method: Literal["coral", "mmd"] = "coral"
    steps: int = 60
    batch_size: int = 256
    learning_rate: float = 1e-4
    alignment_weight: float = 0.05
    preservation_weight: float = 0.2
    seed: int = 42


def synthetic_domain_view(
    trace: torch.Tensor,
    context: torch.Tensor,
    missing: torch.Tensor,
    *,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a label-independent source-side acquisition pseudo-domain."""
    scale = 0.7 + 0.6 * torch.rand(
        (len(trace), 1), generator=generator, device=trace.device
    )
    offset = -0.5 + torch.rand(
        (len(trace), 1), generator=generator, device=trace.device
    )
    phase = 2 * torch.pi * torch.rand(
        (len(trace), 1), generator=generator, device=trace.device
    )
    axis = torch.arange(30, device=trace.device, dtype=trace.dtype)[None, :]
    target_trace = (
        trace * scale
        + 0.3 * offset
        + 0.12 * torch.sin(axis / 3 + phase)
        + 0.04 * torch.randn(
            trace.shape, generator=generator, device=trace.device, dtype=trace.dtype
        )
    )
    target_context = context + 0.08 * torch.randn(
        context.shape, generator=generator, device=context.device, dtype=context.dtype
    ) * (1 - missing)
    target_missing = missing.clone()
    dropped = torch.rand(
        (len(trace), 2), generator=generator, device=trace.device
    ) < 0.2
    available = target_missing[:, 1:] == 0
    dropped &= available
    target_context[:, 1:][dropped] = 0
    target_missing[:, 1:][dropped] = 1
    return target_trace, target_context, target_missing


def finetune_source_domain_alignment(
    model: FeatureAssistedOTDR,
    train: LifecycleBatch,
    *,
    device: torch.device | str,
    config: DomainAlignmentConfig,
) -> tuple[FeatureAssistedOTDR, dict[str, object]]:
    """Align a synthetic acquisition pseudo-domain without external target data."""
    device = require_cuda(str(device))
    if config.method not in {"coral", "mmd"}:
        raise ValueError("Domain alignment method must be coral or mmd.")
    seed_everything(config.seed)
    teacher = copy.deepcopy(model).to(device).eval()
    student = model.to(device).train()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    rng = np.random.default_rng(config.seed)
    generator = torch.Generator(device=device).manual_seed(config.seed)
    history = []
    started = time.perf_counter()
    for step in range(config.steps):
        indices = torch.from_numpy(
            rng.choice(len(train), config.batch_size, replace=len(train) < config.batch_size)
            .astype(np.int64)
        )
        trace = train.trace[indices].pin_memory().to(device, non_blocking=True)
        context = train.context[indices].pin_memory().to(device, non_blocking=True)
        missing = train.context_missing[indices].pin_memory().to(device, non_blocking=True)
        labels = train.labels[indices].pin_memory().to(device, non_blocking=True)
        target_trace, target_context, target_missing = synthetic_domain_view(
            trace, context, missing, generator=generator
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_probability = teacher(trace, context, missing)["logits"].softmax(1)
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ):
            source = student(trace, context, missing)
            target = student(target_trace, target_context, target_missing)
            classification = F.cross_entropy(source["logits"], labels)
            if config.method == "coral":
                alignment = coral_loss(source["morphology"], target["morphology"])
            else:
                alignment = mmd_rbf(source["morphology"], target["morphology"])
            preservation = F.kl_div(
                target["logits"].log_softmax(1),
                teacher_probability,
                reduction="batchmean",
            )
            loss = (
                classification
                + config.alignment_weight * alignment
                + config.preservation_weight * preservation
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
        optimizer.step()
        if step in {0, config.steps - 1}:
            history.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "classification": float(classification.detach()),
                "alignment": float(alignment.detach()),
                "preservation": float(preservation.detach()),
            })
    return student.eval(), {
        "config": asdict(config),
        "history": history,
        "duration_seconds": time.perf_counter() - started,
        "device": str(device),
        "target_labels_used": False,
        "external_target_samples_used": False,
        "pseudo_domain": "source-side amplitude/offset/noise/context-missingness",
    }


def finetune_unlabeled_target_alignment(
    model: FeatureAssistedOTDR,
    source: LifecycleBatch,
    target: LifecycleBatch,
    *,
    device: torch.device | str,
    config: DomainAlignmentConfig,
) -> tuple[FeatureAssistedOTDR, dict[str, object]]:
    """Transductively align to an unlabeled target mixture with preservation."""
    device = require_cuda(str(device))
    if config.method not in {"coral", "mmd"}:
        raise ValueError("Domain alignment method must be coral or mmd.")
    if len(target) < 2:
        raise ValueError("Unlabeled target alignment needs at least two target samples.")
    seed_everything(config.seed)
    teacher = copy.deepcopy(model).to(device).eval()
    student = model.to(device).train()
    optimizer = torch.optim.AdamW(
        student.parameters(), lr=config.learning_rate, weight_decay=1e-5
    )
    rng = np.random.default_rng(config.seed)
    history = []
    started = time.perf_counter()
    for step in range(config.steps):
        source_indices = torch.from_numpy(
            rng.choice(
                len(source), config.batch_size, replace=len(source) < config.batch_size
            ).astype(np.int64)
        )
        target_indices = torch.from_numpy(
            rng.choice(
                len(target), config.batch_size, replace=len(target) < config.batch_size
            ).astype(np.int64)
        )
        source_trace = source.trace[source_indices].pin_memory().to(device, non_blocking=True)
        source_context = source.context[source_indices].pin_memory().to(device, non_blocking=True)
        source_missing = source.context_missing[source_indices].pin_memory().to(device, non_blocking=True)
        source_labels = source.labels[source_indices].pin_memory().to(device, non_blocking=True)
        target_trace = target.trace[target_indices].pin_memory().to(device, non_blocking=True)
        target_context = target.context[target_indices].pin_memory().to(device, non_blocking=True)
        target_missing = target.context_missing[target_indices].pin_memory().to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            teacher_target = teacher(
                target_trace, target_context, target_missing
            )["logits"].softmax(1)
        with torch.autocast(
            "cuda",
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        ):
            source_output = student(source_trace, source_context, source_missing)
            target_output = student(target_trace, target_context, target_missing)
            classification = F.cross_entropy(source_output["logits"], source_labels)
            if config.method == "coral":
                alignment = coral_loss(
                    source_output["morphology"], target_output["morphology"]
                )
            else:
                alignment = mmd_rbf(
                    source_output["morphology"], target_output["morphology"]
                )
            preservation = F.kl_div(
                target_output["logits"].log_softmax(1),
                teacher_target,
                reduction="batchmean",
            )
            loss = (
                classification
                + config.alignment_weight * alignment
                + config.preservation_weight * preservation
            )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 2.0)
        optimizer.step()
        if step in {0, config.steps - 1}:
            history.append({
                "step": step + 1,
                "loss": float(loss.detach()),
                "classification": float(classification.detach()),
                "alignment": float(alignment.detach()),
                "preservation": float(preservation.detach()),
            })
    return student.eval(), {
        "config": asdict(config),
        "history": history,
        "duration_seconds": time.perf_counter() - started,
        "device": str(device),
        "target_labels_used": False,
        "target_query_mixture_used": True,
        "transductive": True,
    }


def apply_stress(
    trace: np.ndarray,
    context: np.ndarray,
    *,
    kind: StressKind,
    severity: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply a deterministic declared stress; returns trace, context, missing mask."""
    if not 0 <= severity <= 1:
        raise ValueError("severity must be in [0,1].")
    x = np.asarray(trace, dtype=np.float64).copy()
    c = np.asarray(context, dtype=np.float64).copy()
    if x.ndim != 2 or x.shape[1] != 30 or c.shape != (len(x), 3):
        raise ValueError("Stress inputs must be trace [N,30] and context [N,3].")
    missing = ~np.isfinite(c)
    rng = np.random.default_rng(seed)
    axis = np.arange(30, dtype=float)
    if kind == "snr_noise":
        c[:, 0] += rng.normal(0, severity * 3.0, len(c))
        x += rng.normal(0, severity * 0.25, x.shape)
    elif kind == "amplitude_scale":
        x *= 1 + severity * rng.uniform(-0.5, 0.5, (len(x), 1))
    elif kind == "amplitude_offset":
        x += severity * rng.uniform(-1, 1, (len(x), 1))
    elif kind == "event_width":
        factor = 1 + severity
        x = np.asarray([
            np.interp((axis - 14.5) / factor + 14.5, axis, row, left=row[0], right=row[-1])
            for row in x
        ])
    elif kind == "position_shift":
        shifts = rng.integers(-max(1, round(6 * severity)), max(2, round(6 * severity) + 1), len(x))
        x = np.asarray([np.interp(axis - shift, axis, row, left=row[0], right=row[-1]) for row, shift in zip(x, shifts)])
    elif kind == "resampling":
        coarse = max(8, int(round(30 * (1 - 0.6 * severity))))
        coarse_axis = np.linspace(0, 29, coarse)
        x = np.asarray([np.interp(axis, coarse_axis, np.interp(coarse_axis, axis, row)) for row in x])
    elif kind == "structured_noise":
        phase = rng.uniform(0, 2 * np.pi, (len(x), 1))
        x += severity * 0.3 * np.sin(axis[None, :] / 3 + phase)
    elif kind == "loss_noise":
        c[:, 1] += rng.normal(0, severity * 2.0, len(c))
    elif kind == "reflectance_noise":
        c[:, 2] += rng.normal(0, severity * 2.0, len(c))
    elif kind == "missing_loss":
        mask = rng.random(len(c)) < severity
        c[mask, 1] = np.nan
        missing[mask, 1] = True
    elif kind == "missing_reflectance":
        mask = rng.random(len(c)) < severity
        c[mask, 2] = np.nan
        missing[mask, 2] = True
    elif kind == "scalar_quantization":
        step = max(1e-4, severity)
        c = np.round(c / step) * step
    else:
        raise ValueError(f"Unknown stress kind: {kind}")
    return x.astype(np.float32), c.astype(np.float32), missing.astype(np.float32)
