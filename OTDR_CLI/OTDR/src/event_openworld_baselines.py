from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import weibull_min
from torch import nn
from torch.nn import functional as F

from .event_openworld_data import attach_input_groups
from .event_openworld_metrics import evaluate_zero_day
from .model_functions.event_openworld import MultiScaleBranch, derivative_channels, robust_linear_detrend
from .model_functions.zero_shot import require_cuda
from .zero_shot_data import INPUT_COLUMNS
from .zero_shot_training import seed_everything
from .study_state import atomic_json, file_sha256, validate_run, write_manifest


def heuristic_physics_factors(features: torch.Tensor) -> torch.Tensor:
    """Deterministic, metadata-free approximation of the frozen 12-factor bottleneck."""
    trace = features[:, 1:].float()
    detrended, _ = robust_linear_detrend(trace)
    first, second = derivative_channels(detrended)
    positive = detrended.amax(1).sigmoid()
    negative_step = (-first.amin(1) * 2).sigmoid()
    event = detrended.abs()
    peak = event.amax(1, keepdim=True).clamp_min(1e-6)
    width = ((event > peak * 0.4).float().mean(1) * 3).clamp(0, 1)
    terminality = ((trace[:, :5].mean(1) - trace[:, -5:].mean(1)) * 1.5).sigmoid()
    continuation = (1 - terminality).clamp(0, 1)
    slope_contrast = (first[:, 15:].mean(1) - first[:, :15].mean(1)).abs().mul(4).sigmoid()
    irregularity = (second.abs().mean(1) * 3).sigmoid()
    reverse = torch.flip(detrended, dims=[1])
    symmetry = (1 - (detrended - reverse).abs().mean(1) / (detrended.abs().mean(1) + 1e-5)).clamp(0, 1)
    local_energy = event.amax(1).sigmoid()
    global_energy = event.square().mean(1).sqrt().sigmoid()
    local_max = F.max_pool1d(event[:, None, :], 3, stride=1, padding=1).squeeze(1)
    multiplicity = (((event >= local_max - 1e-6) & (event > peak * 0.5)).sum(1).float() / 4).clamp(0, 1)
    peak_idx = event.argmax(1)
    dead_zone = []
    for row, index in zip(detrended, peak_idx, strict=True):
        left = row[max(0, int(index) - 2):int(index) + 1].mean()
        right = row[int(index) + 1:min(len(row), int(index) + 5)].mean() if int(index) + 1 < len(row) else row[-1]
        dead_zone.append((left - right).sigmoid())
    return torch.stack([
        positive, negative_step, width, terminality, continuation, slope_contrast,
        irregularity, symmetry, local_energy, global_energy, multiplicity, torch.stack(dead_zone),
    ], dim=1)


def deterministic_recipe_scores(
    features: torch.Tensor,
    recipe_means: torch.Tensor,
    recipe_stds: torch.Tensor,
    known_class_ids: list[int] | tuple[int, ...] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    factors = heuristic_physics_factors(features)
    distance = ((factors[:, None, :] - recipe_means[None, :, :]).square() / recipe_stds[None, :, :].square()).mean(-1)
    known = list(range(distance.shape[1])) if known_class_ids is None else list(known_class_ids)
    return -distance, distance[:, known].amin(1)


@dataclass(frozen=True)
class OpenMaxEVT:
    class_ids: tuple[int, ...]
    centroids: np.ndarray
    shapes: np.ndarray
    scales: np.ndarray

    @classmethod
    def fit(cls, embeddings: torch.Tensor, labels: torch.Tensor, class_ids: list[int], tail_size: int = 64) -> "OpenMaxEVT":
        values = F.normalize(embeddings.float(), dim=-1).numpy()
        labels_np = labels.numpy()
        centroids, shapes, scales = [], [], []
        for class_id in class_ids:
            class_values = values[labels_np == class_id]
            centroid = class_values.mean(0)
            centroid /= np.linalg.norm(centroid) + 1e-12
            distance = 1 - class_values @ centroid
            tail = np.sort(distance)[-min(tail_size, len(distance)):]
            tail = np.maximum(tail, 1e-8)
            if float(np.std(tail)) < 1e-8:
                shape, scale = 1.0, float(np.max(tail))
            else:
                try:
                    shape, _, scale = weibull_min.fit(tail, floc=0)
                except (ValueError, FloatingPointError):
                    shape, scale = 1.0, float(np.quantile(tail, 0.95))
            if not np.isfinite(shape) or not np.isfinite(scale):
                shape, scale = 1.0, float(np.quantile(tail, 0.95))
            centroids.append(centroid); shapes.append(shape); scales.append(max(scale, 1e-8))
        return cls(tuple(class_ids), np.asarray(centroids), np.asarray(shapes), np.asarray(scales))

    def novelty(self, embeddings: torch.Tensor) -> np.ndarray:
        values = F.normalize(embeddings.float(), dim=-1).numpy()
        distance = 1 - values @ self.centroids.T
        tail_probability = weibull_min.cdf(np.maximum(distance, 0), self.shapes[None, :], scale=self.scales[None, :])
        return tail_probability.min(axis=1)


class ClosedSetTraceClassifier(nn.Module):
    def __init__(self, kind: str = "multiscale", width: int = 64) -> None:
        super().__init__()
        if kind not in {"multiscale", "bilstm"}:
            raise ValueError("kind must be multiscale or bilstm")
        self.kind = kind
        self.gate = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 3), nn.Sigmoid())
        if kind == "multiscale":
            self.branch = MultiScaleBranch(3, width)
            hidden = width * 2
        else:
            self.lstm = nn.LSTM(2, width // 2, num_layers=1, batch_first=True, bidirectional=True)
            hidden = width
        self.head = nn.Sequential(nn.Linear(hidden, width), nn.GELU(), nn.Dropout(0.1), nn.Linear(width, 8))

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        snr, trace = features[:, :1], features[:, 1:]
        detrended, _ = robust_linear_detrend(trace)
        first, _ = derivative_channels(detrended)
        if self.kind == "multiscale":
            channels = torch.stack([detrended, first, trace], 1)
            embedding = self.branch(channels * (0.5 + self.gate(snr).unsqueeze(-1)))
        else:
            sequence = torch.stack([detrended, snr.expand(-1, 30)], dim=-1)
            values, _ = self.lstm(sequence)
            embedding = values.mean(1)
        return self.head(embedding), F.normalize(embedding, dim=-1)


def _balanced_indices(labels: torch.Tensor, batch_size: int, rng: np.random.Generator) -> torch.Tensor:
    pieces = []
    labels_np = labels.numpy()
    class_ids = sorted(int(value) for value in labels.unique())
    for class_id in class_ids:
        candidates = np.flatnonzero(labels_np == class_id)
        count = max(2, batch_size // len(class_ids))
        pieces.append(rng.choice(candidates, size=count, replace=len(candidates) < count))
    values = np.concatenate(pieces)
    rng.shuffle(values)
    return torch.from_numpy(values[:batch_size].astype(np.int64))


def train_closed_set_classifier(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    *,
    kind: str,
    device: torch.device,
    seed: int = 42,
    epochs: int = 8,
    steps_per_epoch: int = 64,
    batch_size: int = 512,
) -> tuple[ClosedSetTraceClassifier, dict[str, Any]]:
    device = require_cuda(str(device))
    seed_everything(seed)
    model = ClosedSetTraceClassifier(kind=kind).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    rng = np.random.default_rng(seed)
    started = time.perf_counter()
    history = []
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(epochs):
        total = 0.0
        for _ in range(steps_per_epoch):
            idx = _balanced_indices(train_y, batch_size, rng)
            x, y = train_x[idx].to(device), train_y[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=dtype):
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
            total += float(loss.detach())
        history.append({"epoch": epoch + 1, "loss": total / steps_per_epoch})
    return model, {
        "duration_seconds": time.perf_counter() - started,
        "parameter_count": sum(value.numel() for value in model.parameters()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "cuda_device": str(device), "history": history,
    }


@torch.no_grad()
def infer_closed_set(model: ClosedSetTraceClassifier, features: torch.Tensor, device: torch.device, batch_size: int = 2048) -> tuple[torch.Tensor, torch.Tensor]:
    device = require_cuda(str(device))
    model.eval()
    logits, embeddings = [], []
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    for start in range(0, len(features), batch_size):
        with torch.autocast("cuda", dtype=dtype):
            out, emb = model(features[start:start + batch_size].pin_memory().to(device))
        logits.append(out.float().cpu())
        embeddings.append(emb.float().cpu())
    return torch.cat(logits), torch.cat(embeddings)


def rocket_features(features: torch.Tensor, *, kernels: int = 256, seed: int = 42, device: torch.device) -> torch.Tensor:
    device = require_cuda(str(device))
    generator = torch.Generator(device=device).manual_seed(seed)
    trace = features[:, 1:].to(device)
    rows = []
    for kernel_size in (3, 5, 7, 9):
        count = kernels // 4
        weight = torch.randn(count, 1, kernel_size, generator=generator, device=device)
        weight -= weight.mean(-1, keepdim=True)
        bias = torch.rand(count, generator=generator, device=device) * 2 - 1
        chunks = []
        for start in range(0, len(trace), 2048):
            value = F.conv1d(trace[start:start + 2048, None, :], weight, bias=bias, padding=kernel_size // 2)
            chunks.append(torch.cat([value.amax(-1), (value > 0).float().mean(-1)], dim=1).cpu())
        rows.append(torch.cat(chunks))
    return torch.cat(rows, dim=1)


def closed_set_group_split(frame: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = attach_input_groups(frame).drop_duplicates("_input_group")
    groups = [[], [], []]
    for class_id, class_frame in frame.groupby("Class"):
        ranked = sorted(class_frame.index.tolist(), key=lambda index: hashlib.sha256(
            f"closed:{seed}:{class_id}:{frame.at[index, '_input_group']}".encode()
        ).hexdigest())
        first, second = int(len(ranked) * 0.7), int(len(ranked) * 0.85)
        for target, values in zip(groups, (ranked[:first], ranked[first:second], ranked[second:]), strict=True):
            target.extend(values)
    return tuple(frame.loc[values].copy() for values in groups)  # type: ignore[return-value]


def evaluate_rocket(train_x: torch.Tensor, train_y: torch.Tensor, test_x: torch.Tensor, test_y: torch.Tensor,
                    *, device: torch.device) -> dict[str, float]:
    train_features = rocket_features(train_x, device=device)
    test_features = rocket_features(test_x, device=device)
    scaler = StandardScaler().fit(train_features.numpy())
    classifier = RidgeClassifier(alpha=1.0).fit(scaler.transform(train_features.numpy()), train_y.numpy())
    predicted = classifier.predict(scaler.transform(test_features.numpy()))
    return {"accuracy": float((predicted == test_y.numpy()).mean()),
            "balanced_accuracy": float(balanced_accuracy_score(test_y.numpy(), predicted))}


def run_closed_set_sanity(frame: pd.DataFrame, *, study_root: Any, device: torch.device) -> dict[str, Any]:
    train, validation, test = closed_set_group_split(frame)
    split_path = study_root / "manifests" / "closed_set_sanity_groups.npz"
    split_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(split_path, **{
        name: np.asarray([bytes.fromhex(value) for value in part["_input_group"]], dtype="V32")
        for name, part in (("train", train), ("validation", validation), ("test", test))
    })
    scaler = StandardScaler().fit(train[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    def transform(part: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = scaler.transform(part[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True)).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(part["Class"].to_numpy(dtype=np.int64, copy=True))
    train_x, train_y = transform(train)
    validation_x, validation_y = transform(validation)
    test_x, test_y = transform(test)
    results: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "closed-set sanity only; never substituted for held-out-class results",
        "partitions": {"train": len(train), "validation": len(validation), "test": len(test)},
        "exact_group_manifest": str(split_path.relative_to(study_root).as_posix()),
        "exact_group_manifest_sha256": file_sha256(split_path),
        "features": INPUT_COLUMNS,
        "models": {},
    }
    checkpoint_root = study_root / "checkpoints" / "closed_set"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    for kind in ("multiscale", "bilstm"):
        model, metadata = train_closed_set_classifier(train_x, train_y, kind=kind, device=device,
                                                      epochs=6, steps_per_epoch=48)
        logits, embeddings = infer_closed_set(model, test_x, device)
        predicted = logits.argmax(1)
        results["models"][kind] = {
            "accuracy": float((predicted == test_y).float().mean()),
            "balanced_accuracy": float(balanced_accuracy_score(test_y.numpy(), predicted.numpy())),
            "per_class_recall": {str(class_id): float((predicted[test_y == class_id] == class_id).float().mean()) for class_id in range(8)},
            "metadata": metadata,
        }
        torch.save({"kind": kind, "state_dict": {key: value.cpu() for key, value in model.state_dict().items()},
                    "metadata": metadata}, checkpoint_root / f"{kind}.pt")
    results["models"]["rocket_style"] = evaluate_rocket(train_x, train_y, test_x, test_y, device=device)
    atomic_json(study_root / "tables" / "closed_set_sanity.json", results)
    return results


def run_outer_closed_encoder_open_set_baselines(
    *,
    fold: Any,
    tensor_fold: Any,
    holdout: tuple[int, int],
    seed: int,
    study_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Train/cache OpenMax and energy baselines on the strongest sanity encoder."""
    device = require_cuda(str(device))
    sanity_path = study_root / "tables" / "closed_set_sanity.json"
    if not sanity_path.exists():
        raise FileNotFoundError("Run the group-safe closed-set sanity benchmark before outer pilots.")
    sanity = json.loads(sanity_path.read_text(encoding="utf-8"))
    split_fingerprint = hashlib.sha256("\n".join(
        f"{name}:{group}"
        for name, part in fold.partitions().items()
        for group in sorted(part["_input_group"])
    ).encode()).hexdigest()
    run_id = f"closed-selected-{holdout[0]:02d}_{holdout[1]:02d}-s{seed}-v1"
    run_dir = study_root / "baselines" / "outer_closed_encoder" / run_id
    baseline_sources = [
        Path(__file__),
        Path(__file__).with_name("event_openworld_metrics.py"),
        Path(__file__).with_name("model_functions") / "event_openworld.py",
    ]
    baseline_source_sha256 = hashlib.sha256("\n".join(
        f"{path.name}:{file_sha256(path)}" for path in baseline_sources
    ).encode()).hexdigest()
    expected = {
        "run_id": run_id,
        "holdout": list(holdout),
        "seed": seed,
        "split_fingerprint": split_fingerprint,
        "selection_protocol": "outer_seen_validation_only_v1",
        "source_sha256": baseline_source_sha256,
        "cuda_verified": True,
    }
    valid, _ = validate_run(run_dir, expected)
    if valid:
        result = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        result["source_manifest_sha256"] = file_sha256(run_dir / "manifest.json")
        return result

    run_dir.mkdir(parents=True, exist_ok=True)
    train_x, train_y = tensor_fold.tensors["train"]
    validation_x, validation_y_tensor = tensor_fold.tensors["validation"]
    seen_ids = sorted(set(range(8)) - set(holdout))
    fitted: dict[str, tuple[ClosedSetTraceClassifier, dict[str, Any]]] = {}
    selection_scores: dict[str, float] = {}
    for index, candidate_kind in enumerate(("multiscale", "bilstm")):
        candidate_model, candidate_training = train_closed_set_classifier(
            train_x, train_y, kind=candidate_kind, device=device, seed=seed + index * 10_000,
            epochs=4, steps_per_epoch=32,
        )
        validation_logits, _ = infer_closed_set(candidate_model, validation_x, device)
        validation_local = validation_logits[:, seen_ids].argmax(1)
        validation_predicted = np.asarray(seen_ids)[validation_local.numpy()]
        selection_scores[candidate_kind] = float(balanced_accuracy_score(
            validation_y_tensor.numpy(), validation_predicted
        ))
        fitted[candidate_kind] = (candidate_model, candidate_training)
    kind = max(selection_scores, key=selection_scores.get)
    model, training = fitted[kind]
    fitted.clear()
    del candidate_model, candidate_training, validation_logits
    torch.cuda.empty_cache()
    names = ("train", "validation", "seen_test", "query")
    outputs = {
        name: infer_closed_set(model, tensor_fold.tensors[name][0], device)
        for name in names
    }
    validation_y = tensor_fold.tensors["validation"][1].numpy()
    seen_y = tensor_fold.tensors["seen_test"][1]
    query_y = tensor_fold.tensors["query"][1]
    test_y = torch.cat([seen_y, query_y]).numpy()
    test_snr = torch.cat([
        tensor_fold.tensors["seen_test"][0][:, 0],
        tensor_fold.tensors["query"][0][:, 0],
    ]).numpy()
    validation_normal = validation_y == 0
    predicted_local = torch.cat([outputs["seen_test"][0], outputs["query"][0]])[:, seen_ids].argmax(1)
    predicted = np.asarray(seen_ids)[predicted_local.numpy()]

    def evaluate(validation_score: np.ndarray, seen_score: np.ndarray, query_score: np.ndarray) -> dict[str, Any]:
        return evaluate_zero_day(
            validation_normal_score=validation_score[validation_normal],
            validation_normal_snr=tensor_fold.tensors["validation"][0][:, 0].numpy()[validation_normal],
            test_score=np.concatenate([seen_score, query_score]),
            test_snr=test_snr,
            true_labels=test_y,
            predicted=predicted,
            holdout=holdout,
            calibration="normalized",
        )

    energy = {
        name: -torch.logsumexp(logits[:, seen_ids], dim=1).numpy()
        for name, (logits, _) in outputs.items()
    }
    openmax = OpenMaxEVT.fit(outputs["train"][1], train_y, seen_ids)
    openmax_score = {name: openmax.novelty(embedding) for name, (_, embedding) in outputs.items()}
    result = {
        "schema_version": 1,
        "encoder_kind": kind,
        "selection": "highest outer-seen validation balanced accuracy; no held-out trace used",
        "selection_scores": selection_scores,
        "descriptive_all_class_sanity": {
            name: sanity["models"][name]["balanced_accuracy"]
            for name in ("multiscale", "bilstm")
        },
        "holdout": list(holdout),
        "seed": seed,
        "split_fingerprint": split_fingerprint,
        "calibration": "normalized; outer-seen validation normals only",
        "energy": evaluate(energy["validation"], energy["seen_test"], energy["query"]),
        "openmax_evt": evaluate(openmax_score["validation"], openmax_score["seen_test"], openmax_score["query"]),
        "training": training,
    }
    torch.save({
        "kind": kind,
        "holdout": list(holdout),
        "seed": seed,
        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "training": training,
    }, run_dir / "model.pt")
    np.savez_compressed(
        run_dir / "predictions.npz",
        labels=test_y.astype(np.int8),
        predicted=predicted.astype(np.int8),
        energy=np.concatenate([energy["seen_test"], energy["query"]]).astype(np.float32),
        openmax=np.concatenate([openmax_score["seen_test"], openmax_score["query"]]).astype(np.float32),
        validation_labels=validation_y.astype(np.int8),
        validation_energy=energy["validation"].astype(np.float32),
        validation_openmax=openmax_score["validation"].astype(np.float32),
    )
    atomic_json(run_dir / "metrics.json", result)
    write_manifest(run_dir, {**expected, "cuda_device": training["cuda_device"]})
    result["source_manifest_sha256"] = file_sha256(run_dir / "manifest.json")
    return result
