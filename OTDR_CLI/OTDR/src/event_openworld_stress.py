from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .event_openworld_baselines import closed_set_group_split
from .event_openworld_metrics import ScoreNormalizer
from .event_openworld_training import ECConfig, PC2Config, infer_event_model, train_ec_czsl, train_pc2_oe
from .model_functions.event_openworld import load_event_recipes
from .model_functions.zero_shot import require_cuda
from .study_state import atomic_json
from .zero_shot_data import INPUT_COLUMNS


STRESS_GRID = {
    "snr": [0.0, 4.0, 8.0, 16.0, 32.0, 45.0],
    "position": [0.05, 0.12, 0.30, 0.55, 0.88, 0.95],
    "width": [0.01, 0.025, 0.08, 0.18, 0.28, 0.40],
    "magnitude": [0.03, 0.08, 0.25, 0.70, 1.40, 2.0],
    "overlap": [0.0, 0.02, 0.06, 0.12, 0.20, 0.32],
}


def controlled_render(
    factors: torch.Tensor,
    *,
    snr: float,
    position: float,
    width: float,
    magnitude: float,
    overlap: float,
    snr_mean: float,
    snr_scale: float,
    seed: int,
) -> torch.Tensor:
    device = factors.device
    generator = torch.Generator(device=device).manual_seed(seed)
    batch = len(factors)
    t = torch.linspace(0, 1, 30, device=device)[None, :]
    pos = torch.full((batch, 1), position, device=device)
    w = torch.full((batch, 1), max(width, 0.005), device=device)
    mag = torch.full((batch, 1), magnitude, device=device)
    z = (t - pos) / w
    baseline = -0.35 * (t - 0.5) * (0.45 + 1.10 * factors[:, 9:10])
    skew = 0.75 * (1.0 - factors[:, 7:8])
    reflection_width = torch.where(t < pos, 1.0 - skew, 1.0 + skew).clamp_min(0.20)
    reflection = factors[:, 0:1] * mag * torch.exp(-0.5 * (z / (0.4 * reflection_width)).square())
    step = -factors[:, 1:2] * mag * torch.sigmoid(z * (9 - 6 * factors[:, 2:3]))
    broad = -0.45 * factors[:, 2:3] * factors[:, 1:2] * mag * torch.sigmoid(z * 2.2)
    terminal = -1.6 * factors[:, 3:4] * mag * torch.sigmoid(z * 12)
    irregular = 0.22 * factors[:, 6:7] * mag * torch.sin((t - pos) * 55) * torch.exp(-0.5 * (z / 1.3).square())
    dead = -0.65 * factors[:, 11:12] * mag * torch.exp(-0.5 * ((t - pos - w) / (w * 1.5)).square())
    second_pos = (pos + overlap).clamp_max(0.99)
    second = factors[:, 10:11] * mag * 0.55 * torch.exp(-0.5 * ((t - second_pos) / (w * 0.55)).square())
    slope_contrast = -(factors[:, 5:6] - 0.5) * mag * (t - pos).clamp_min(0.0) * torch.sigmoid(z * 8.0)
    trace = baseline + slope_contrast + reflection + step + broad + terminal + irregular + dead + second
    noise_std = 0.015 + 0.20 * math.exp(-(snr - 4) / 9)
    trace += torch.randn(trace.shape, generator=generator, device=device) * noise_std
    snr_value = torch.full((batch, 1), (snr - snr_mean) / max(snr_scale, 1e-8), device=device)
    return torch.cat([snr_value, trace], 1).float()


def run_stress_validation(
    *, frame: pd.DataFrame, study_root: Path, recipe_path: Path, device: torch.device,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    train, validation, _ = closed_set_group_split(frame)
    scaler = StandardScaler().fit(train[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    def transform(part: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = scaler.transform(part[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True)).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(part["Class"].to_numpy(dtype=np.int64, copy=True))
    train_x, train_y = transform(train); validation_x, validation_y = transform(validation)
    recipes = load_event_recipes(recipe_path, device=device)
    models = {}
    for approach, cls in (("ec", ECConfig), ("pc2", PC2Config)):
        frozen = json.loads((study_root / "configs" / f"{approach}_frozen.json").read_text(encoding="utf-8"))["config"]
        config = cls(**{**frozen, "seed": 42})
        if approach == "ec":
            model, metadata = train_ec_czsl(train_x, train_y, recipes["means"].cpu(), recipes["stds"].cpu(),
                                             device=device, config=config)
        else:
            model, metadata = train_pc2_oe(
                train_x, train_y, recipes["means"].cpu(), recipes["stds"].cpu(),
                snr_mean=float(scaler.mean_[0]), snr_scale=float(scaler.scale_[0]), device=device, config=config,
            )
        validation_output = infer_event_model(model, validation_x, recipes["means"].cpu(), recipes["stds"].cpu(), device=device)
        normal = validation_y.numpy() == 0
        normalizer = ScoreNormalizer.fit(validation_output["novelty_components"].numpy()[normal], config.fusion_weights)
        models[approach] = (model, metadata, normalizer)
    rows = []
    defaults = {"snr": 20.0, "position": 0.50, "width": 0.08, "magnitude": 0.70, "overlap": 0.08}
    labels = torch.arange(1, 8, device=device).repeat_interleave(32)
    factor_means = recipes["means"][labels]
    for variable, values in STRESS_GRID.items():
        for value_index, value in enumerate(values):
            settings = {**defaults, variable: value}
            features = controlled_render(
                factor_means, snr=settings["snr"], position=settings["position"], width=settings["width"],
                magnitude=settings["magnitude"], overlap=settings["overlap"],
                snr_mean=float(scaler.mean_[0]), snr_scale=float(scaler.scale_[0]), seed=50_000 + value_index,
            ).cpu()
            for approach, (model, _, normalizer) in models.items():
                output = infer_event_model(model, features, recipes["means"].cpu(), recipes["stds"].cpu(), device=device)
                predicted = output["logits"].argmax(1)
                score = normalizer.transform(output["novelty_components"].numpy())
                rows.append({
                    "approach": approach, "variable": variable, "value": value,
                    "accuracy": float((predicted == labels.cpu()).float().mean()),
                    "worst_class_recall": float(min((predicted[labels.cpu() == class_id] == class_id).float().mean() for class_id in range(1, 8))),
                    "event_center_mae_bins": float(np.abs(output["center"].numpy() - settings["position"] * 29).mean()),
                    "novelty_score_mean": float(score.mean()),
                    "outside_frozen_renderer_range": bool(
                        (variable == "snr" and not 4 <= value <= 40) or
                        (variable == "position" and not 0.12 <= value <= 0.88) or
                        (variable == "width" and not 0.025 <= value <= 0.28) or
                        (variable == "magnitude" and not 0.08 <= value <= 1.40) or
                        (variable == "overlap" and not 0.02 <= value <= 0.20)
                    ),
                })
    result = {"schema_version": 1, "selection_use": "none; frozen finalists only", "grid": STRESS_GRID,
              "rows": rows, "model_training": {key: value[1] for key, value in models.items()}}
    atomic_json(study_root / "tables" / "stress_validation.json", result)
    pd.DataFrame(rows).to_csv(study_root / "tables" / "stress_validation.csv", index=False)
    return result
