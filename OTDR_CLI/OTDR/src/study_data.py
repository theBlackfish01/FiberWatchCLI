from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .one_shot_data import OneShotSplit, build_one_shot_split
from .zero_shot_data import INPUT_COLUMNS, FORBIDDEN_FEATURES, OuterFold, build_outer_fold, validate_zero_shot_frame


@dataclass(frozen=True)
class PreparedFold:
    outer: OuterFold
    enrollment: OneShotSplit
    scaler: StandardScaler
    train_x: torch.Tensor
    train_y: torch.Tensor
    validation_x: torch.Tensor
    validation_y: torch.Tensor
    seen_test_x: torch.Tensor
    seen_test_y: torch.Tensor
    support_x: torch.Tensor
    support_y: torch.Tensor
    query_x: torch.Tensor
    query_y: torch.Tensor


def feature_signature() -> str:
    return hashlib.sha256("|".join(INPUT_COLUMNS).encode()).hexdigest()


def validate_model_features(columns: list[str] | tuple[str, ...]) -> None:
    if list(columns) != INPUT_COLUMNS:
        raise ValueError(f"Model features must be exactly {INPUT_COLUMNS}.")
    forbidden = set(columns) & FORBIDDEN_FEATURES
    if forbidden:
        raise ValueError(f"Forbidden model inputs: {sorted(forbidden)}")


def transform(frame: pd.DataFrame, scaler: StandardScaler) -> tuple[torch.Tensor, torch.Tensor]:
    validate_model_features(INPUT_COLUMNS)
    x = scaler.transform(frame[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True)).astype(np.float32)
    y = frame["Class"].to_numpy(dtype=np.int64, copy=True)
    return torch.from_numpy(x), torch.from_numpy(y)


def prepare_fold(frame: pd.DataFrame, *, holdout: tuple[int, int], seed: int, support_fraction: float = 0.2) -> PreparedFold:
    validate_zero_shot_frame(frame)
    outer = build_outer_fold(frame, holdout=holdout, seed=seed)
    enrollment = build_one_shot_split(outer, support_fraction=support_fraction, seed=seed + 70_000)
    partitions = [outer.train, outer.validation, outer.seen_test, enrollment.support_pool, enrollment.query]
    group_sets = [set(part["_input_group"]) for part in partitions]
    for left in range(len(group_sets)):
        for right in range(left + 1, len(group_sets)):
            if group_sets[left] & group_sets[right]:
                raise AssertionError(f"Exact-input leakage between partitions {left} and {right}.")
    if set(outer.train["Class"]) & set(holdout) or set(outer.validation["Class"]) & set(holdout):
        raise AssertionError("Outer-held-out class reached model fitting or calibration.")
    scaler = StandardScaler().fit(outer.train[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    train_x, train_y = transform(outer.train, scaler)
    val_x, val_y = transform(outer.validation, scaler)
    seen_x, seen_y = transform(outer.seen_test, scaler)
    support_x, support_y = transform(enrollment.support_pool, scaler)
    query_x, query_y = transform(enrollment.query, scaler)
    return PreparedFold(
        outer, enrollment, scaler, train_x, train_y, val_x, val_y,
        seen_x, seen_y, support_x, support_y, query_x, query_y,
    )


def group_stratified_inner_splits(frame: pd.DataFrame, *, n_splits: int = 3, seed: int = 42) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if "_input_group" not in frame:
        raise ValueError("Frame must be exact-input grouped before inner splitting.")
    assignment: dict[str, int] = {}
    for class_id, class_frame in frame.groupby("Class"):
        groups = sorted(class_frame["_input_group"].unique(), key=lambda g: hashlib.sha256(f"{seed}:{class_id}:{g}".encode()).hexdigest())
        if len(groups) < n_splits:
            raise ValueError(f"Class {class_id} has too few groups for {n_splits} inner folds.")
        for position, group in enumerate(groups):
            assignment[group] = position % n_splits
    folds = frame["_input_group"].map(assignment).to_numpy()
    for fold in range(n_splits):
        train = np.flatnonzero(folds != fold)
        validation = np.flatnonzero(folds == fold)
        if set(frame.iloc[train]["_input_group"]) & set(frame.iloc[validation]["_input_group"]):
            raise AssertionError("Inner exact-input group leakage.")
        if set(frame.iloc[train]["Class"]) != set(frame.iloc[validation]["Class"]):
            raise AssertionError("Inner fold is not class-valid.")
        yield train, validation


def deterministic_support_indices(labels: np.ndarray, class_ids: tuple[int, ...], *, count: int, draw: int, seed: int) -> np.ndarray:
    selected: list[int] = []
    for class_id in class_ids:
        indices = np.flatnonzero(labels == class_id)
        if len(indices) < count:
            raise ValueError(f"Class {class_id} lacks {count} enrollment references.")
        ranked = sorted(indices.tolist(), key=lambda idx: hashlib.sha256(f"{seed}:{draw}:{class_id}:{idx}".encode()).hexdigest())
        selected.extend(ranked[:count])
    return np.asarray(selected, dtype=np.int64)


def load_frame(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)
