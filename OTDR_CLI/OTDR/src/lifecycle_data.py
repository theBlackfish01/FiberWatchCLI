from __future__ import annotations

"""Data contracts and leakage-safe preprocessing for the OTDR lifecycle study."""

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Iterable, Literal

import numpy as np
import pandas as pd
import torch

from .event_openworld_data import (
    EventOpenWorldFold,
    attach_input_groups,
    build_event_openworld_fold,
    validate_fold_isolation,
)


TRACE_COLUMNS = tuple(f"P{i}" for i in range(1, 31))
CONTEXT_COLUMNS = ("SNR", "loss", "Reflectance")
GROUP_COLUMNS = ("SNR", *TRACE_COLUMNS)
TARGET_COLUMNS = frozenset({"Class", "Position"})
FeatureRegime = Literal["full", "trace_only", "summary_only"]
FEATURE_REGIMES: dict[FeatureRegime, tuple[str, ...]] = {
    "full": (*GROUP_COLUMNS, "loss", "Reflectance"),
    "trace_only": GROUP_COLUMNS,
    "summary_only": CONTEXT_COLUMNS,
}


@dataclass(frozen=True)
class LifecycleScaler:
    regime: FeatureRegime
    trace_location: tuple[float, ...]
    trace_scale: tuple[float, ...]
    context_median: tuple[float, ...]
    context_location: tuple[float, ...]
    context_scale: tuple[float, ...]
    context_active: tuple[bool, ...]

    def payload(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class LifecycleBatch:
    trace: torch.Tensor
    context: torch.Tensor
    context_missing: torch.Tensor
    labels: torch.Tensor
    position: torch.Tensor
    group_ids: tuple[str, ...]

    def __len__(self) -> int:
        return len(self.labels)


@dataclass(frozen=True)
class LifecycleTensorFold:
    split: EventOpenWorldFold
    scaler: LifecycleScaler
    batches: dict[str, LifecycleBatch]


def validate_feature_contract(
    frame: pd.DataFrame,
    *,
    regime: FeatureRegime,
    requested_inputs: Iterable[str] | None = None,
) -> None:
    if regime not in FEATURE_REGIMES:
        raise ValueError(f"Unknown feature regime: {regime}")
    inputs = tuple(FEATURE_REGIMES[regime] if requested_inputs is None else requested_inputs)
    forbidden = TARGET_COLUMNS.intersection(inputs)
    if forbidden:
        raise ValueError(f"Targets cannot be inference inputs: {sorted(forbidden)}")
    unexpected = set(inputs) - set(FEATURE_REGIMES[regime])
    if unexpected:
        raise ValueError(f"Inputs outside the {regime} feature contract: {sorted(unexpected)}")
    required = {"Class", *GROUP_COLUMNS}
    if regime in {"full", "summary_only"}:
        required.update({"loss", "Reflectance"})
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing lifecycle columns: {missing}")
    labels = pd.to_numeric(frame["Class"], errors="coerce")
    if labels.isna().any() or not np.allclose(labels, np.round(labels)):
        raise ValueError("Class labels must be integral.")
    if not set(labels.astype(int)).issubset(set(range(8))):
        raise ValueError("Class labels must be in 0..7.")
    if frame[list(GROUP_COLUMNS)].isna().any().any():
        raise ValueError("SNR and P1..P30 cannot be missing.")


def _robust_location_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    location = np.nanmedian(values, axis=0)
    scale = 1.4826 * np.nanmedian(np.abs(values - location), axis=0)
    fallback = np.nanstd(values, axis=0, ddof=1)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, fallback)
    scale = np.where(np.isfinite(scale) & (scale > 1e-8), scale, 1.0)
    return location.astype(np.float64), scale.astype(np.float64)


def fit_lifecycle_scaler(frame: pd.DataFrame, *, regime: FeatureRegime) -> LifecycleScaler:
    """Fit preprocessing using a declared training partition only."""
    validate_feature_contract(frame, regime=regime)
    trace = frame[list(TRACE_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    trace_location, trace_scale = _robust_location_scale(trace)
    context = frame[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    medians = np.nanmedian(context, axis=0)
    if not np.isfinite(medians).all():
        raise ValueError("Every context feature needs at least one finite training value.")
    filled = np.where(np.isfinite(context), context, medians)
    context_location, context_scale = _robust_location_scale(filled)
    active = {
        "full": (True, True, True),
        "trace_only": (True, False, False),
        "summary_only": (True, True, True),
    }[regime]
    return LifecycleScaler(
        regime=regime,
        trace_location=tuple(trace_location),
        trace_scale=tuple(trace_scale),
        context_median=tuple(medians),
        context_location=tuple(context_location),
        context_scale=tuple(context_scale),
        context_active=active,
    )


def transform_lifecycle(frame: pd.DataFrame, scaler: LifecycleScaler) -> LifecycleBatch:
    validate_feature_contract(frame, regime=scaler.regime)
    trace = frame[list(TRACE_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    if scaler.regime == "summary_only":
        trace = np.zeros_like(trace)
    else:
        trace = (trace - np.asarray(scaler.trace_location)) / np.asarray(scaler.trace_scale)

    raw_context = frame[list(CONTEXT_COLUMNS)].to_numpy(dtype=np.float64, copy=True)
    missing = ~np.isfinite(raw_context)
    median = np.asarray(scaler.context_median)
    context = np.where(missing, median, raw_context)
    context = (context - np.asarray(scaler.context_location)) / np.asarray(scaler.context_scale)
    active = np.asarray(scaler.context_active, dtype=bool)
    context[:, ~active] = 0.0
    missing[:, ~active] = True

    labels = frame["Class"].to_numpy(dtype=np.int64, copy=True)
    if "Position" in frame:
        position = pd.to_numeric(frame["Position"], errors="coerce").to_numpy(dtype=np.float32, copy=True)
    else:
        position = np.full(len(frame), np.nan, dtype=np.float32)
    group_ids = tuple(str(value) for value in frame["_input_group"]) if "_input_group" in frame else tuple(
        hashlib.sha256("|".join(format(float(value), ".17g") for value in row).encode()).hexdigest()
        for row in frame[list(GROUP_COLUMNS)].to_numpy(dtype=np.float64)
    )
    return LifecycleBatch(
        trace=torch.from_numpy(trace.astype(np.float32)),
        context=torch.from_numpy(context.astype(np.float32)),
        context_missing=torch.from_numpy(missing.astype(np.float32)),
        labels=torch.from_numpy(labels),
        position=torch.from_numpy(position),
        group_ids=group_ids,
    )


def fit_lifecycle_fold(
    frame: pd.DataFrame,
    *,
    holdout: tuple[int, int],
    seed: int,
    regime: FeatureRegime = "full",
) -> LifecycleTensorFold:
    validate_feature_contract(frame, regime=regime)
    grouped = attach_input_groups(frame)
    fold = build_event_openworld_fold(grouped, holdout=holdout, seed=seed)
    validate_fold_isolation(fold)
    scaler = fit_lifecycle_scaler(fold.train, regime=regime)
    batches = {name: transform_lifecycle(part, scaler) for name, part in fold.partitions().items()}
    return LifecycleTensorFold(fold, scaler, batches)


def split_known_calibration(
    validation: pd.DataFrame,
    *,
    seed: int,
    selector_fraction: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split known validation groups into selector-fit and threshold-calibration sets."""
    if "_input_group" not in validation:
        raise ValueError("Validation frame must contain exact input groups.")
    if not 0.2 <= selector_fraction <= 0.8:
        raise ValueError("selector_fraction must be in [0.2, 0.8].")
    left: set[str] = set()
    right: set[str] = set()
    for class_id, part in validation.groupby("Class", sort=True):
        groups = sorted(
            part["_input_group"].unique(),
            key=lambda value: hashlib.sha256(f"cal:{seed}:{class_id}:{value}".encode()).hexdigest(),
        )
        cut = min(max(1, int(len(groups) * selector_fraction)), len(groups) - 1)
        left.update(groups[:cut])
        right.update(groups[cut:])
    if left & right:
        raise AssertionError("Selector and threshold calibration groups overlap.")
    return (
        validation[validation["_input_group"].isin(left)].copy(),
        validation[validation["_input_group"].isin(right)].copy(),
    )


def deterministic_support_indices(
    frame: pd.DataFrame,
    *,
    class_ids: tuple[int, ...],
    shots: int,
    seed: int,
    draw: int,
    namespace: str = "lifecycle-support",
) -> np.ndarray:
    """Select group-distinct support rows without consulting query outcomes."""
    if "_input_group" not in frame:
        raise ValueError("Support pool must contain exact input groups.")
    if shots < 1:
        raise ValueError("shots must be positive.")
    chosen: list[int] = []
    for class_id in class_ids:
        part = frame[frame["Class"] == class_id]
        unique = part.drop_duplicates("_input_group")
        ranked = sorted(
            unique.index.tolist(),
            key=lambda index: hashlib.sha256(
                f"{namespace}:{seed}:{draw}:{shots}:{class_id}:{unique.at[index, '_input_group']}".encode()
            ).hexdigest(),
        )
        if len(ranked) < shots:
            raise ValueError(f"Class {class_id} provides {len(ranked)} groups, needs {shots}.")
        chosen.extend(ranked[:shots])
    return np.asarray(chosen, dtype=np.int64)


def data_audit(frame: pd.DataFrame, *, data_path: str | Path) -> dict[str, object]:
    validate_feature_contract(frame, regime="full")
    grouped = attach_input_groups(frame)
    counts = grouped.groupby("_input_group", sort=False).size()
    conflicts = grouped.groupby("_input_group", sort=False)["Class"].nunique()
    numeric = [*GROUP_COLUMNS, "loss", "Reflectance", "Position"]
    ranges = {
        column: {
            "minimum": float(pd.to_numeric(grouped[column], errors="coerce").min()),
            "maximum": float(pd.to_numeric(grouped[column], errors="coerce").max()),
        }
        for column in numeric
        if column in grouped
    }
    from .study_state import file_sha256

    return {
        "data_path": str(Path(data_path).resolve()),
        "sha256": file_sha256(data_path),
        "rows": len(grouped),
        "columns": len(frame.columns),
        "class_counts": {str(int(k)): int(v) for k, v in grouped["Class"].value_counts().sort_index().items()},
        "exact_input_groups": int(grouped["_input_group"].nunique()),
        "duplicate_groups": int((counts > 1).sum()),
        "duplicate_rows_beyond_first": int((counts - 1).clip(lower=0).sum()),
        "maximum_group_size": int(counts.max()),
        "conflicting_label_groups": int((conflicts > 1).sum()),
        "missing_values": {column: int(value) for column, value in frame.isna().sum().items() if value},
        "feature_ranges": ranges,
        "group_columns": list(GROUP_COLUMNS),
    }


def lifecycle_split_manifest(
    fold: EventOpenWorldFold,
    *,
    data_path: str | Path,
    regime: FeatureRegime,
) -> dict[str, object]:
    """Describe lifecycle partitions without inheriting frozen trace-only policy text."""
    result: dict[str, object] = {
        "schema_version": 1,
        "data_path": str(Path(data_path).resolve()),
        "holdout": list(fold.holdout),
        "seed": fold.seed,
        "feature_regime": regime,
        "inference_inputs": list(FEATURE_REGIMES[regime]),
        "group_columns": list(GROUP_COLUMNS),
        "forbidden_inputs": sorted(TARGET_COLUMNS),
        "partitions": {},
    }
    for name, part in fold.partitions().items():
        row_groups = part["_input_group"].astype(str)
        groups = sorted(row_groups.unique())
        result["partitions"][name] = {
            "rows": len(part),
            "groups": len(groups),
            "duplicate_rows_beyond_first": int(len(part) - len(groups)),
            "group_list_sha256": hashlib.sha256("\n".join(groups).encode()).hexdigest(),
            "class_counts": {
                str(int(key)): int(value)
                for key, value in part["Class"].value_counts().sort_index().items()
            },
        }
    return result
