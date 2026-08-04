from __future__ import annotations

from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from .zero_shot_data import FORBIDDEN_FEATURES, INPUT_COLUMNS, validate_zero_shot_frame


@dataclass(frozen=True)
class EventOpenWorldFold:
    holdout: tuple[int, int]
    seed: int
    train: pd.DataFrame
    validation: pd.DataFrame
    seen_test: pd.DataFrame
    reference_pool: pd.DataFrame
    adaptation_pool: pd.DataFrame
    query: pd.DataFrame

    def partitions(self) -> dict[str, pd.DataFrame]:
        return {
            "train": self.train,
            "validation": self.validation,
            "seen_test": self.seen_test,
            "reference_pool": self.reference_pool,
            "adaptation_pool": self.adaptation_pool,
            "query": self.query,
        }


@dataclass(frozen=True)
class TensorFold:
    split: EventOpenWorldFold
    scaler: StandardScaler
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]]


def canonical_input_bytes(values: Iterable[float]) -> bytes:
    return "|".join(format(float(value), ".17g") for value in values).encode("utf-8")


def attach_input_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach the frozen SHA-256 group ID and reject conflicting labels."""
    validate_zero_shot_frame(frame)
    work = frame.copy()
    work["Class"] = work["Class"].astype(int)
    if "_input_group" not in work:
        array = work[INPUT_COLUMNS].to_numpy(dtype=np.float64, copy=False)
        work["_input_group"] = [hashlib.sha256(canonical_input_bytes(row)).hexdigest() for row in array]
    conflicts = work.groupby("_input_group", sort=False)["Class"].nunique()
    if (conflicts > 1).any():
        bad = conflicts[conflicts > 1].index.tolist()[:5]
        raise ValueError(f"Conflicting-label exact input groups: {bad}")
    return work


def deduplicate_groups(frame: pd.DataFrame) -> pd.DataFrame:
    if "_input_group" not in frame:
        raise ValueError("Input groups must be attached before de-duplication.")
    return frame.drop_duplicates("_input_group", keep="first").copy()


def _rank_groups(class_frame: pd.DataFrame, *, namespace: str, seed: int) -> list[str]:
    groups = class_frame["_input_group"].unique().tolist()
    return sorted(groups, key=lambda group: hashlib.sha256(f"{namespace}:{seed}:{group}".encode()).hexdigest())


def _slices(groups: list[str], fractions: tuple[float, ...]) -> list[set[str]]:
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError("Partition fractions must sum to one.")
    count = len(groups)
    cuts = [0]
    cumulative = 0.0
    for fraction in fractions[:-1]:
        cumulative += fraction
        cuts.append(int(np.floor(count * cumulative)))
    cuts.append(count)
    result = [set(groups[cuts[i]:cuts[i + 1]]) for i in range(len(fractions))]
    if any(not part for part in result):
        raise ValueError(f"A requested group partition is empty (n={count}, fractions={fractions}).")
    return result


def build_event_openworld_fold(frame: pd.DataFrame, *, holdout: tuple[int, int], seed: int) -> EventOpenWorldFold:
    holdout = tuple(sorted(int(value) for value in holdout))
    if len(set(holdout)) != 2 or any(value not in range(1, 8) for value in holdout):
        raise ValueError("holdout must be two distinct fault IDs in 1..7")
    work = attach_input_groups(frame) if "_input_group" not in frame else frame.copy()
    work = deduplicate_groups(work)
    seen_ids = sorted(set(range(8)) - set(holdout))
    seen_sets = [set(), set(), set()]
    for class_id in seen_ids:
        ranked = _rank_groups(work[work["Class"] == class_id], namespace=f"seen:{class_id}", seed=seed)
        for target, values in zip(seen_sets, _slices(ranked, (0.70, 0.15, 0.15)), strict=True):
            target.update(values)
    unseen_sets = [set(), set(), set()]
    for class_id in holdout:
        ranked = _rank_groups(work[work["Class"] == class_id], namespace=f"unseen:{class_id}", seed=seed)
        for target, values in zip(unseen_sets, _slices(ranked, (0.05, 0.15, 0.80)), strict=True):
            target.update(values)
    partitions = [
        work[work["_input_group"].isin(seen_sets[0])].copy(),
        work[work["_input_group"].isin(seen_sets[1])].copy(),
        work[work["_input_group"].isin(seen_sets[2])].copy(),
        work[work["_input_group"].isin(unseen_sets[0])].copy(),
        work[work["_input_group"].isin(unseen_sets[1])].copy(),
        work[work["_input_group"].isin(unseen_sets[2])].copy(),
    ]
    fold = EventOpenWorldFold(holdout, seed, *partitions)
    validate_fold_isolation(fold)
    return fold


def validate_fold_isolation(fold: EventOpenWorldFold) -> None:
    parts = fold.partitions()
    group_sets = {name: set(part["_input_group"]) for name, part in parts.items()}
    names = list(parts)
    for left, left_name in enumerate(names):
        for right_name in names[left + 1:]:
            overlap = group_sets[left_name] & group_sets[right_name]
            if overlap:
                raise AssertionError(f"Exact-input leakage: {left_name}/{right_name}: {len(overlap)} groups")
    for name in ("train", "validation", "seen_test"):
        if set(parts[name]["Class"]) & set(fold.holdout):
            raise AssertionError(f"Outer-held-out class entered {name}.")
    for name in ("reference_pool", "adaptation_pool", "query"):
        if not set(parts[name]["Class"]).issubset(set(fold.holdout)):
            raise AssertionError(f"Seen class entered held-out {name}.")


def fit_tensor_fold(fold: EventOpenWorldFold) -> TensorFold:
    scaler = StandardScaler().fit(fold.train[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True))
    tensors: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    for name, part in fold.partitions().items():
        x = scaler.transform(part[INPUT_COLUMNS].to_numpy(dtype=np.float32, copy=True)).astype(np.float32)
        y = part["Class"].to_numpy(dtype=np.int64, copy=True)
        tensors[name] = (torch.from_numpy(x), torch.from_numpy(y))
    return TensorFold(fold, scaler, tensors)


def deterministic_group_sample(
    frame: pd.DataFrame,
    *,
    class_ids: tuple[int, ...],
    per_class: int,
    seed: int,
    draw: int,
    namespace: str,
) -> np.ndarray:
    chosen: list[int] = []
    for class_id in class_ids:
        candidates = frame.index[frame["Class"] == class_id].tolist()
        if len(candidates) < per_class:
            raise ValueError(f"Class {class_id} has {len(candidates)} groups, needs {per_class}.")
        ranked = sorted(candidates, key=lambda idx: hashlib.sha256(
            f"{namespace}:{seed}:{draw}:{class_id}:{frame.at[idx, '_input_group']}".encode()
        ).hexdigest())
        chosen.extend(ranked[:per_class])
    return np.asarray(chosen, dtype=np.int64)


def split_manifest(fold: EventOpenWorldFold, *, data_path: str | Path) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": 1,
        "data_path": str(Path(data_path)),
        "holdout": list(fold.holdout),
        "seed": fold.seed,
        "features": INPUT_COLUMNS,
        "forbidden": sorted(FORBIDDEN_FEATURES),
        "partitions": {},
    }
    for name, part in fold.partitions().items():
        groups = sorted(part["_input_group"].tolist())
        digest = hashlib.sha256("\n".join(groups).encode()).hexdigest()
        result["partitions"][name] = {
            "groups": len(groups),
            "group_list_sha256": digest,
            "class_counts": {str(int(k)): int(v) for k, v in part["Class"].value_counts().sort_index().items()},
        }
    return result


def write_exact_group_manifest(fold: EventOpenWorldFold, path: str | Path) -> Path:
    """Atomically persist every exact group ID as raw 32-byte SHA-256 values."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        name: np.asarray([bytes.fromhex(value) for value in part["_input_group"]], dtype="V32")
        for name, part in fold.partitions().items()
    }
    if target.exists():
        try:
            with np.load(target) as existing:
                if all(name in existing and np.array_equal(existing[name], expected) for name, expected in arrays.items()):
                    return target
        except (OSError, ValueError):
            pass
    fd, temporary = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".npz", dir=target.parent)
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, target)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return target
