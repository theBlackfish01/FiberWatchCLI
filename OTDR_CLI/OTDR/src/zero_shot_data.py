from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import pandas as pd


INPUT_COLUMNS = ["SNR", *[f"P{i}" for i in range(1, 31)]]
FORBIDDEN_FEATURES = {"Class", "Position", "loss", "Loss", "Reflectance"}


@dataclass(frozen=True)
class FaultPrototype:
    class_id: int
    name: str
    descriptions: tuple[str, ...]


@dataclass(frozen=True)
class OuterFold:
    holdout: tuple[int, int]
    train: pd.DataFrame
    validation: pd.DataFrame
    seen_test: pd.DataFrame
    unseen_test: pd.DataFrame
    feature_columns: tuple[str, ...] = tuple(INPUT_COLUMNS)


def load_fault_prototypes(path: str | Path) -> list[FaultPrototype]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Prototype schema_version must be 1.")
    classes = payload.get("classes", [])
    if [item.get("id") for item in classes] != list(range(8)):
        raise ValueError("Prototype classes must be ordered unique IDs 0 through 7.")
    result: list[FaultPrototype] = []
    for item in classes:
        descriptions = tuple(text.strip() for text in item.get("descriptions", []))
        if len(descriptions) != 5 or any(not text for text in descriptions):
            raise ValueError(f"Class {item['id']} must define exactly five non-empty descriptions.")
        if len(set(descriptions)) != 5:
            raise ValueError(f"Class {item['id']} contains duplicate descriptions.")
        result.append(FaultPrototype(int(item["id"]), str(item["name"]), descriptions))
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_group(row: pd.Series) -> str:
    canonical = "|".join(format(float(row[column]), ".17g") for column in INPUT_COLUMNS)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_zero_shot_frame(frame: pd.DataFrame) -> None:
    missing = [column for column in ["Class", *INPUT_COLUMNS] if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    labels = frame["Class"]
    if labels.isna().any() or not labels.map(lambda value: float(value).is_integer()).all():
        raise ValueError("Class labels must be integral values from 0 through 7.")
    if not set(labels.astype(int)).issubset(set(range(8))):
        raise ValueError("Class labels must be in the range 0 through 7.")
    if frame[INPUT_COLUMNS].isna().any().any():
        raise ValueError("Zero-shot input columns cannot contain missing values.")


def _partition_seen_groups(class_frame: pd.DataFrame, seed: int) -> tuple[set[str], set[str], set[str]]:
    groups = sorted(class_frame["_input_group"].unique())
    ranked = sorted(groups, key=lambda group: hashlib.sha256(f"{seed}:{group}".encode()).hexdigest())
    count = len(ranked)
    train_end = max(1, int(count * 0.8))
    val_end = max(train_end + 1, int(count * 0.9)) if count >= 3 else train_end
    val_end = min(val_end, count)
    return set(ranked[:train_end]), set(ranked[train_end:val_end]), set(ranked[val_end:])


def build_outer_fold(frame: pd.DataFrame, *, holdout: tuple[int, int], seed: int = 42) -> OuterFold:
    validate_zero_shot_frame(frame)
    holdout = tuple(sorted(int(item) for item in holdout))
    if len(set(holdout)) != 2 or any(item not in range(1, 8) for item in holdout):
        raise ValueError("holdout must contain two distinct fault class IDs from 1 through 7.")
    work = frame.copy()
    work["Class"] = work["Class"].astype(int)
    work["_input_group"] = work.apply(_input_group, axis=1)
    conflicting = work.groupby("_input_group")["Class"].nunique()
    if (conflicting > 1).any():
        raise ValueError(
            "Identical zero-shot inputs have conflicting class labels; "
            "these groups cannot be split without leakage."
        )
    unseen = work[work["Class"].isin(holdout)].copy()
    seen_mask = ~work["Class"].isin(holdout)
    train_groups: set[str] = set()
    validation_groups: set[str] = set()
    test_groups: set[str] = set()
    for class_id in sorted(set(range(8)) - set(holdout)):
        class_frame = work[work["Class"] == class_id]
        tr, va, te = _partition_seen_groups(class_frame, seed + class_id)
        train_groups.update(tr)
        validation_groups.update(va)
        test_groups.update(te)
    train = work[seen_mask & work["_input_group"].isin(train_groups)].copy()
    validation = work[seen_mask & work["_input_group"].isin(validation_groups)].copy()
    seen_test = work[seen_mask & work["_input_group"].isin(test_groups)].copy()
    return OuterFold(holdout, train, validation, seen_test, unseen)
