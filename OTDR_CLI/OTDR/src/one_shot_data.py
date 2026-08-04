from __future__ import annotations

from dataclasses import dataclass
import hashlib
from itertools import combinations

import numpy as np
import pandas as pd

from .zero_shot_data import OuterFold


@dataclass(frozen=True)
class PairIndices:
    left: np.ndarray
    right: np.ndarray
    targets: np.ndarray


@dataclass(frozen=True)
class OneShotSplit:
    outer: OuterFold
    support_pool: pd.DataFrame
    query: pd.DataFrame


def _rank(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def build_one_shot_split(
    outer: OuterFold,
    *,
    support_fraction: float = 0.2,
    seed: int = 42,
) -> OneShotSplit:
    """Reserve held-out groups for enrollment support before evaluating queries."""

    if not 0.0 < support_fraction < 1.0:
        raise ValueError("support_fraction must lie strictly between zero and one.")
    support_groups: set[str] = set()
    for class_id in outer.holdout:
        groups = sorted(outer.unseen_test.loc[outer.unseen_test["Class"] == class_id, "_input_group"].unique())
        if len(groups) < 2:
            raise ValueError(f"Held-out class {class_id} needs at least two unique input groups.")
        groups = sorted(groups, key=lambda value: _rank(value, seed + class_id))
        count = min(len(groups) - 1, max(1, int(round(len(groups) * support_fraction))))
        support_groups.update(groups[:count])
    support = outer.unseen_test[outer.unseen_test["_input_group"].isin(support_groups)].copy()
    query = outer.unseen_test[~outer.unseen_test["_input_group"].isin(support_groups)].copy()
    return OneShotSplit(outer=outer, support_pool=support, query=query)


def sample_class_references(
    frame: pd.DataFrame,
    *,
    references_per_class: int,
    seed: int,
    class_ids: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    if references_per_class < 1:
        raise ValueError("references_per_class must be positive.")
    selected: list[pd.Series] = []
    active_classes = sorted(frame["Class"].astype(int).unique()) if class_ids is None else list(class_ids)
    for class_id in active_classes:
        class_frame = frame[frame["Class"].astype(int) == class_id]
        if class_frame.empty:
            raise ValueError(f"No references are available for class {class_id}.")
        ranked_indices = sorted(class_frame.index.tolist(), key=lambda value: _rank(value, seed + class_id))
        for index in ranked_indices[:references_per_class]:
            selected.append(class_frame.loc[index])
    if not selected:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(selected).reset_index(names="_source_index")


def build_balanced_pair_indices(
    labels: np.ndarray,
    *,
    pair_count: int,
    seed: int,
) -> PairIndices:
    """Build deterministic 50/50 same/different pairs without self-pairs."""

    labels = np.asarray(labels, dtype=np.int64)
    if pair_count < 2 or pair_count % 2:
        raise ValueError("pair_count must be a positive even integer.")
    class_to_indices = {
        int(class_id): np.flatnonzero(labels == class_id)
        for class_id in np.unique(labels)
    }
    positive_classes = [class_id for class_id, indices in class_to_indices.items() if len(indices) >= 2]
    negative_pairs = list(combinations(sorted(class_to_indices), 2))
    if not positive_classes or not negative_pairs:
        raise ValueError("Balanced pairs need at least two classes and two rows in a positive class.")
    rng = np.random.default_rng(seed)
    half = pair_count // 2
    left: list[int] = []
    right: list[int] = []
    targets: list[float] = []
    for position in range(half):
        class_id = positive_classes[position % len(positive_classes)]
        pair = rng.choice(class_to_indices[class_id], size=2, replace=False)
        left.append(int(pair[0]))
        right.append(int(pair[1]))
        targets.append(1.0)
    for position in range(half):
        left_class, right_class = negative_pairs[position % len(negative_pairs)]
        left.append(int(rng.choice(class_to_indices[left_class])))
        right.append(int(rng.choice(class_to_indices[right_class])))
        targets.append(0.0)
    order = rng.permutation(pair_count)
    return PairIndices(
        left=np.asarray(left, dtype=np.int64)[order],
        right=np.asarray(right, dtype=np.int64)[order],
        targets=np.asarray(targets, dtype=np.float32)[order],
    )
