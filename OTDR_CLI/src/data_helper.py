from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple
import re

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "SplitTensors",
    "load_raw_dataframe",
    "make_splits",
    "fit_scaler",
    "tensorise_splits",
    "make_dataloaders",
],

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SplitTensors:
    """Container holding the tensors of a single dataset split."""

    X: torch.Tensor
    y_class: torch.Tensor
    y_pos: torch.Tensor

    def __iter__(self):
        return iter((self.X, self.y_class, self.y_pos))


# ---------------------------------------------------------------------------
# Core I/O pipeline                         |
# ---------------------------------------------------------------------------


def _measurement_columns(df: pd.DataFrame) -> list[str]:
    """Return measurement columns: P\d+ and SNR."""

    pattern = re.compile(r"P\d+")
    cols = [c for c in df.columns if pattern.fullmatch(c)]
    return cols + ["SNR"]


def load_raw_dataframe(path: str | Path) -> pd.DataFrame:
    """Load CSV or Parquet file, dropping any unnamed index columns."""

    df = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_parquet(path)
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    return df


def make_splits(
    df: pd.DataFrame,
    *,
    test_size: float = 0.20,
    val_size: float = 0.20,
    label_col: str = "Class",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified **train / val / test** splits on *label_col*."""

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df[label_col],
    )

    val_rel = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_df,
        test_size=val_rel,
        random_state=random_state,
        stratify=train_df[label_col],
    )
    return train_df, val_df, test_df


def fit_scaler(train_X: np.ndarray) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_X)
    return scaler


# ---------------------------------------------------------------------------
# Torch conversion helpers                                                   |
# ---------------------------------------------------------------------------


def _prepare_arrays(
    df: pd.DataFrame,
    measurement_cols: list[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[measurement_cols].values.astype(np.float32)
    y_class = df["Class"].astype(np.int64).values
    y_pos = df["Position"].astype(np.float32).values.reshape(-1, 1)
    return X, y_class, y_pos


def tensorise_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    scaler: StandardScaler,
) -> Dict[str, SplitTensors]:
    measurement_cols = _measurement_columns(train_df)

    def _process(df):
        X, y_cls, y_pos = _prepare_arrays(df, measurement_cols)
        X = scaler.transform(X)
        return SplitTensors(
            X=torch.tensor(X, dtype=torch.float32),
            y_class=torch.tensor(y_cls, dtype=torch.long),
            y_pos=torch.tensor(y_pos, dtype=torch.float32),
        )

    return {"train": _process(train_df), "val": _process(val_df), "test": _process(test_df)}


def make_dataloaders(
    splits: Dict[str, SplitTensors],
    batch_size: int = 64,
    num_workers: int = 0,
    *,
    shuffle_train: bool = True,
) -> Dict[str, DataLoader]:
    """Build DataLoaders for each split."""

    def _dl(split: SplitTensors, shuffle: bool):
        ds = TensorDataset(split.X, split.y_class, split.y_pos)
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
        )

    return {
        "train": _dl(splits["train"], shuffle_train),
        "val": _dl(splits["val"], False),
        "test": _dl(splits["test"], False),
    }
