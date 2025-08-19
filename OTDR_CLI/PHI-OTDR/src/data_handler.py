# data_handler.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any

import argparse
import numpy as np
import scipy.io as scio
import torch
from torch.utils.data import Dataset, DataLoader

CLASS_NAMES = ["background", "digging", "knocking", "watering", "shaking", "walking"]

# ----------------------------- path helpers ----------------------------- #

def _normalize_rel(rel: str) -> Path:
    """
    Normalize label paths to be truly relative and portable.
    Handles leading slashes, backslashes, accidental 'train/' or 'test/' prefixes.
    """
    s = rel.strip().strip('"').strip("'").replace("\\", "/")
    s = s.lstrip("/")  # kill any leading slash
    for prefix in ("train/", "test/", "./train/", "./test/"):
        if s.lower().startswith(prefix):
            s = s[len(prefix):]
            break
    p = Path(s)
    if p.is_absolute() or p.anchor or getattr(p, "drive", ""):
        p = Path(*[part for part in p.parts if part not in (p.anchor,)])
    return p

def read_label_file(path: Path) -> List[Tuple[Path, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Label file not found: {path}")
    pairs: List[Tuple[Path, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln_no, ln in enumerate(f, start=1):
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            parts = ln.split()
            if len(parts) < 2:
                raise ValueError(f"Bad line in {path} (line {ln_no}): {ln!r}")
            rel = _normalize_rel(parts[0])
            lab = int(parts[1])
            pairs.append((rel, lab))
    if not pairs:
        raise ValueError(f"No valid entries parsed from: {path}")
    return pairs

def _load_mat(path: Path) -> np.ndarray:
    """Robust .mat loader: returns (T,C) float32, raises if unreadable."""
    raw = scio.loadmat(path.as_posix())
    arr = raw.get("data", None)
    if arr is None:
        keys = [k for k in raw.keys() if not k.startswith("__")]
        if not keys:
            raise KeyError(f"No array key found in {path}")
        arr = raw[keys[0]]
    arr = np.asarray(arr)
    if arr.ndim != 2:
        arr = arr.reshape(arr.shape[0], -1)
    return arr.astype(np.float32, copy=False)

def minmax01(a: np.ndarray) -> np.ndarray:
    vmin = float(np.min(a))
    vmax = float(np.max(a))
    if vmax <= vmin:
        return np.zeros_like(a, dtype=np.float32)
    return (a - vmin) / (vmax - vmin)

# --------------------------- Dataset + DLs ------------------------------- #

class PhiOTDRDataset(Dataset):
    """
    Returns {'data': FloatTensor[T,C], 'label': LongTensor[]}
    Skips:
      - missing files at init (counted in skipped_missing)
      - unreadable/broken .mat at __getitem__ (returns None; counted in skipped_broken)
    """
    def __init__(self, root_dir: Path, label_list: Path, drop_missing: bool = True):
        self.root_dir = Path(root_dir)
        entries = read_label_file(label_list)
        all_samples: List[Tuple[Path, int]] = [(self.root_dir / rel, lab) for rel, lab in entries]

        # Filter out missing files now (so the DataLoader won't even try them)
        if drop_missing:
            present = [(p, lab) for (p, lab) in all_samples if p.is_file()]
            self.skipped_missing = len(all_samples) - len(present)
            self.samples = present
        else:
            self.skipped_missing = 0
            self.samples = all_samples

        self.skipped_broken = 0  # incremented in __getitem__ when a .mat fails to load/parse

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any] | None:
        mat_path, lab = self.samples[idx]
        # mat may exist but be broken/truncated -> catch and skip
        try:
            arr = _load_mat(mat_path)   # (T, C)
            arr = minmax01(arr)         # [0,1]
        except Exception:
            self.skipped_broken += 1
            return None  # collate_fn will filter this out

        data = torch.from_numpy(arr)          # float32 [T,C]
        label = torch.tensor(lab, dtype=torch.long)
        return {"data": data, "label": label}

@dataclass
class LoaderConfig:
    batch_size: int = 64
    num_workers: int = 0      # 0 for Windows stability and shared counters
    pin_memory: bool = False
    shuffle_train: bool = True

def _collate_filter(batch: List[Dict[str, Any] | None]) -> Dict[str, torch.Tensor] | None:
    """Filter out None samples; return None for empty batches."""
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    data = torch.stack([b["data"] for b in batch], dim=0)   # (B,T,C)
    labels = torch.stack([b["label"] for b in batch], dim=0)  # (B,)
    return {"data": data, "label": labels}

def make_dataloaders(
    train_root: Path, train_list: Path,
    test_root: Path, test_list: Path,
    cfg: LoaderConfig = LoaderConfig(),
) -> tuple[DataLoader, DataLoader]:
    ds_train = PhiOTDRDataset(train_root, train_list, drop_missing=True)
    ds_test  = PhiOTDRDataset(test_root,  test_list,  drop_missing=True)
    train_loader = DataLoader(
        ds_train, batch_size=cfg.batch_size, shuffle=cfg.shuffle_train,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=_collate_filter
    )
    test_loader = DataLoader(
        ds_test, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=cfg.pin_memory,
        collate_fn=_collate_filter
    )
    return train_loader, test_loader

# ----------------------------- Viz helpers ------------------------------- #

def save_sample_images(dataset: PhiOTDRDataset, out_dir: Path, num: int = 6) -> None:
    """Save heatmap images for quick visual inspection; auto-skips bad items."""
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1234)
    tried, saved = 0, 0
    while saved < min(num, len(dataset)) and tried < len(dataset) * 3:
        idx = int(rng.integers(0, len(dataset)))
        tried += 1
        sample = dataset[idx]
        if sample is None:  # broken -> skip
            continue
        arr = sample["data"].numpy()  # (T,C)
        lab = int(sample["label"].item())
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(arr.T, aspect="auto", origin="lower")
        ax.set_title(f"Sample #{idx} – class={lab} ({CLASS_NAMES[lab]})")
        ax.set_ylabel("Channel")
        ax.set_xlabel("Time index")
        fig.colorbar(im, ax=ax, shrink=0.8, label="normalized amplitude")
        fig.tight_layout()
        fig.savefig(out_dir / f"sample_{idx}_class{lab}.png", dpi=150)
        plt.close(fig)
        saved += 1

# ------------------------ Label (re)builder CLI -------------------------- #

_FOLDER_TO_LABEL = {
    "background": 0, "bg": 0, "noise": 0,
    "dig": 1, "digging": 1,
    "knock": 2, "knocking": 2,
    "water": 3, "watering": 3,
    "shake": 4, "shaking": 4,
    "walk": 5, "walking": 5,
}

def _infer_label_from_parent(p: Path) -> int:
    name = p.parent.name.lower()
    for key, lab in _FOLDER_TO_LABEL.items():
        if key in name:
            return lab
    name2 = p.parent.parent.name.lower() if p.parent.parent else ""
    for key, lab in _FOLDER_TO_LABEL.items():
        if key in name2:
            return lab
    raise ValueError(f"Cannot infer label from folder name for: {p}")

def rebuild_labels(root: Path, out_file: Path) -> None:
    mats = sorted(root.rglob("*.mat"))
    if not mats:
        raise FileNotFoundError(f"No .mat files under {root}")
    lines = []
    for m in mats:
        lab = _infer_label_from_parent(m)
        rel = m.relative_to(root).as_posix()
        lines.append(f"{rel} {lab}\n")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with out_file.open("w", encoding="utf-8") as f:
        f.writelines(lines)
    print(f"Rebuilt label file with {len(lines)} entries -> {out_file}")

def validate_labels(root: Path, label_file: Path, limit: int = 10) -> None:
    pairs = read_label_file(label_file)
    missing = []
    for rel, _ in pairs:
        p = root / rel
        if not p.is_file():
            missing.append(p)
            if len(missing) >= limit:
                break
    if missing:
        print(f"Found {len(missing)} missing paths (showing {len(missing)}):")
        for p in missing:
            print("  MISSING:", p)
    else:
        print("All labeled paths exist.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Phi-OTDR label tools")
    ap.add_argument("--rebuild-labels", action="store_true")
    ap.add_argument("--validate-labels", action="store_true")
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--labels", type=Path, default=None)
    args = ap.parse_args()
    if args.rebuild_labels:
        if not args.out:
            raise SystemExit("--out is required with --rebuild-labels")
        rebuild_labels(args.root, args.out)
    elif args.validate_labels:
        if not args.labels:
            raise SystemExit("--labels is required with --validate-labels")
        validate_labels(args.root, args.labels)
