from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA

from .zero_shot_data import load_fault_prototypes


def load_physics_prototypes(path: str | Path) -> tuple[list[str], list[str], torch.Tensor]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema_version") != 2:
        raise ValueError("Physics prototype schema_version must be 2.")
    attributes = payload.get("attribute_names", [])
    classes = payload.get("classes", [])
    if len(attributes) < 8 or len(set(attributes)) != len(attributes):
        raise ValueError("Physics prototypes require at least eight unique attributes.")
    if [item.get("id") for item in classes] != list(range(8)):
        raise ValueError("Physics classes must be ordered IDs 0 through 7.")
    matrix = np.asarray([item.get("attributes", []) for item in classes], dtype=np.float32)
    if matrix.shape != (8, len(attributes)) or not np.isfinite(matrix).all() or (matrix < 0).any() or (matrix > 1).any():
        raise ValueError("Physics attributes must be a finite 8 x attribute_count matrix in [0, 1].")
    if np.unique(matrix, axis=0).shape[0] != 8:
        raise ValueError("Every class needs a distinct physics prototype.")
    return attributes, [str(item["name"]) for item in classes], torch.from_numpy(matrix)


def text_prototypes(description_path: str | Path, *, model_name: str, device: torch.device) -> torch.Tensor:
    """Create compact, deterministic text prototypes using only a locally cached model."""
    from sentence_transformers import SentenceTransformer

    specifications = load_fault_prototypes(description_path)
    model = SentenceTransformer(model_name, device=str(device), local_files_only=True)
    rows = []
    for item in specifications:
        encoded = model.encode(list(item.descriptions), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False)
        rows.append(encoded.mean(axis=0))
    full = np.asarray(rows, dtype=np.float32)
    compact = PCA(n_components=7, svd_solver="full").fit_transform(full)
    minimum = compact.min(axis=0, keepdims=True)
    scale = np.maximum(compact.max(axis=0, keepdims=True) - minimum, 1e-8)
    return torch.from_numpy(((compact - minimum) / scale).astype(np.float32))


def semantic_prototypes(*, mode: str, physics_path: str | Path, description_path: str | Path,
                        text_model: str, device: torch.device, cache_dir: str | Path | None = None) -> tuple[list[str], list[str], torch.Tensor]:
    names, classes, physics = load_physics_prototypes(physics_path)
    if mode == "physics":
        return names, classes, physics
    cache = Path(cache_dir) / "text_prototypes.pt" if cache_dir else None
    if cache is not None and cache.exists():
        text = torch.load(cache, map_location="cpu", weights_only=True)
    else:
        text = text_prototypes(description_path, model_name=text_model, device=device)
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(text, cache)
    text_names = [f"text_pc_{index + 1}" for index in range(text.shape[1])]
    if mode == "text":
        return text_names, classes, text
    if mode == "combined":
        return names + text_names, classes, torch.cat([physics, text], dim=1)
    raise ValueError(f"Unknown semantic prototype mode: {mode}")
