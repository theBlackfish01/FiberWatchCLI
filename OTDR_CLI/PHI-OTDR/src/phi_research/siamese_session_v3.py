"""CUDA supervised contrastive/Siamese session embeddings for Phi-OTDR v3."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score, pairwise_distances
from sklearn.preprocessing import StandardScaler
from torch import nn

from .data_contract import CLASS_NAMES, canonical_json_hash


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_cuda() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for Phi-OTDR v3 Siamese training and inference")
    return {
        "torch_version": torch.__version__,
        "cuda_build": torch.version.cuda,
        "device": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "amp": "float16 autocast",
    }


class SiameseSessionEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.head = nn.Linear(embedding_dim, len(CLASS_NAMES))

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = nn.functional.normalize(self.encoder(values), dim=1)
        return embedding, self.head(embedding)


def supervised_pair_loss(embedding: torch.Tensor, labels: torch.Tensor, margin: float) -> torch.Tensor:
    """Siamese cosine loss over distinct session pairs in a batch."""
    similarity = embedding @ embedding.T
    identity = torch.eye(len(labels), dtype=torch.bool, device=labels.device)
    positive = (labels[:, None] == labels[None, :]) & ~identity
    negative = (labels[:, None] != labels[None, :]) & ~identity
    positive_loss = (1.0 - similarity[positive]).mean() if torch.any(positive) else similarity.sum() * 0.0
    negative_loss = torch.relu(similarity[negative] - margin).square().mean() if torch.any(negative) else similarity.sum() * 0.0
    return positive_loss + negative_loss


@torch.no_grad()
def _embed(model: nn.Module, values: np.ndarray, device: torch.device) -> tuple[np.ndarray, float]:
    model.eval()
    started = time.perf_counter()
    rows = []
    for start in range(0, len(values), 128):
        batch = torch.from_numpy(values[start : start + 128]).to(device)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            embedding, _ = model(batch)
        rows.append(embedding.float().cpu().numpy())
    torch.cuda.synchronize()
    return np.concatenate(rows), time.perf_counter() - started


def _prototype_macro_f1(
    train_embedding: np.ndarray,
    train_labels: np.ndarray,
    validation_embedding: np.ndarray,
    validation_labels: np.ndarray,
) -> float:
    class_ids = sorted(set(train_labels.tolist()))
    prototypes = np.stack([np.mean(train_embedding[train_labels == class_id], axis=0) for class_id in class_ids])
    prediction = np.asarray(class_ids)[np.argmin(pairwise_distances(validation_embedding, prototypes), axis=1)]
    return float(f1_score(validation_labels, prediction, average="macro", zero_division=0))


def run(
    *,
    session_aggregate_path: Path,
    manifests: Sequence[Path],
    config_path: Path,
    config_hash_path: Path,
    output_dir: Path,
    smoke: bool = False,
) -> dict[str, object]:
    cuda = _assert_cuda()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = config_hash_path.read_text(encoding="utf-8").split()[0]
    if canonical_json_hash(config) != expected_hash:
        raise ValueError("Enrollment v3 config hash mismatch")
    with np.load(session_aggregate_path, allow_pickle=False) as source:
        features = source["features"].astype(np.float32)
        sessions = source["sessions"].astype(str)
        labels = source["labels"].astype(np.int64)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0")
    siamese = config["siamese"]
    seeds = siamese["seeds"][:1] if smoke else siamese["seeds"]
    holdouts = range(1) if smoke else range(len(CLASS_NAMES))
    max_epochs = 3 if smoke else int(siamese["max_epochs"])
    patience = 2 if smoke else int(siamese["patience"])
    results = []
    started_all = time.perf_counter()
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
        partitions = np.asarray([session_rows[session]["partition"] for session in sessions])
        direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
        for holdout in holdouts:
            train = (partitions == "source_train") & (labels != holdout)
            validation = (partitions == "source_validation") & (labels != holdout)
            scaler = StandardScaler().fit(features[train])
            normalized = scaler.transform(features).astype(np.float32)
            train_x = torch.from_numpy(normalized[train]).to(device)
            train_y = torch.from_numpy(labels[train]).to(device)
            class_counts = np.bincount(labels[train], minlength=len(CLASS_NAMES)).astype(np.float32)
            weights = class_counts.sum() / np.maximum(class_counts * max(len(set(labels[train].tolist())), 1), 1.0)
            class_weights = torch.as_tensor(weights, device=device)
            for seed in seeds:
                torch.manual_seed(int(seed))
                torch.cuda.manual_seed_all(int(seed))
                model = SiameseSessionEncoder(
                    normalized.shape[1],
                    int(siamese["hidden_dim"]),
                    int(siamese["embedding_dim"]),
                ).to(device)
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=float(siamese["learning_rate"]),
                    weight_decay=float(siamese["weight_decay"]),
                )
                scaler_amp = torch.amp.GradScaler("cuda", enabled=True)
                classification_loss = nn.CrossEntropyLoss(weight=class_weights)
                best_score = -np.inf
                best_state = None
                best_epoch = -1
                history = []
                stale = 0
                torch.cuda.reset_peak_memory_stats(device)
                started = time.perf_counter()
                for epoch in range(max_epochs):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast(device_type="cuda", dtype=torch.float16):
                        embedding, logits = model(train_x)
                        cls = classification_loss(logits, train_y)
                        pair = supervised_pair_loss(
                            embedding,
                            train_y,
                            float(siamese["contrastive_margin_cosine"]),
                        )
                        loss = (
                            float(siamese["classification_loss_weight"]) * cls
                            + float(siamese["pair_loss_weight"]) * pair
                        )
                    scaler_amp.scale(loss).backward()
                    scaler_amp.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler_amp.step(optimizer)
                    scaler_amp.update()
                    train_embedding, _ = _embed(model, normalized[train], device)
                    validation_embedding, _ = _embed(model, normalized[validation], device)
                    macro = _prototype_macro_f1(
                        train_embedding,
                        labels[train],
                        validation_embedding,
                        labels[validation],
                    )
                    history.append(
                        {
                            "epoch": epoch + 1,
                            "loss": float(loss.detach().cpu()),
                            "classification_loss": float(cls.detach().cpu()),
                            "pair_loss": float(pair.detach().cpu()),
                            "source_validation_prototype_macro_f1": macro,
                        }
                    )
                    if macro > best_score + 1e-12:
                        best_score = macro
                        best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
                        best_epoch = epoch + 1
                        stale = 0
                    else:
                        stale += 1
                    if stale >= patience:
                        break
                assert best_state is not None
                model.load_state_dict(best_state)
                all_embedding, inference_seconds = _embed(model, normalized, device)
                run_name = f"{direction}__holdout_{holdout}_{CLASS_NAMES[holdout]}__seed{seed}"
                run_dir = output_dir / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                embedding_path = run_dir / "session_embeddings.npz"
                np.savez_compressed(
                    embedding_path,
                    embeddings=all_embedding.astype(np.float32),
                    sessions=sessions,
                    labels=labels,
                )
                checkpoint_path = run_dir / "best_checkpoint.pt"
                torch.save(
                    {
                        "state_dict": best_state,
                        "input_mean": scaler.mean_.astype(np.float32),
                        "input_scale": scaler.scale_.astype(np.float32),
                        "manifest_sha256": manifest["manifest_sha256"],
                        "config_sha256": expected_hash,
                        "holdout": holdout,
                        "seed": int(seed),
                    },
                    checkpoint_path,
                )
                result: dict[str, object] = {
                    "run_name": run_name,
                    "direction": direction,
                    "heldout_class": CLASS_NAMES[holdout],
                    "heldout_class_id": holdout,
                    "seed": int(seed),
                    "best_epoch": best_epoch,
                    "source_validation_prototype_macro_f1": best_score,
                    "epochs_run": len(history),
                    "training_seconds": time.perf_counter() - started,
                    "inference_seconds_all_sessions": inference_seconds,
                    "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
                    "checkpoint_path": checkpoint_path.as_posix(),
                    "checkpoint_sha256": _sha256(checkpoint_path),
                    "embedding_path": embedding_path.as_posix(),
                    "embedding_sha256": _sha256(embedding_path),
                    "cuda": cuda,
                    "selection_used_target_query": False,
                }
                result["payload_sha256"] = canonical_json_hash(result)
                (run_dir / "run_result.json").write_text(
                    json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
                )
                (run_dir / "training_history.json").write_text(
                    json.dumps(history, indent=2, sort_keys=True), encoding="utf-8"
                )
                results.append(result)
                print(
                    f"[SIAMESE] {run_name} val_retrieval={best_score:.4f} epoch={best_epoch}",
                    flush=True,
                )
                del model
                torch.cuda.empty_cache()
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "CUDA session-level supervised Siamese embedding v3",
        "evidence_status": "smoke" if smoke else "retrospective development",
        "config_sha256": expected_hash,
        "cuda": cuda,
        "run_count": len(results),
        "elapsed_seconds": time.perf_counter() - started_all,
        "runs": results,
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "siamese_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-aggregate", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(
        session_aggregate_path=args.session_aggregate,
        manifests=args.manifests,
        config_path=args.config,
        config_hash_path=args.config_hash,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )
    print(json.dumps({"run_count": result["run_count"], "payload_sha256": result["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
