"""CUDA-only session-level DeepSets/attention models for Phi-OTDR v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from sklearn.metrics import f1_score, log_loss
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .data_contract import CLASS_NAMES, canonical_json_hash
from .morphology_features import transform_view
from .spatial_experiment import _metrics, _select_temperature, _temperature


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _cuda_metadata() -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for Phi-OTDR v3 neural training and inference")
    properties = torch.cuda.get_device_properties(0)
    return {
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "device_index": 0,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "total_vram_bytes": int(properties.total_memory),
        "amp": "float16 autocast; float32 parameters",
    }


class SessionDataset(Dataset):
    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        sessions: np.ndarray,
        window_ids: np.ndarray,
        selected_sessions: Sequence[str],
    ) -> None:
        self.rows = []
        session_array = sessions.astype(str)
        for session in sorted(selected_sessions):
            indices = np.flatnonzero(session_array == session)
            order = np.argsort(window_ids[indices], kind="stable")
            indices = indices[order]
            unique_labels = set(labels[indices].tolist())
            if len(unique_labels) != 1:
                raise ValueError(f"Inconsistent session labels: {session}")
            local_ids = window_ids[indices].astype(np.float32)
            if len(local_ids) > 1 and local_ids[-1] > local_ids[0]:
                time_feature = 2.0 * (local_ids - local_ids[0]) / (local_ids[-1] - local_ids[0]) - 1.0
            else:
                time_feature = np.zeros(len(local_ids), dtype=np.float32)
            self.rows.append(
                (
                    session,
                    int(labels[indices[0]]),
                    np.column_stack((features[indices], time_feature)).astype(np.float32),
                )
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[str, int, np.ndarray]:
        return self.rows[index]


def collate_sessions(
    batch: Sequence[tuple[str, int, np.ndarray]],
) -> tuple[list[str], torch.Tensor, torch.Tensor, torch.Tensor]:
    sessions = [row[0] for row in batch]
    labels = torch.as_tensor([row[1] for row in batch], dtype=torch.long)
    maximum = max(len(row[2]) for row in batch)
    dimensions = batch[0][2].shape[1]
    values = torch.zeros((len(batch), maximum, dimensions), dtype=torch.float32)
    mask = torch.zeros((len(batch), maximum), dtype=torch.bool)
    for index, (_, _, local) in enumerate(batch):
        values[index, : len(local)] = torch.from_numpy(local)
        mask[index, : len(local)] = True
    return sessions, labels, values, mask


class SessionNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float, architecture: str) -> None:
        super().__init__()
        if architecture not in {"deepsets", "attention"}:
            raise ValueError(f"Unknown session architecture: {architecture}")
        self.architecture = architecture
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )
        self.attention = nn.Linear(hidden_dim, 1) if architecture == "attention" else None
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, len(CLASS_NAMES)),
        )

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(values)
        valid = mask.unsqueeze(-1)
        count = valid.sum(dim=1).clamp_min(1)
        mean = (encoded * valid).sum(dim=1) / count
        variance = ((encoded - mean[:, None, :]) ** 2 * valid).sum(dim=1) / count
        standard_deviation = torch.sqrt(variance + 1e-6)
        if self.attention is None:
            pooled = encoded.masked_fill(~valid, torch.finfo(encoded.dtype).min).max(dim=1).values
        else:
            score = self.attention(encoded).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(score, dim=1)
            pooled = torch.sum(encoded * weights.unsqueeze(-1), dim=1)
        return self.head(torch.cat((pooled, mean, standard_deviation), dim=1))


def _prepare_view(
    bundle: dict[str, np.ndarray],
    *,
    spec: dict[str, str],
    cache_path: Path,
) -> np.ndarray:
    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as cached:
            values = cached["features"]
        if len(values) != len(bundle["features"]):
            raise ValueError("Neural window-view cache length mismatch")
        return values
    names = bundle["feature_names"].astype(str).tolist()
    values = np.stack(
        [
            transform_view(
                row,
                names,
                view=spec["view"],
                estimator=spec["estimator"],
                ablation=spec["ablation"],
            )[0]
            for row in bundle["features"]
        ]
    ).astype(np.float32)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=values)
    return values


@torch.no_grad()
def _infer(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, list[str], float]:
    model.eval()
    probabilities = []
    labels = []
    sessions = []
    started = time.perf_counter()
    for local_sessions, local_labels, values, mask in loader:
        values = values.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(values, mask)
        probabilities.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        labels.append(local_labels.numpy())
        sessions.extend(local_sessions)
    torch.cuda.synchronize()
    return (
        np.concatenate(probabilities),
        np.concatenate(labels),
        sessions,
        time.perf_counter() - started,
    )


def _train_candidate(
    *,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    input_dim: int,
    hidden_dim: int,
    dropout: float,
    architecture: str,
    learning_rate: float,
    weight_decay: float,
    class_weights: torch.Tensor,
    max_epochs: int,
    patience: int,
    gradient_clip: float,
    seed: int,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = SessionNet(input_dim, hidden_dim, dropout, architecture).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    best_key = None
    best_state = None
    best_epoch = -1
    history = []
    stale = 0
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(max_epochs):
        model.train()
        losses = []
        for _, labels, values, mask in train_loader:
            values = values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            mask = mask.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(values, mask)
                loss = loss_function(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        validation_probs, validation_labels, _, _ = _infer(model, validation_loader, device)
        macro = float(
            f1_score(
                validation_labels,
                np.argmax(validation_probs, axis=1),
                average="macro",
                zero_division=0,
            )
        )
        nll = float(log_loss(validation_labels, validation_probs, labels=np.arange(len(CLASS_NAMES))))
        history.append({"epoch": epoch + 1, "train_loss": float(np.mean(losses)), "validation_macro_f1": macro, "validation_nll": nll})
        key = (macro, -nll)
        if best_key is None or key > best_key:
            best_key = key
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            best_epoch = epoch + 1
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    assert best_state is not None and best_key is not None
    model.load_state_dict(best_state)
    torch.cuda.synchronize()
    return {
        "model": model,
        "best_state": best_state,
        "best_epoch": best_epoch,
        "best_macro_f1": best_key[0],
        "best_nll": -best_key[1],
        "history": history,
        "training_seconds": time.perf_counter() - started,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def run(
    *,
    bundle_path: Path,
    manifests: Sequence[Path],
    config_path: Path,
    config_hash_path: Path,
    output_dir: Path,
    smoke: bool = False,
) -> dict[str, object]:
    cuda = _cuda_metadata()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = config_hash_path.read_text(encoding="utf-8").split()[0]
    if canonical_json_hash(config) != expected_hash:
        raise ValueError("Session-neural v3 config hash mismatch")
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    if len(bundle["features"]) != 15418:
        raise ValueError("Neural input bundle is not the complete readable cohort")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir.parent / "neural_window_cache"
    device = torch.device("cuda:0")
    manifests_payload = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    view_specs = config["views"][:1] if smoke else config["views"]
    architectures = config["architectures"][:1] if smoke else config["architectures"]
    seeds = config["seeds"][:1] if smoke else config["seeds"]
    learning_rates = config["training"]["learning_rates"][:1] if smoke else config["training"]["learning_rates"]
    max_epochs = 3 if smoke else int(config["training"]["max_epochs"])
    patience = 2 if smoke else int(config["training"]["patience"])
    results = []
    prediction_rows = []
    started_all = time.perf_counter()
    for spec in view_specs:
        window_features = _prepare_view(
            bundle,
            spec=spec,
            cache_path=cache_dir / f"{spec['name']}.npz",
        )
        for manifest_path, manifest in zip(manifests, manifests_payload, strict=True):
            session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
            partition_sessions: dict[str, list[str]] = defaultdict(list)
            for session, row in session_rows.items():
                partition_sessions[str(row["partition"])].append(session)
            direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
            source_train_mask = np.asarray(
                [session_rows[s]["partition"] == "source_train" for s in bundle["sessions"].astype(str)]
            )
            mean = np.mean(window_features[source_train_mask], axis=0)
            std = np.std(window_features[source_train_mask], axis=0)
            std = np.maximum(std, 1e-6)
            normalized = ((window_features - mean) / std).astype(np.float32)
            datasets = {
                partition: SessionDataset(
                    normalized,
                    bundle["labels"],
                    bundle["sessions"],
                    bundle["window_ids"],
                    partition_sessions[partition],
                )
                for partition in ("source_train", "source_validation", "source_calibration", "target_query")
            }
            for architecture in architectures:
                for seed in seeds:
                    generator = torch.Generator().manual_seed(int(seed))
                    train_loader = DataLoader(
                        datasets["source_train"],
                        batch_size=int(config["training"]["batch_size_sessions"]),
                        shuffle=True,
                        generator=generator,
                        num_workers=0,
                        collate_fn=collate_sessions,
                    )
                    loaders = {
                        name: DataLoader(
                            dataset,
                            batch_size=int(config["training"]["batch_size_sessions"]),
                            shuffle=False,
                            num_workers=0,
                            collate_fn=collate_sessions,
                        )
                        for name, dataset in datasets.items()
                        if name != "source_train"
                    }
                    train_labels = np.asarray([row[1] for row in datasets["source_train"].rows])
                    counts = np.bincount(train_labels, minlength=len(CLASS_NAMES)).astype(np.float32)
                    weights = counts.sum() / np.maximum(counts * len(CLASS_NAMES), 1.0)
                    class_weights = torch.as_tensor(weights, device=device)
                    candidates = []
                    for learning_rate in learning_rates:
                        candidate = _train_candidate(
                            train_loader=train_loader,
                            validation_loader=loaders["source_validation"],
                            input_dim=normalized.shape[1] + 1,
                            hidden_dim=int(config["model"]["hidden_dim"]),
                            dropout=float(config["model"]["dropout"]),
                            architecture=architecture,
                            learning_rate=float(learning_rate),
                            weight_decay=float(config["training"]["weight_decay"]),
                            class_weights=class_weights,
                            max_epochs=max_epochs,
                            patience=patience,
                            gradient_clip=float(config["training"]["gradient_clip_norm"]),
                            seed=int(seed),
                            device=device,
                        )
                        candidate["learning_rate"] = float(learning_rate)
                        candidates.append(candidate)
                    selected = max(candidates, key=lambda row: (row["best_macro_f1"], -row["best_nll"], -row["learning_rate"]))
                    model = selected["model"]
                    calibration_probs, calibration_labels, _, calibration_seconds = _infer(
                        model, loaders["source_calibration"], device
                    )
                    temperature = _select_temperature(calibration_labels, calibration_probs)
                    validation_probs, validation_labels, _, validation_seconds = _infer(
                        model, loaders["source_validation"], device
                    )
                    target_probs, target_labels, target_sessions, target_seconds = _infer(
                        model, loaders["target_query"], device
                    )
                    validation_probs = _temperature(validation_probs, temperature)
                    target_probs = _temperature(target_probs, temperature)
                    run_name = f"{direction}__{spec['name']}__{architecture}__seed{seed}"
                    run_dir = output_dir / run_name
                    run_dir.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = run_dir / "best_checkpoint.pt"
                    torch.save(
                        {
                            "state_dict": selected["best_state"],
                            "input_mean": mean.astype(np.float32),
                            "input_std": std.astype(np.float32),
                            "view": spec,
                            "architecture": architecture,
                            "seed": int(seed),
                            "learning_rate": selected["learning_rate"],
                            "manifest_sha256": manifest["manifest_sha256"],
                            "config_sha256": expected_hash,
                        },
                        checkpoint_path,
                    )
                    selection_trace = [
                        {
                            "learning_rate": row["learning_rate"],
                            "best_epoch": row["best_epoch"],
                            "best_source_validation_macro_f1": row["best_macro_f1"],
                            "best_source_validation_nll": row["best_nll"],
                            "epochs_run": len(row["history"]),
                            "training_seconds": row["training_seconds"],
                        }
                        for row in candidates
                    ]
                    result: dict[str, object] = {
                        "run_name": run_name,
                        "direction": direction,
                        "view": spec,
                        "architecture": architecture,
                        "seed": int(seed),
                        "selected_learning_rate": selected["learning_rate"],
                        "selected_epoch": selected["best_epoch"],
                        "selection_trace": selection_trace,
                        "source_validation": _metrics(validation_labels, validation_probs),
                        "target_query_retrospective": _metrics(target_labels, target_probs),
                        "temperature": temperature,
                        "training_seconds_selected_candidate": selected["training_seconds"],
                        "total_candidate_training_seconds": float(sum(row["training_seconds"] for row in candidates)),
                        "calibration_inference_seconds": calibration_seconds,
                        "validation_inference_seconds": validation_seconds,
                        "target_inference_seconds": target_seconds,
                        "peak_cuda_memory_bytes": selected["peak_cuda_memory_bytes"],
                        "checkpoint_path": checkpoint_path.as_posix(),
                        "checkpoint_sha256": _sha256(checkpoint_path),
                        "cuda": cuda,
                        "selection_used_target_query": False,
                    }
                    result["payload_sha256"] = canonical_json_hash(result)
                    (run_dir / "run_result.json").write_text(
                        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
                    )
                    (run_dir / "training_history.json").write_text(
                        json.dumps(
                            {
                                str(row["learning_rate"]): row["history"] for row in candidates
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    results.append(result)
                    for index, session in enumerate(target_sessions):
                        prediction = int(np.argmax(target_probs[index]))
                        row = {
                            "run_name": run_name,
                            "direction": direction,
                            "view": spec["name"],
                            "architecture": architecture,
                            "seed": int(seed),
                            "session_id": session,
                            "true_label": int(target_labels[index]),
                            "predicted_label": prediction,
                            "true_class": CLASS_NAMES[int(target_labels[index])],
                            "predicted_class": CLASS_NAMES[prediction],
                        }
                        for class_id, class_name in enumerate(CLASS_NAMES):
                            row[f"prob_{class_name}"] = float(target_probs[index, class_id])
                        prediction_rows.append(row)
                    print(
                        f"[NEURAL] {run_name} val={result['source_validation']['macro_f1']:.4f} "
                        f"target={result['target_query_retrospective']['macro_f1']:.4f} "
                        f"peakMiB={result['peak_cuda_memory_bytes'] / 2**20:.1f}",
                        flush=True,
                    )
                    del model
                    for candidate in candidates:
                        candidate.pop("model", None)
                    torch.cuda.empty_cache()
    summary: dict[str, object] = {
        "schema_version": 1,
        "protocol": "CUDA-only complete-data session neural v3",
        "evidence_status": "smoke" if smoke else "retrospective development",
        "config_sha256": expected_hash,
        "dataset_fingerprint_sha256": config["dataset_fingerprint_sha256"],
        "bundle_sha256": _sha256(bundle_path),
        "cuda": cuda,
        "run_count": len(results),
        "elapsed_seconds": time.perf_counter() - started_all,
        "runs": results,
    }
    summary["payload_sha256"] = canonical_json_hash(summary)
    (output_dir / "neural_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    if prediction_rows:
        with (output_dir / "neural_target_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(prediction_rows[0]))
            writer.writeheader()
            writer.writerows(prediction_rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = run(
        bundle_path=args.bundle,
        manifests=args.manifests,
        config_path=args.config,
        config_hash_path=args.config_hash,
        output_dir=args.output_dir,
        smoke=args.smoke,
    )
    print(
        json.dumps(
            {
                "run_count": result["run_count"],
                "elapsed_seconds": result["elapsed_seconds"],
                "payload_sha256": result["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
