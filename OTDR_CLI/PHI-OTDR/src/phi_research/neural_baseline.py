"""CUDA training for corrected CNN, TCN, and TFT-style supervised baselines."""

from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from model_functions.cnn import CNN
from model_functions.tcn import TCN
from model_functions.tft import TemporalFusionTransformer

from .data_contract import CLASS_NAMES
from .metrics import aggregate_session_predictions, classification_metrics
from .neural_data import ManifestWindowDataset, cached_window_dataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Avoid the non-deterministic memory-efficient attention kernel on CUDA.
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)


def build_model(name: str) -> tuple[nn.Module, dict[str, object]]:
    if name == "cnn":
        return CNN(n_classes=len(CLASS_NAMES)), {"n_classes": len(CLASS_NAMES)}
    if name == "tcn":
        config = {
            "in_channels": 12,
            "n_classes": len(CLASS_NAMES),
            "channels": (32, 64, 64, 96),
            "kernel_size": 5,
            "dropout": 0.2,
        }
        return TCN(**config), config
    if name == "tft":
        config = {
            "in_channels": 12,
            "n_classes": len(CLASS_NAMES),
            "d_model": 72,
            "n_heads": 3,
            "num_layers": 2,
            "d_ff": 144,
            "dropout": 0.1,
            "max_tokens": 512,
        }
        return TemporalFusionTransformer(**config), config
    raise ValueError(f"Unknown model: {name}")


def _prepare_input(model_name: str, data: torch.Tensor, device: torch.device) -> torch.Tensor:
    data = data.to(device, dtype=torch.float32, non_blocking=True)
    return data.unsqueeze(1) if model_name == "cnn" else data


def _evaluate(
    model: nn.Module,
    model_name: str,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    *,
    max_batches: int | None = None,
) -> dict[str, object]:
    model.eval()
    losses: list[float] = []
    labels: list[np.ndarray] = []
    logits_rows: list[np.ndarray] = []
    sessions: list[str] = []
    paths: list[str] = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            x = _prepare_input(model_name, batch["data"], device)
            y = batch["label"].to(device, dtype=torch.long, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logits = model(x)
                loss = criterion(logits, y)
            losses.append(float(loss.detach().cpu()))
            labels.append(y.cpu().numpy())
            logits_rows.append(logits.float().cpu().numpy())
            sessions.extend(str(value) for value in batch["session"])
            paths.extend(str(value) for value in batch["rel_path"])
    y_true = np.concatenate(labels)
    logits_np = np.concatenate(logits_rows)
    probabilities = torch.softmax(torch.from_numpy(logits_np), dim=1).numpy()
    predictions = np.argmax(logits_np, axis=1)
    window_metrics = classification_metrics(y_true, predictions)
    session_true, session_pred, ordered_sessions = aggregate_session_predictions(
        y_true, sessions, probabilities=probabilities
    )
    return {
        "loss": float(np.mean(losses)),
        "window_metrics": window_metrics,
        "session_metrics": classification_metrics(session_true, session_pred),
        "y_true": y_true,
        "logits": logits_np,
        "probabilities": probabilities,
        "predictions": predictions,
        "sessions": np.asarray(sessions),
        "paths": np.asarray(paths),
        "session_ids": np.asarray(ordered_sessions),
        "session_true": session_true,
        "session_pred": session_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("cnn", "tcn", "tft"), required=True)
    parser.add_argument("--normalization", choices=("global_minmax", "global_zscore", "channel_zscore"), default="global_minmax")
    parser.add_argument("--temporal-pool", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-validation-batches", type=int)
    parser.add_argument("--train-partition", default="train")
    parser.add_argument("--validation-partition", default="validation")
    parser.add_argument(
        "--cache-root",
        type=Path,
        help="Optional generated float32 cache root; shared safely by model runs with the same manifest.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for the Phi-OTDR neural research protocol")
    device = torch.device("cuda")
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ManifestWindowDataset(
        args.data_root, args.manifest, (args.train_partition,), normalization=args.normalization,
        temporal_pool=args.temporal_pool,
    )
    validation_dataset = ManifestWindowDataset(
        args.data_root, args.manifest, (args.validation_partition,), normalization=args.normalization,
        temporal_pool=args.temporal_pool,
    )
    if args.cache_root is not None:
        cache_suffix = f"{args.normalization}_pool{args.temporal_pool}"
        train_dataset = cached_window_dataset(
            train_dataset,
            args.manifest,
            args.cache_root / args.train_partition / cache_suffix,
        )
        validation_dataset = cached_window_dataset(
            validation_dataset,
            args.manifest,
            args.cache_root / args.validation_partition / cache_suffix,
        )
    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": 0,
        "pin_memory": True,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, generator=generator, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)

    model, model_config = build_model(args.model)
    model = model.to(device)
    counts = Counter(sample.class_id for sample in train_dataset.samples)
    class_weights = torch.tensor(
        [len(train_dataset) / (len(CLASS_NAMES) * counts[class_id]) for class_id in range(len(CLASS_NAMES))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda")
    history: list[dict[str, object]] = []
    best_macro_f1 = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint = args.output_dir / "best_model.pt"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        train_true: list[np.ndarray] = []
        train_pred: list[np.ndarray] = []
        for batch_index, batch in enumerate(train_loader):
            if args.max_train_batches is not None and batch_index >= args.max_train_batches:
                break
            x = _prepare_input(args.model, batch["data"], device)
            y = batch["label"].to(device, dtype=torch.long, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                _, logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(logits.detach().argmax(dim=1).cpu().numpy())
        validation = _evaluate(
            model,
            args.model,
            validation_loader,
            device,
            criterion,
            max_batches=args.max_validation_batches,
        )
        train_y = np.concatenate(train_true)
        train_p = np.concatenate(train_pred)
        train_macro_f1 = float(f1_score(train_y, train_p, average="macro", zero_division=0))
        validation_macro_f1 = float(validation["window_metrics"]["macro_f1"])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            "train_macro_f1": train_macro_f1,
            "validation_loss": validation["loss"],
            "validation_window_macro_f1": validation_macro_f1,
            "validation_session_macro_f1": validation["session_metrics"]["macro_f1"],
        }
        history.append(row)
        print(f"[{args.model}] {row}", flush=True)
        if validation_macro_f1 > best_macro_f1:
            best_macro_f1 = validation_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "model": args.model,
                    "model_config": model_config,
                    "normalization": args.normalization,
                    "temporal_pool": args.temporal_pool,
                    "seed": args.seed,
                    "epoch": epoch,
                    "validation_macro_f1": validation_macro_f1,
                    "train_partition": args.train_partition,
                    "validation_partition": args.validation_partition,
                },
                checkpoint,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                break

    saved = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(saved["state_dict"])
    final_validation = _evaluate(model, args.model, validation_loader, device, criterion)
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device))
    artifact = {
        "protocol": "session-safe supervised development baseline; selected source partitions; target query untouched",
        "model": args.model,
        "model_config": model_config,
        "normalization": args.normalization,
        "temporal_pool": args.temporal_pool,
        "seed": args.seed,
        "train_partition": args.train_partition,
        "validation_partition": args.validation_partition,
        "cache_root": None if args.cache_root is None else args.cache_root.as_posix(),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "history": history,
        "train_windows": len(train_dataset),
        "validation_windows": len(validation_dataset),
        "train_sessions": len({sample.session_id for sample in train_dataset.samples}),
        "validation_sessions": len({sample.session_id for sample in validation_dataset.samples}),
        "validation_window_metrics": final_validation["window_metrics"],
        "validation_session_metrics": final_validation["session_metrics"],
        "elapsed_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
        "final_query_used": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "pid": os.getpid(),
        },
    }
    (args.output_dir / "development_results.json").write_text(
        json.dumps(artifact, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        y_true=final_validation["y_true"],
        logits=final_validation["logits"],
        probabilities=final_validation["probabilities"],
        predictions=final_validation["predictions"],
        sessions=final_validation["sessions"],
        rel_paths=final_validation["paths"],
        session_ids=final_validation["session_ids"],
        session_true=final_validation["session_true"],
        session_pred=final_validation["session_pred"],
    )
    print(json.dumps({"best_epoch": best_epoch, "validation": final_validation["window_metrics"], "session": final_validation["session_metrics"], "elapsed_seconds": elapsed, "peak_cuda_memory_bytes": peak_memory}, indent=2))


if __name__ == "__main__":
    main()
