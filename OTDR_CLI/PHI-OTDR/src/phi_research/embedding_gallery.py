"""Frozen supervised feature encoder with true held-out-class gallery evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from .data_contract import CLASS_NAMES
from .gallery_baseline import (
    _aggregate_session_scores,
    _calibration_scores,
    _class_prototypes,
    _draw_seed,
    _post_enrollment_metrics,
    _predict,
    _session_prototypes,
)
from .metrics import calibrate_rejection_threshold, open_set_metrics


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim: int, class_count: int, embedding_dim: int = 64) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, class_count)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(values)
        return embedding, self.classifier(embedding)


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.benchmark = False


def _feature_masks(names: np.ndarray) -> dict[str, np.ndarray]:
    names = names.astype(str)
    amplitude_globals = {"global_mean", "global_std", "global_range"}
    amplitude = np.asarray([name.startswith("raw_") or name in amplitude_globals for name in names])
    return {"dynamics": ~amplitude, "full": np.ones(len(names), dtype=bool)}


def _encode(model: FeatureEncoder, values: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 1024):
            batch = torch.from_numpy(values[start : start + 1024]).to(device)
            embedding, _ = model(batch)
            rows.append(embedding.float().cpu().numpy())
    return np.concatenate(rows)


def _session_sampler(labels: np.ndarray, sessions: np.ndarray, seed: int) -> WeightedRandomSampler:
    session_counts = Counter(sessions.astype(str))
    class_session_counts = {
        class_id: len(np.unique(sessions[labels == class_id])) for class_id in np.unique(labels)
    }
    weights = np.asarray(
        [1.0 / (session_counts[str(session)] * class_session_counts[int(label)]) for label, session in zip(labels, sessions)],
        dtype=np.float64,
    )
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        torch.from_numpy(weights), num_samples=len(weights), replacement=True, generator=generator
    )


def _fold_seed(base_seed: int, holdout: int, ablation: str) -> int:
    digest = hashlib.sha256(f"{base_seed}|{holdout}|{ablation}".encode()).digest()
    return int.from_bytes(digest[:4], "little")


def _train_fold(
    x: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    partitions: np.ndarray,
    *,
    holdout: int,
    feature_mask: np.ndarray,
    seed: int,
    output_dir: Path,
    epochs: int,
    patience: int,
) -> tuple[FeatureEncoder, StandardScaler, dict[str, object]]:
    device = torch.device("cuda")
    seen_classes = [class_id for class_id in range(len(CLASS_NAMES)) if class_id != holdout]
    class_to_local = {class_id: index for index, class_id in enumerate(seen_classes)}
    train = (partitions == "train") & np.isin(y, seen_classes)
    validation = (partitions == "validation") & np.isin(y, seen_classes)
    scaler = StandardScaler().fit(x[train][:, feature_mask])
    transformed_train = scaler.transform(x[train][:, feature_mask]).astype(np.float32)
    transformed_validation = scaler.transform(x[validation][:, feature_mask]).astype(np.float32)
    train_targets = np.asarray([class_to_local[int(value)] for value in y[train]], dtype=np.int64)
    validation_targets = np.asarray([class_to_local[int(value)] for value in y[validation]], dtype=np.int64)
    dataset = TensorDataset(torch.from_numpy(transformed_train), torch.from_numpy(train_targets))
    sampler = _session_sampler(y[train], sessions[train], seed)
    loader = DataLoader(dataset, batch_size=256, sampler=sampler, num_workers=0, pin_memory=True)
    model = FeatureEncoder(transformed_train.shape[1], len(seen_classes)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    checkpoint = output_dir / "best_model.pt"
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        for values, targets in loader:
            values = values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            _, logits = model(values)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.inference_mode():
            values = torch.from_numpy(transformed_validation).to(device)
            _, logits = model(values)
            predicted = logits.argmax(dim=1).cpu().numpy()
        macro_f1 = float(f1_score(validation_targets, predicted, average="macro", zero_division=0))
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "seen_validation_macro_f1": macro_f1})
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    joblib.dump(scaler, output_dir / "scaler.joblib")
    metadata = {
        "holdout_class": CLASS_NAMES[holdout],
        "seen_classes": [CLASS_NAMES[value] for value in seen_classes],
        "seed": seed,
        "best_epoch": best_epoch,
        "seen_validation_macro_f1": best_f1,
        "epochs_completed": len(history),
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
    }
    (output_dir / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, scaler, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--support-draws", type=int, default=20)
    parser.add_argument("--ablations", nargs="+", choices=("dynamics", "full"), default=["dynamics", "full"])
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory")
    bundle = np.load(args.features, allow_pickle=False)
    x = bundle["features"]
    y = bundle["labels"]
    sessions = bundle["sessions"].astype(str)
    partitions = bundle["partitions"].astype(str)
    names = bundle["feature_names"].astype(str)
    if np.any(partitions == "final_query"):
        raise ValueError("Development embedding run cannot receive final_query")
    results: list[dict[str, object]] = []

    available_masks = _feature_masks(names)
    for ablation in args.ablations:
        mask = available_masks[ablation]
        for holdout in range(len(CLASS_NAMES)):
            fold_seed = _fold_seed(args.seed, holdout, ablation)
            _seed(fold_seed)
            fold_dir = args.output_dir / ablation / CLASS_NAMES[holdout]
            model, scaler, training = _train_fold(
                x,
                y,
                sessions,
                partitions,
                holdout=holdout,
                feature_mask=mask,
                seed=fold_seed,
                output_dir=fold_dir,
                epochs=args.epochs,
                patience=args.patience,
            )
            transformed = scaler.transform(x[:, mask]).astype(np.float32)
            embedding = _encode(model, transformed, torch.device("cuda"))
            seen_classes = [class_id for class_id in range(len(CLASS_NAMES)) if class_id != holdout]
            train = (partitions == "train") & np.isin(y, seen_classes)
            validation = partitions == "validation"
            calibration = (partitions == "calibration") & np.isin(y, seen_classes)
            support = (partitions == "support") & (y == holdout)
            train_session_x, train_session_y, _ = _session_prototypes(
                embedding[train], y[train], sessions[train]
            )
            support_session_x, _, support_session_ids = _session_prototypes(
                embedding[support], y[support], sessions[support]
            )
            base_prototypes, base_labels = _class_prototypes(
                train_session_x, train_session_y, seen_classes
            )
            for metric in ("cosine", "euclidean"):
                known_calibration, unknown_calibration = _calibration_scores(
                    train_session_x,
                    train_session_y,
                    embedding[calibration],
                    y[calibration],
                    seen_classes,
                    metric,
                )
                thresholds = calibrate_rejection_threshold(known_calibration, unknown_calibration)
                validation_pred, confidence, _ = _predict(
                    embedding[validation], base_prototypes, base_labels, metric
                )
                is_known = y[validation] != holdout
                correct = validation_pred == y[validation]
                pre = {
                    mode: open_set_metrics(confidence, is_known, correct, threshold=float(threshold))
                    for mode, threshold in (
                        ("balanced", thresholds["balanced_threshold"]),
                        ("known_acceptance_95", thresholds["known_acceptance_threshold"]),
                    )
                }
                shots: dict[str, object] = {}
                for shot in (1, 3, 5):
                    draws: list[dict[str, object]] = []
                    for draw in range(args.support_draws):
                        rng = np.random.default_rng(_draw_seed(args.seed, holdout, shot, draw))
                        selected = np.sort(rng.choice(len(support_session_ids), size=shot, replace=False))
                        enrolled = np.mean(support_session_x[selected], axis=0, keepdims=True)
                        prototypes = np.concatenate((base_prototypes, enrolled))
                        prototype_labels = np.concatenate((base_labels, np.asarray([holdout])))
                        predicted, _, scores = _predict(
                            embedding[validation], prototypes, prototype_labels, metric
                        )
                        window_metrics = _post_enrollment_metrics(y[validation], predicted, holdout)
                        session_true, session_pred = _aggregate_session_scores(
                            scores, y[validation], sessions[validation], prototype_labels
                        )
                        draws.append(
                            {
                                "draw": draw,
                                "support_sessions": support_session_ids[selected].tolist(),
                                "window": window_metrics,
                                "session": _post_enrollment_metrics(session_true, session_pred, holdout),
                            }
                        )
                    shots[str(shot)] = {
                        "draws": draws,
                        "session_h_mean": float(np.mean([row["session"]["enrollment_h"] for row in draws])),
                        "session_h_min": float(np.min([row["session"]["enrollment_h"] for row in draws])),
                        "window_h_mean": float(np.mean([row["window"]["enrollment_h"] for row in draws])),
                    }
                row = {
                    "ablation": ablation,
                    "holdout_class_id": holdout,
                    "holdout_class": CLASS_NAMES[holdout],
                    "metric": metric,
                    "training": training,
                    "calibration": thresholds,
                    "pre_enrollment": pre,
                    "post_enrollment": shots,
                }
                results.append(row)
                print(
                    f"[{ablation} {CLASS_NAMES[holdout]} {metric}] "
                    f"seenF1={training['seen_validation_macro_f1']:.3f} "
                    f"AUROC={pre['balanced']['unknown_auroc']:.3f} "
                    f"preH={pre['balanced']['detection_h']:.3f} "
                    f"postH@5={shots['5']['session_h_mean']:.3f}",
                    flush=True,
                )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[f"{row['ablation']}__{row['metric']}"] .append(row)
    summary: dict[str, dict[str, float]] = {}
    for key, rows in grouped.items():
        summary[key] = {
            "pre_unknown_auroc_mean": float(np.mean([row["pre_enrollment"]["balanced"]["unknown_auroc"] for row in rows])),
            "pre_detection_h_mean": float(np.mean([row["pre_enrollment"]["balanced"]["detection_h"] for row in rows])),
            "pre_detection_h_worst": float(np.min([row["pre_enrollment"]["balanced"]["detection_h"] for row in rows])),
            "post_session_h_1shot_mean": float(np.mean([row["post_enrollment"]["1"]["session_h_mean"] for row in rows])),
            "post_session_h_5shot_mean": float(np.mean([row["post_enrollment"]["5"]["session_h_mean"] for row in rows])),
            "post_session_h_5shot_worst_draw": float(np.min([row["post_enrollment"]["5"]["session_h_min"] for row in rows])),
        }
    ranking = sorted(
        summary,
        key=lambda key: (summary[key]["pre_detection_h_mean"], summary[key]["post_session_h_5shot_mean"]),
        reverse=True,
    )
    payload = {
        "protocol": "frozen supervised encoder; outer held-out class excluded from fitting",
        "seed": args.seed,
        "support_draws": args.support_draws,
        "final_query_used": False,
        "summary": summary,
        "development_ranking": ranking,
        "fold_results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "development_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"ranking": ranking, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
