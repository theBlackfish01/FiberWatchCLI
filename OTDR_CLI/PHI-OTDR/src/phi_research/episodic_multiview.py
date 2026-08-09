"""Multi-view episodic session encoder for open-world Phi-OTDR enrollment."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES
from .density_open_analysis import DensityGallery, _confidence_prediction
from .gallery_baseline import _draw_seed, _post_enrollment_metrics, _session_prototypes
from .metrics import calibrate_rejection_threshold, open_set_metrics


VIEW_NAMES = ("amplitude", "temporal", "spectral", "correlation")


def build_view_masks(feature_names: np.ndarray) -> dict[str, np.ndarray]:
    names = feature_names.astype(str)
    masks = {
        "amplitude": np.asarray(
            [name.startswith("raw_") or name in {"global_mean", "global_std", "global_range"} for name in names]
        ),
        "temporal": np.asarray(
            [
                name.startswith(("delta_", "block_dynamic_", "spatial_"))
                or name in {"global_delta_mean_abs", "global_delta_std"}
                for name in names
            ]
        ),
        "spectral": np.asarray([name.startswith("spectrum_") for name in names]),
        "correlation": np.asarray(
            [name.startswith(("correlation_", "neighbor_correlation_")) or name == "global_mean_neighbor_correlation" for name in names]
        ),
    }
    coverage = np.sum(np.stack(list(masks.values())), axis=0)
    if not np.all(coverage == 1):
        bad = names[coverage != 1].tolist()
        raise ValueError(f"Feature views must partition the schema exactly; bad={bad}")
    return masks


class MultiViewEncoder(nn.Module):
    def __init__(self, view_indices: dict[str, np.ndarray], view_dim: int = 32, embedding_dim: int = 64):
        super().__init__()
        self.view_order = list(VIEW_NAMES)
        self.indices: dict[str, torch.Tensor] = {}
        self.view_nets = nn.ModuleDict()
        for name in self.view_order:
            indices = torch.from_numpy(np.flatnonzero(view_indices[name]).astype(np.int64))
            self.register_buffer(f"indices_{name}", indices)
            self.indices[name] = indices
            self.view_nets[name] = nn.Sequential(
                nn.Linear(len(indices), 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(64, view_dim),
                nn.GELU(),
            )
        self.fusion = nn.Sequential(
            nn.Linear(view_dim * len(self.view_order), 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, embedding_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        views = []
        for name in self.view_order:
            indices = getattr(self, f"indices_{name}")
            views.append(self.view_nets[name](values.index_select(1, indices)))
        embedding = self.fusion(torch.cat(views, dim=1))
        return F.normalize(embedding, dim=1)


def sample_episode(
    labels: np.ndarray,
    class_ids: list[int],
    rng: np.random.Generator,
    *,
    support_per_class: int = 3,
    query_per_class: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support: list[int] = []
    query: list[int] = []
    support_labels: list[int] = []
    query_labels: list[int] = []
    for local_label, class_id in enumerate(class_ids):
        candidates = np.flatnonzero(labels == class_id)
        selected = rng.choice(candidates, size=support_per_class + query_per_class, replace=False)
        support.extend(selected[:support_per_class])
        query.extend(selected[support_per_class:])
        support_labels.extend([local_label] * support_per_class)
        query_labels.extend([local_label] * query_per_class)
    if set(support) & set(query):
        raise AssertionError("Episode support/query sessions overlap")
    return (
        np.asarray(support),
        np.asarray(support_labels),
        np.asarray(query),
        np.asarray(query_labels),
    )


def _prototypes(embedding: torch.Tensor, labels: torch.Tensor, class_count: int) -> torch.Tensor:
    return torch.stack([embedding[labels == class_id].mean(dim=0) for class_id in range(class_count)])


def _episode_loss(
    model: MultiViewEncoder,
    values: torch.Tensor,
    labels: np.ndarray,
    class_ids: list[int],
    rng: np.random.Generator,
    pseudo_unknown_local: int,
    rejection_weight: float,
) -> tuple[torch.Tensor, float, float]:
    support, support_labels, query, query_labels = sample_episode(labels, class_ids, rng)
    support_t = torch.from_numpy(support_labels).to(values.device)
    query_t = torch.from_numpy(query_labels).to(values.device)
    support_embedding = model(values[support])
    query_embedding = model(values[query])
    prototypes = _prototypes(support_embedding, support_t, len(class_ids))
    scale = model.logit_scale.exp().clamp(max=100.0)
    logits = scale * query_embedding @ prototypes.T
    classification_loss = F.cross_entropy(logits, query_t)

    gallery_mask = torch.arange(len(class_ids), device=values.device) != pseudo_unknown_local
    gallery = prototypes[gallery_mask]
    similarity = query_embedding @ gallery.T
    known_confidence = similarity[query_t != pseudo_unknown_local].max(dim=1).values
    unknown_confidence = similarity[query_t == pseudo_unknown_local].max(dim=1).values
    # Rank pseudo-unknown sessions below known sessions without fixing a raw
    # threshold that would be brittle across outer held-out classes.
    rejection_loss = F.softplus(unknown_confidence.mean() - known_confidence.mean() + 0.20)
    loss = classification_loss + rejection_weight * rejection_loss
    return loss, float(classification_loss.detach()), float(rejection_loss.detach())


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=False)


def _fold_seed(seed: int, holdout: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}|{holdout}|episodic".encode()).digest()[:4], "little")


def _encode(model: MultiViewEncoder, values: np.ndarray) -> np.ndarray:
    model.eval()
    rows: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(values), 1024):
            rows.append(model(torch.from_numpy(values[start : start + 1024]).cuda()).cpu().numpy())
    return np.concatenate(rows)


def _fit_fold(
    window_x: np.ndarray,
    window_y: np.ndarray,
    window_sessions: np.ndarray,
    partitions: np.ndarray,
    names: np.ndarray,
    holdout: int,
    seed: int,
    output_dir: Path,
    epochs: int,
    episodes_per_epoch: int,
    patience: int,
    rejection_weight: float,
) -> tuple[MultiViewEncoder, StandardScaler, dict[str, object]]:
    seen_classes = [value for value in range(len(CLASS_NAMES)) if value != holdout]
    train_windows = (partitions == "train") & np.isin(window_y, seen_classes)
    development_windows = (partitions != "final_query")
    scaler = StandardScaler().fit(window_x[train_windows])
    transformed = scaler.transform(window_x[development_windows]).astype(np.float32)
    dev_y = window_y[development_windows]
    dev_sessions = window_sessions[development_windows]
    dev_partitions = partitions[development_windows]
    session_x, session_y, session_ids = _session_prototypes(transformed, dev_y, dev_sessions)
    session_partition = np.asarray(
        [np.unique(dev_partitions[dev_sessions == session])[0] for session in session_ids]
    )
    train = (session_partition == "train") & np.isin(session_y, seen_classes)
    validation = (session_partition == "validation") & np.isin(session_y, seen_classes)
    train_x = torch.from_numpy(session_x[train]).cuda()
    train_y = session_y[train]
    model = MultiViewEncoder(build_view_masks(names)).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_model.pt"
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        classification_losses: list[float] = []
        rejection_losses: list[float] = []
        for episode in range(episodes_per_epoch):
            optimizer.zero_grad(set_to_none=True)
            loss, classification_loss, rejection_loss = _episode_loss(
                model,
                train_x,
                train_y,
                seen_classes,
                rng,
                pseudo_unknown_local=(epoch + episode) % len(seen_classes),
                rejection_weight=rejection_weight,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            classification_losses.append(classification_loss)
            rejection_losses.append(rejection_loss)
        embedded = _encode(model, session_x)
        train_prototypes = np.stack(
            [np.mean(embedded[train & (session_y == class_id)], axis=0) for class_id in seen_classes]
        )
        predicted = np.asarray(seen_classes)[
            np.argmax(embedded[validation] @ train_prototypes.T, axis=1)
        ]
        macro_f1 = float(
            f1_score(session_y[validation], predicted, labels=seen_classes, average="macro", zero_division=0)
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "classification_loss": float(np.mean(classification_losses)),
            "rejection_loss": float(np.mean(rejection_losses)),
            "seen_validation_session_macro_f1": macro_f1,
        }
        history.append(row)
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(checkpoint, map_location="cuda", weights_only=True))
    joblib.dump(scaler, output_dir / "scaler.joblib")
    metadata = {
        "holdout_class": CLASS_NAMES[holdout],
        "seen_classes": [CLASS_NAMES[value] for value in seen_classes],
        "seed": seed,
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "seen_validation_session_macro_f1": best_f1,
        "elapsed_seconds": time.perf_counter() - started,
        "history": history,
        "rejection_weight": rejection_weight,
    }
    (output_dir / "training.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return model, scaler, metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--episodes-per-epoch", type=int, default=30)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--support-draws", type=int, default=20)
    parser.add_argument("--rejection-weight", type=float, default=0.5)
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
        raise ValueError("Development episodic run cannot receive final_query")
    results: list[dict[str, object]] = []

    for holdout in range(len(CLASS_NAMES)):
        seed = _fold_seed(args.seed, holdout)
        _seed(seed)
        fold_dir = args.output_dir / CLASS_NAMES[holdout]
        model, scaler, training = _fit_fold(
            x,
            y,
            sessions,
            partitions,
            names,
            holdout,
            seed,
            fold_dir,
            args.epochs,
            args.episodes_per_epoch,
            args.patience,
            args.rejection_weight,
        )
        transformed = scaler.transform(x).astype(np.float32)
        represented = _encode(model, transformed)
        session_x, session_y, session_ids = _session_prototypes(represented, y, sessions)
        session_partition = np.asarray(
            [np.unique(partitions[sessions == session])[0] for session in session_ids]
        )
        seen_classes = [value for value in range(len(CLASS_NAMES)) if value != holdout]
        train = (session_partition == "train") & np.isin(session_y, seen_classes)
        calibration = (session_partition == "calibration") & np.isin(session_y, seen_classes)
        validation = session_partition == "validation"
        support = (session_partition == "support") & (session_y == holdout)

        for method in ("knn_euclidean_3", "mahalanobis"):
            gallery = DensityGallery(session_x[train], session_y[train], seen_classes, method)
            calibration_scores = gallery.score(session_x[calibration])
            _, known_confidence = _confidence_prediction(calibration_scores, gallery.class_ids)
            pseudo_unknown_confidence: list[np.ndarray] = []
            for pseudo_class in seen_classes:
                reduced_classes = [value for value in seen_classes if value != pseudo_class]
                reduced = DensityGallery(session_x[train], session_y[train], reduced_classes, method)
                selected = calibration & (session_y == pseudo_class)
                scores = reduced.score(session_x[selected])
                _, confidence = _confidence_prediction(scores, reduced.class_ids)
                pseudo_unknown_confidence.append(confidence)
            thresholds = calibrate_rejection_threshold(
                known_confidence, np.concatenate(pseudo_unknown_confidence), target_known_acceptance=0.95
            )
            scores = gallery.score(session_x[validation])
            predicted, confidence = _confidence_prediction(scores, gallery.class_ids)
            true = session_y[validation]
            is_known = true != holdout
            correct = predicted == true
            pre = {
                mode: open_set_metrics(confidence, is_known, correct, threshold=float(threshold))
                for mode, threshold in (
                    ("balanced", thresholds["balanced_threshold"]),
                    ("known_acceptance_95", thresholds["known_acceptance_threshold"]),
                )
            }
            post = None
            if method == "knn_euclidean_3":
                shots: dict[str, object] = {}
                support_indices = np.flatnonzero(support)
                for shot in (1, 3, 5):
                    draws: list[dict[str, object]] = []
                    for draw in range(args.support_draws):
                        rng = np.random.default_rng(_draw_seed(args.seed, holdout, shot, draw))
                        selected = np.sort(rng.choice(support_indices, size=shot, replace=False))
                        enrolled = DensityGallery(
                            np.concatenate((session_x[train], session_x[selected])),
                            np.concatenate((session_y[train], session_y[selected])),
                            seen_classes + [holdout],
                            method,
                        )
                        enrolled_scores = enrolled.score(session_x[validation])
                        enrolled_predicted, _ = _confidence_prediction(enrolled_scores, enrolled.class_ids)
                        metrics = _post_enrollment_metrics(true, enrolled_predicted, holdout)
                        draws.append(
                            {
                                "draw": draw,
                                "support_sessions": session_ids[selected].tolist(),
                                **metrics,
                            }
                        )
                    shots[str(shot)] = {
                        "draws": draws,
                        "enrollment_h_mean": float(np.mean([row["enrollment_h"] for row in draws])),
                        "enrollment_h_min": float(np.min([row["enrollment_h"] for row in draws])),
                        "base_accuracy_mean": float(np.mean([row["base_accuracy"] for row in draws])),
                        "enrolled_recall_mean": float(np.mean([row["enrolled_recall"] for row in draws])),
                    }
                post = shots
            results.append(
                {
                    "holdout_class_id": holdout,
                    "holdout_class": CLASS_NAMES[holdout],
                    "method": method,
                    "training": training,
                    "calibration": thresholds,
                    "pre_enrollment": pre,
                    "post_enrollment": post,
                }
            )
            post_text = "" if post is None else f" postH@5={post['5']['enrollment_h_mean']:.3f}"
            print(
                f"[{CLASS_NAMES[holdout]} {method}] AUROC={pre['known_acceptance_95']['unknown_auroc']:.3f} "
                f"H95={pre['known_acceptance_95']['detection_h']:.3f}{post_text}",
                flush=True,
            )

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in results:
        grouped[row["method"]].append(row)
    summary: dict[str, dict[str, float]] = {}
    for method, group in grouped.items():
        summary[method] = {
            "unknown_auroc_mean": float(np.mean([row["pre_enrollment"]["known_acceptance_95"]["unknown_auroc"] for row in group])),
            "known_acceptance_mean": float(np.mean([row["pre_enrollment"]["known_acceptance_95"]["known_acceptance"] for row in group])),
            "unknown_recall_mean": float(np.mean([row["pre_enrollment"]["known_acceptance_95"]["unknown_recall"] for row in group])),
            "detection_h_mean": float(np.mean([row["pre_enrollment"]["known_acceptance_95"]["detection_h"] for row in group])),
            "detection_h_worst_holdout": float(np.min([row["pre_enrollment"]["known_acceptance_95"]["detection_h"] for row in group])),
            "oscr_mean": float(np.mean([row["pre_enrollment"]["known_acceptance_95"]["oscr"] for row in group])),
        }
        post_rows = [row for row in group if row["post_enrollment"] is not None]
        if post_rows:
            summary[method].update(
                {
                    "post_session_h_1shot_mean": float(np.mean([row["post_enrollment"]["1"]["enrollment_h_mean"] for row in post_rows])),
                    "post_session_h_5shot_mean": float(np.mean([row["post_enrollment"]["5"]["enrollment_h_mean"] for row in post_rows])),
                    "post_session_h_5shot_worst_draw": float(np.min([row["post_enrollment"]["5"]["enrollment_h_min"] for row in post_rows])),
                }
            )
    payload = {
        "protocol": "multi-view episodic session support/query learning with pseudo-unknown ranking",
        "seed": args.seed,
        "rejection_weight": args.rejection_weight,
        "final_query_used": False,
        "summary": summary,
        "fold_results": results,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "development_results.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
