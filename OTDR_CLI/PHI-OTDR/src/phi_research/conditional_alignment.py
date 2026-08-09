"""CUDA pilot for class-conditional acquisition-date alignment.

The regularizer compares normalized embedding centers only between date groups
of the same event class.  It therefore avoids the invalid assumption that the
heavily class-confounded date/source groups should be aligned unconditionally.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
import torch.nn.functional as functional
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .embedding_gallery import FeatureEncoder, _encode, _seed, _session_sampler
from .metrics import aggregate_session_predictions, classification_metrics
from .session_distribution import feature_masks


def conditional_center_alignment(
    embedding: torch.Tensor, labels: torch.Tensor, domains: torch.Tensor
) -> torch.Tensor:
    """Penalize between-domain centers within class, never across classes."""
    normalized = functional.normalize(embedding.float(), dim=1)
    penalties: list[torch.Tensor] = []
    for class_id in torch.unique(labels):
        selected_class = labels == class_id
        centers: list[torch.Tensor] = []
        for domain_id in torch.unique(domains[selected_class]):
            selected = selected_class & (domains == domain_id)
            if int(torch.sum(selected)) >= 2:
                centers.append(torch.mean(normalized[selected], dim=0))
        if len(centers) >= 2:
            stacked = torch.stack(centers)
            penalties.append(torch.mean((stacked - torch.mean(stacked, dim=0)) ** 2))
    if not penalties:
        return embedding.sum() * 0.0
    return torch.mean(torch.stack(penalties))


def _metadata_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | int]:
    return {
        "sessions": int(len(y_true)),
        "classes": int(len(set(y_true.tolist()))),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }


def _session_table(
    embedding: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    partitions: np.ndarray,
    date_tokens: np.ndarray,
    source_tokens: np.ndarray,
) -> dict[str, np.ndarray]:
    rows: dict[str, list[object]] = {
        "embedding": [], "label": [], "session": [], "partition": [],
        "date_token": [], "source_token": [], "date_source": [],
    }
    for session_id in sorted(set(sessions.tolist())):
        selected = sessions == session_id
        rows["embedding"].append(np.mean(embedding[selected], axis=0))
        rows["label"].append(int(labels[selected][0]))
        rows["session"].append(session_id)
        rows["partition"].append(partitions[selected][0])
        rows["date_token"].append(date_tokens[selected][0])
        rows["source_token"].append(source_tokens[selected][0])
        rows["date_source"].append(f"{date_tokens[selected][0]}|{source_tokens[selected][0]}")
    return {
        key: np.stack(values) if key == "embedding" else np.asarray(values)
        for key, values in rows.items()
    }


def _embedding_metadata_probe(
    table: dict[str, np.ndarray], train: np.ndarray, validation: np.ndarray, target: str
) -> dict[str, object]:
    known = set(table[target][train].tolist())
    evaluable = validation & np.asarray([value in known for value in table[target]])
    unseen = sorted(set(table[target][validation]) - known)
    if not np.any(evaluable):
        return {"target": target, "evaluable": False, "unseen_validation_labels": unseen}
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    C=1.0, class_weight="balanced", max_iter=3000, random_state=20260805
                ),
            ),
        ]
    )
    model.fit(table["embedding"][train], table[target][train])
    predicted = model.predict(table["embedding"][evaluable])
    return {
        "target": target,
        "evaluable": True,
        "validation": _metadata_metrics(table[target][evaluable], predicted),
        "excluded_unseen_validation_sessions": int(np.sum(validation) - np.sum(evaluable)),
        "unseen_validation_labels": unseen,
    }


def choose_alignment_weight(results: list[dict[str, object]]) -> dict[str, object]:
    baseline = next(row for row in results if float(row["alignment_weight"]) == 0.0)
    baseline_class = float(baseline["validation"]["session_metrics"]["macro_f1"])
    baseline_date = float(baseline["metadata_probe"]["date_token"]["validation"]["macro_f1"])
    candidates: list[dict[str, object]] = []
    for row in results:
        weight = float(row["alignment_weight"])
        if weight == 0.0:
            continue
        class_score = float(row["validation"]["session_metrics"]["macro_f1"])
        date_score = float(row["metadata_probe"]["date_token"]["validation"]["macro_f1"])
        comparison = {
            "alignment_weight": weight,
            "session_macro_f1_change": class_score - baseline_class,
            "date_probe_macro_f1_reduction": baseline_date - date_score,
        }
        comparison["passes_gate"] = (
            comparison["session_macro_f1_change"] >= -0.02
            and comparison["date_probe_macro_f1_reduction"] >= 0.05
        )
        candidates.append(comparison)
    passing = [row for row in candidates if row["passes_gate"]]
    selected = max(
        passing,
        key=lambda row: (row["date_probe_macro_f1_reduction"], row["session_macro_f1_change"]),
        default=None,
    )
    return {
        "gate": "session macro-F1 drop <=0.02 and date-probe macro-F1 reduction >=0.05",
        "baseline_session_macro_f1": baseline_class,
        "baseline_date_probe_macro_f1": baseline_date,
        "candidates": candidates,
        "decision": "continue" if selected is not None else "stop",
        "selected_alignment_weight": None if selected is None else selected["alignment_weight"],
    }


def _fit_one(
    x: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    partitions: np.ndarray,
    date_tokens: np.ndarray,
    source_tokens: np.ndarray,
    feature_mask: np.ndarray,
    *,
    alignment_weight: float,
    seed: int,
    epochs: int,
    patience: int,
    output_dir: Path,
) -> dict[str, object]:
    _seed(seed)
    device = torch.device("cuda")
    train = partitions == "source_train"
    validation = partitions == "source_validation"
    scaler = StandardScaler().fit(x[train][:, feature_mask])
    transformed_train = scaler.transform(x[train][:, feature_mask]).astype(np.float32)
    transformed_validation = scaler.transform(x[validation][:, feature_mask]).astype(np.float32)
    domain_names = sorted(set(date_tokens[train].tolist()))
    domain_map = {value: index for index, value in enumerate(domain_names)}
    train_domains = np.asarray([domain_map[value] for value in date_tokens[train]], dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(transformed_train),
        torch.from_numpy(labels[train].astype(np.int64)),
        torch.from_numpy(train_domains),
    )
    sampler = _session_sampler(labels[train], sessions[train], seed)
    loader = DataLoader(dataset, batch_size=512, sampler=sampler, num_workers=0, pin_memory=True)
    model = FeatureEncoder(transformed_train.shape[1], 6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    scaler_amp = torch.amp.GradScaler("cuda")
    best_f1 = -1.0
    best_epoch = 0
    stale = 0
    history: list[dict[str, float]] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = output_dir / "best_model.pt"
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        losses: list[float] = []
        class_losses: list[float] = []
        alignment_losses: list[float] = []
        for values, targets, domains in loader:
            values = values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            domains = domains.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embedding, logits = model(values)
                class_loss = criterion(logits, targets)
                alignment_loss = conditional_center_alignment(embedding, targets, domains)
                loss = class_loss + alignment_weight * alignment_loss
            scaler_amp.scale(loss).backward()
            scaler_amp.step(optimizer)
            scaler_amp.update()
            losses.append(float(loss.detach().cpu()))
            class_losses.append(float(class_loss.detach().cpu()))
            alignment_losses.append(float(alignment_loss.detach().cpu()))
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _, logits = model(torch.from_numpy(transformed_validation).to(device))
        predicted = logits.argmax(dim=1).cpu().numpy()
        macro_f1 = float(f1_score(labels[validation], predicted, average="macro", zero_division=0))
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "class_loss": float(np.mean(class_losses)),
                "conditional_alignment_loss": float(np.mean(alignment_losses)),
                "validation_window_macro_f1": macro_f1,
            }
        )
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_epoch = epoch
            stale = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= patience:
                break
    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    source = train | validation
    transformed_source = scaler.transform(x[source][:, feature_mask]).astype(np.float32)
    inference_started = time.perf_counter()
    source_embedding = _encode(model, transformed_source, device)
    inference_seconds = time.perf_counter() - inference_started
    source_labels = labels[source]
    source_sessions = sessions[source]
    source_partitions = partitions[source]
    source_dates = date_tokens[source]
    source_tokens_selected = source_tokens[source]
    model.eval()
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _, validation_logits = model(torch.from_numpy(transformed_validation).to(device))
        validation_prob = torch.softmax(validation_logits.float(), dim=1).cpu().numpy()
    validation_pred = np.argmax(validation_prob, axis=1)
    session_true, session_pred, _ = aggregate_session_predictions(
        labels[validation], sessions[validation], probabilities=validation_prob
    )
    table = _session_table(
        source_embedding,
        source_labels,
        source_sessions,
        source_partitions,
        source_dates,
        source_tokens_selected,
    )
    session_train = table["partition"] == "source_train"
    session_validation = table["partition"] == "source_validation"
    metadata_probe = {
        target: _embedding_metadata_probe(table, session_train, session_validation, target)
        for target in ("date_token", "source_token", "date_source")
    }
    joblib.dump(scaler, output_dir / "scaler.joblib")
    payload = {
        "alignment_weight": alignment_weight,
        "seed": seed,
        "feature_count": int(np.sum(feature_mask)),
        "domain_field": "date_token",
        "domain_count": len(domain_names),
        "best_epoch": best_epoch,
        "epochs_completed": len(history),
        "training_seconds": training_seconds,
        "inference_seconds": inference_seconds,
        "history": history,
        "validation": {
            "window_metrics": classification_metrics(labels[validation], validation_pred),
            "session_metrics": classification_metrics(session_true, session_pred),
        },
        "metadata_probe": metadata_probe,
        "cuda": {
            "required": True,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "precision": "float16 autocast with float32 conditional-center loss",
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        },
    }
    (output_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--view", choices=("dynamics", "full"), default="dynamics")
    parser.add_argument("--alignment-weights", nargs="+", type=float, default=[0.0, 0.01, 0.05, 0.20])
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=6)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for conditional-alignment training")
    bundle = np.load(args.features, allow_pickle=False)
    try:
        partitions = bundle["partitions"].astype(str)
        if "target_query" in set(partitions.tolist()):
            raise ValueError("Alignment pilot must not receive target-query features")
        x = bundle["features"].astype(np.float32)
        labels = bundle["labels"].astype(np.int64)
        sessions = bundle["sessions"].astype(str)
        date_tokens = bundle["date_tokens"].astype(str)
        source_tokens = bundle["source_tokens"].astype(str)
        mask = feature_masks(bundle["feature_names"].astype(str))[args.view]
    finally:
        bundle.close()
    results: list[dict[str, object]] = []
    for weight in args.alignment_weights:
        weight_name = str(weight).replace(".", "p")
        result = _fit_one(
            x,
            labels,
            sessions,
            partitions,
            date_tokens,
            source_tokens,
            mask,
            alignment_weight=weight,
            seed=args.seed,
            epochs=args.epochs,
            patience=args.patience,
            output_dir=args.output_dir / f"weight_{weight_name}",
        )
        results.append(result)
        print(
            f"[weight={weight:g}] sessionF1={result['validation']['session_metrics']['macro_f1']:.3f} "
            f"dateProbeF1={result['metadata_probe']['date_token']['validation']['macro_f1']:.3f} "
            f"time={result['training_seconds']:.1f}s",
            flush=True,
        )
    selection = choose_alignment_weight(results)
    payload = {
        "protocol": "CUDA source-era class-conditional date-alignment pilot",
        "final_query_used": False,
        "view": args.view,
        "seed": args.seed,
        "results": results,
        "selection": selection,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pilot_summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(selection, indent=2))


if __name__ == "__main__":
    main()
