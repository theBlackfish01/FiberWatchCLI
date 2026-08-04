from __future__ import annotations

from itertools import combinations
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Sequence

import click
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler

from .model_functions.zero_shot import ZeroShotClassifier, require_cuda
from .zero_shot_data import INPUT_COLUMNS, build_outer_fold, file_sha256, load_fault_prototypes
from .zero_shot_training import (
    TrainingConfig,
    apply_seen_penalty,
    choose_seen_penalty,
    compute_gzsl_metrics,
    config_dict,
    encode_fault_prototypes,
    fit_seen_scaler,
    gpu_metadata,
    predict_scores,
    save_json,
    train_zero_shot_model,
    transform_frame,
)


DEFAULT_TEXT_MODEL = "sentence-transformers/all-mpnet-base-v2"


def fault_pairs() -> list[tuple[int, int]]:
    return list(combinations(range(1, 8), 2))


def _class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    per_class: dict[str, dict[str, float | int]] = {}
    for class_id in sorted(set(y_true.tolist())):
        mask = y_true == class_id
        p, r, class_f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=[class_id], average=None, zero_division=0
        )
        per_class[str(class_id)] = {
            "precision": float(p[0]),
            "recall": float(r[0]),
            "f1": float(class_f1[0]),
            "support": int(support[0]),
            "accuracy": float((y_pred[mask] == class_id).mean()),
        }
    return {
        "accuracy": float((y_true == y_pred).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "per_class": per_class,
    }


def _predict_from_columns(scores: torch.Tensor, class_ids: Sequence[int]) -> np.ndarray:
    return np.asarray([class_ids[index] for index in scores.argmax(1).tolist()], dtype=np.int64)


def _save_scaler(path: Path, scaler) -> None:
    save_json(
        path,
        {
            "feature_names": INPUT_COLUMNS,
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
            "n_features_in": int(scaler.n_features_in_),
        },
    )


def _split_manifest(fold) -> dict[str, object]:
    def entry(frame: pd.DataFrame) -> dict[str, object]:
        return {
            "rows": int(len(frame)),
            "classes": {str(int(key)): int(value) for key, value in frame["Class"].value_counts().sort_index().items()},
            "group_count": int(frame["_input_group"].nunique()),
            "groups_sha256": __import__("hashlib").sha256(
                "\n".join(sorted(frame["_input_group"].unique())).encode("utf-8")
            ).hexdigest(),
        }

    return {
        "holdout": list(fold.holdout),
        "feature_names": INPUT_COLUMNS,
        "train": entry(fold.train),
        "validation": entry(fold.validation),
        "seen_test": entry(fold.seen_test),
        "unseen_test": entry(fold.unseen_test),
    }


def run_fold(
    *,
    data_path: Path,
    prototype_path: Path,
    out_dir: Path,
    holdout: tuple[int, int],
    device: torch.device,
    text_model: str,
    config: TrainingConfig,
) -> dict[str, object]:
    device = require_cuda(str(device))
    frame = pd.read_csv(data_path)
    fold = build_outer_fold(frame, holdout=holdout, seed=config.seed)
    prototypes_spec = load_fault_prototypes(prototype_path)
    prototype_embeddings = encode_fault_prototypes(prototypes_spec, model_name=text_model, device=device)
    scaler = fit_seen_scaler(fold)
    train_x, train_y = transform_frame(fold.train, scaler)
    val_x, val_y = transform_frame(fold.validation, scaler)
    seen_ids = sorted(set(range(8)) - set(holdout))

    calibration_class = min(class_id for class_id in seen_ids if class_id != 0)
    inner_seen = [class_id for class_id in seen_ids if class_id != calibration_class]
    inner_train_mask = train_y != calibration_class
    inner_val_mask = val_y != calibration_class
    inner_model, inner_training = train_zero_shot_model(
        train_x[inner_train_mask],
        train_y[inner_train_mask],
        val_x[inner_val_mask],
        val_y[inner_val_mask],
        prototype_embeddings,
        seen_class_ids=inner_seen,
        device=device,
        config=config,
    )
    calibration_candidates = [*inner_seen, calibration_class]
    calibration_scores_all = predict_scores(inner_model, val_x, prototype_embeddings, device=device)
    calibration_scores = calibration_scores_all[:, calibration_candidates]
    gamma, calibration_rows = choose_seen_penalty(
        calibration_scores,
        val_y.numpy(),
        seen_class_ids=set(inner_seen),
        candidate_class_ids=calibration_candidates,
    )
    del inner_model
    torch.cuda.empty_cache()

    model, training = train_zero_shot_model(
        train_x,
        train_y,
        val_x,
        val_y,
        prototype_embeddings,
        seen_class_ids=seen_ids,
        device=device,
        config=config,
    )

    unseen_x, unseen_y = transform_frame(fold.unseen_test, scaler)
    unseen_scores_all = predict_scores(model, unseen_x, prototype_embeddings, device=device)
    zsl_scores = unseen_scores_all[:, list(holdout)]
    zsl_pred = _predict_from_columns(zsl_scores, list(holdout))
    zsl_true = unseen_y.numpy()
    zsl_metrics = _class_metrics(zsl_true, zsl_pred)

    seen_x, seen_y = transform_frame(fold.seen_test, scaler)
    gzsl_x = torch.cat([seen_x, unseen_x])
    gzsl_true = np.concatenate([seen_y.numpy(), zsl_true])
    gzsl_scores = predict_scores(model, gzsl_x, prototype_embeddings, device=device)
    adjusted = apply_seen_penalty(
        gzsl_scores,
        gamma,
        seen_class_ids=set(seen_ids),
        candidate_class_ids=list(range(8)),
    )
    gzsl_pred = adjusted.argmax(1).numpy()
    gzsl_metrics = compute_gzsl_metrics(
        y_true=gzsl_true,
        y_pred=gzsl_pred,
        seen_class_ids=set(seen_ids),
        unseen_class_ids=set(holdout),
    )

    fold_dir = out_dir / f"fold_{holdout[0]:02d}_{holdout[1]:02d}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), fold_dir / "checkpoint.pt")
    torch.save(prototype_embeddings.detach().cpu(), fold_dir / "prototype_embeddings.pt")
    _save_scaler(fold_dir / "scaler.json", scaler)
    manifest = _split_manifest(fold)
    manifest_path = fold_dir / "split_manifest.json"
    save_json(manifest_path, manifest)
    save_json(fold_dir / "calibration.json", {"class": calibration_class, "gamma": gamma, "curve": calibration_rows, "inner_training": inner_training})
    metrics = {"holdout": list(holdout), "zsl": zsl_metrics, "gzsl": gzsl_metrics}
    save_json(fold_dir / "metrics.json", metrics)
    np.savetxt(fold_dir / "confusion_zsl.csv", confusion_matrix(zsl_true, zsl_pred, labels=list(holdout)), delimiter=",", fmt="%d")
    np.savetxt(fold_dir / "confusion_gzsl.csv", confusion_matrix(gzsl_true, gzsl_pred, labels=list(range(8))), delimiter=",", fmt="%d")
    top2 = adjusted.topk(2, dim=1)
    predictions = pd.DataFrame(
        {
            "row_index": np.concatenate([fold.seen_test.index.to_numpy(), fold.unseen_test.index.to_numpy()]),
            "true_class": gzsl_true,
            "predicted_class": gzsl_pred,
            "seen_status": ["seen"] * len(seen_y) + ["unseen"] * len(unseen_y),
            "top1_score": adjusted.max(1).values.numpy(),
            "top2_class": top2.indices[:, 1].numpy(),
            "top2_score": top2.values[:, 1].numpy(),
            "score_margin": (top2.values[:, 0] - top2.values[:, 1]).numpy(),
            "fold_id": f"{holdout[0]:02d}_{holdout[1]:02d}",
        }
    )
    predictions.to_csv(fold_dir / "predictions.csv", index=False)
    feature_signature = hashlib.sha256(json.dumps(INPUT_COLUMNS, separators=(",", ":")).encode()).hexdigest()
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        git_commit = None
    metadata = {
        "schema_version": 1,
        "holdout": list(holdout),
        "seen_class_ids": seen_ids,
        "feature_names": INPUT_COLUMNS,
        "feature_signature": feature_signature,
        "forbidden_features_disabled": True,
        "text_model": text_model,
        "sentence_transformers_version": importlib.metadata.version("sentence-transformers"),
        "embedding_dim": int(prototype_embeddings.shape[-1]),
        "prototype_sha256": file_sha256(prototype_path),
        "prototype_path": str(prototype_path.resolve()),
        "dataset_sha256": file_sha256(data_path),
        "dataset_path": str(data_path.resolve()),
        "dataset_size_bytes": data_path.stat().st_size,
        "split_manifest_sha256": file_sha256(manifest_path),
        "git_commit": git_commit,
        "training_config": config_dict(config),
        "training": training,
        "gamma": gamma,
        "gpu": gpu_metadata(device),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }
    save_json(fold_dir / "metadata.json", metadata)
    return metrics


@click.group()
def cli() -> None:
    """CUDA-only semantic zero-shot learning for OTDR fault classes."""


def _common_options(function):
    options = [
        click.option("--data", "data_path", type=click.Path(path_type=Path, exists=True), default=Path("src/data/OTDR_DATA.csv"), show_default=True),
        click.option("--prototypes", "prototype_path", type=click.Path(path_type=Path, exists=True), default=Path("src/corpus/zero_shot_fault_prototypes.json"), show_default=True),
        click.option("--out-dir", type=click.Path(path_type=Path), default=Path("models/zero_shot"), show_default=True),
        click.option("--device", default="cuda:0", show_default=True),
        click.option("--text-model", default=DEFAULT_TEXT_MODEL, show_default=True),
        click.option("--epochs", default=40, type=click.IntRange(1), show_default=True),
        click.option("--batch-size", default=256, type=click.IntRange(2), show_default=True),
        click.option("--learning-rate", default=3e-4, type=click.FloatRange(min=0, min_open=True), show_default=True),
        click.option("--seed", default=42, type=int, show_default=True),
    ]
    for option in reversed(options):
        function = option(function)
    return function


@cli.command("train-fold")
@click.option("--holdout", type=click.IntRange(1, 7), multiple=True, required=True)
@_common_options
def train_fold_command(holdout, data_path, prototype_path, out_dir, device, text_model, epochs, batch_size, learning_rate, seed) -> None:
    if len(set(holdout)) != 2:
        raise click.ClickException("Provide exactly two distinct --holdout fault classes.")
    try:
        cuda = require_cuda(device)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    config = TrainingConfig(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed)
    metrics = run_fold(data_path=data_path, prototype_path=prototype_path, out_dir=out_dir, holdout=tuple(sorted(holdout)), device=cuda, text_model=text_model, config=config)
    click.echo(json.dumps(metrics, indent=2))


@cli.command("evaluate-fold")
@click.option("--fold-dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--data", "data_path", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--prototypes", "prototype_path", type=click.Path(path_type=Path, exists=True), default=None)
def evaluate_fold_command(fold_dir: Path, device: str, data_path: Path | None, prototype_path: Path | None) -> None:
    try:
        cuda = require_cuda(device)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    metadata = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
    data_path = data_path or Path(metadata["dataset_path"])
    prototype_path = prototype_path or Path(metadata["prototype_path"])
    if file_sha256(data_path) != metadata["dataset_sha256"]:
        raise click.ClickException("Dataset hash does not match the fold metadata.")
    if file_sha256(prototype_path) != metadata["prototype_sha256"]:
        raise click.ClickException("Prototype hash does not match the fold metadata.")
    if file_sha256(fold_dir / "split_manifest.json") != metadata["split_manifest_sha256"]:
        raise click.ClickException("Split manifest hash does not match the fold metadata.")
    prototypes = torch.load(fold_dir / "prototype_embeddings.pt", map_location=cuda, weights_only=True)
    model = ZeroShotClassifier(embedding_dim=int(metadata["embedding_dim"])).to(cuda)
    model.load_state_dict(torch.load(fold_dir / "checkpoint.pt", map_location=cuda, weights_only=True))
    scaler_payload = json.loads((fold_dir / "scaler.json").read_text(encoding="utf-8"))
    scaler = StandardScaler()
    scaler.mean_ = np.asarray(scaler_payload["mean"], dtype=float)
    scaler.scale_ = np.asarray(scaler_payload["scale"], dtype=float)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = int(scaler_payload["n_features_in"])
    frame = pd.read_csv(data_path)
    holdout = tuple(int(value) for value in metadata["holdout"])
    fold = build_outer_fold(frame, holdout=holdout, seed=int(metadata["training_config"]["seed"]))
    unseen_x, unseen_y = transform_frame(fold.unseen_test, scaler)
    unseen_scores = predict_scores(model, unseen_x, prototypes, device=cuda)
    zsl_pred = _predict_from_columns(unseen_scores[:, list(holdout)], list(holdout))
    zsl_true = unseen_y.numpy()
    seen_x, seen_y = transform_frame(fold.seen_test, scaler)
    gzsl_scores = predict_scores(model, torch.cat([seen_x, unseen_x]), prototypes, device=cuda)
    seen_ids = set(metadata["seen_class_ids"])
    adjusted = apply_seen_penalty(
        gzsl_scores,
        float(metadata["gamma"]),
        seen_class_ids=seen_ids,
        candidate_class_ids=list(range(8)),
    )
    gzsl_true = np.concatenate([seen_y.numpy(), zsl_true])
    recomputed = {
        "holdout": list(holdout),
        "zsl": _class_metrics(zsl_true, zsl_pred),
        "gzsl": compute_gzsl_metrics(
            y_true=gzsl_true,
            y_pred=adjusted.argmax(1).numpy(),
            seen_class_ids=seen_ids,
            unseen_class_ids=set(holdout),
        ),
    }
    saved = json.loads((fold_dir / "metrics.json").read_text(encoding="utf-8"))
    if json.dumps(recomputed, sort_keys=True) != json.dumps(saved, sort_keys=True):
        raise click.ClickException("Recomputed metrics do not match metrics.json.")
    click.echo(json.dumps({"evaluated": True, "device": str(cuda), **recomputed}, indent=2))


@cli.command("benchmark")
@click.option("--force", is_flag=True, help="Retrain folds even when metrics already exist.")
@_common_options
def benchmark_command(force, data_path, prototype_path, out_dir, device, text_model, epochs, batch_size, learning_rate, seed) -> None:
    try:
        cuda = require_cuda(device)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc
    config = TrainingConfig(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, seed=seed)
    results: list[dict[str, object]] = []
    for pair in fault_pairs():
        metrics_path = out_dir / f"fold_{pair[0]:02d}_{pair[1]:02d}" / "metrics.json"
        if metrics_path.exists() and not force:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            click.echo(f"[ZSL] Training fold {pair[0]}-{pair[1]} on {cuda}")
            metrics = run_fold(data_path=data_path, prototype_path=prototype_path, out_dir=out_dir, holdout=pair, device=cuda, text_model=text_model, config=config)
        results.append(metrics)
    def distribution(values):
        array = np.asarray(values, dtype=float)
        return {
            "mean": float(array.mean()),
            "std": float(array.std()),
            "median": float(np.median(array)),
            "min": float(array.min()),
            "max": float(array.max()),
        }

    per_fault = {}
    for class_id in range(1, 8):
        containing = [row for row in results if class_id in row["holdout"]]
        per_fault[str(class_id)] = {
            "fold_count": len(containing),
            "zsl_recall": distribution([row["zsl"]["per_class"][str(class_id)]["recall"] for row in containing]),
            "gzsl_unseen_accuracy": distribution([row["gzsl"]["unseen_accuracy"] for row in containing]),
        }
    summary = {
        "fold_count": len(results),
        "zsl_accuracy": distribution([row["zsl"]["accuracy"] for row in results]),
        "gzsl_seen_accuracy": distribution([row["gzsl"]["seen_accuracy"] for row in results]),
        "gzsl_unseen_accuracy": distribution([row["gzsl"]["unseen_accuracy"] for row in results]),
        "gzsl_harmonic_mean": distribution([row["gzsl"]["harmonic_mean"] for row in results]),
        "per_fault": per_fault,
        "folds": results,
    }
    save_json(out_dir / "benchmark_summary.json", summary)
    pd.DataFrame([{"holdout": "-".join(map(str, row["holdout"])), **{f"zsl_{k}": v for k, v in row["zsl"].items()}, **{f"gzsl_{k}": v for k, v in row["gzsl"].items()}} for row in results]).to_csv(out_dir / "benchmark_summary.csv", index=False)
    click.echo(json.dumps({key: value for key, value in summary.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    cli()
