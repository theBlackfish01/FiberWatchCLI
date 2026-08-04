from __future__ import annotations

from itertools import combinations
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from .model_functions.zero_shot import require_cuda
from .one_shot_experiment import (
    METHODS,
    OPERATING_POINTS,
    REGIMES,
    classify_saved_frame,
    encode_saved_references,
    recompute_saved_configuration,
    run_crossfit_fold,
    summarize_crossfit_benchmark,
    write_benchmark_tables,
)
from .one_shot_gallery import ReferenceGallery, attach_semantic_suggestions
from .one_shot_training import OneShotTrainingConfig
from .zero_shot_training import save_json


def fault_pairs() -> list[tuple[int, int]]:
    return list(combinations(range(1, 8), 2))


def _cuda(device: str) -> torch.device:
    try:
        return require_cuda(device)
    except (ValueError, RuntimeError) as exc:
        raise click.ClickException(str(exc)) from exc


def _common_options(function):
    options = [
        click.option("--data", "data_path", type=click.Path(path_type=Path, exists=True), default=Path("src/data/OTDR_DATA.csv"), show_default=True),
        click.option("--out-dir", type=click.Path(path_type=Path), default=Path("models/one_shot_crossfit"), show_default=True),
        click.option("--device", default="cuda:0", show_default=True),
        click.option("--epochs", default=40, type=click.IntRange(1), show_default=True),
        click.option("--batch-size", default=256, type=click.IntRange(2), show_default=True),
        click.option("--learning-rate", default=3e-4, type=click.FloatRange(min=0, min_open=True), show_default=True),
        click.option("--pair-count", default=16384, type=click.IntRange(2), show_default=True),
        click.option("--calibration-epochs", default=4, type=click.IntRange(1), show_default=True),
        click.option("--calibration-pair-count", default=4096, type=click.IntRange(2), show_default=True),
        click.option("--similarity-mode", type=click.Choice(["multi", "l1", "l2", "cosine", "product"]), default="multi", show_default=True),
        click.option("--support-draws", default=20, type=click.IntRange(1), show_default=True),
        click.option("--seed", default=42, type=int, show_default=True),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _training_config(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    pair_count: int,
    calibration_epochs: int,
    calibration_pair_count: int,
    similarity_mode: str,
    seed: int,
) -> OneShotTrainingConfig:
    return OneShotTrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        pair_count=pair_count,
        calibration_epochs=calibration_epochs,
        calibration_pair_count=calibration_pair_count,
        similarity_mode=similarity_mode,
        seed=seed,
    )


@click.group()
def cli() -> None:
    """CUDA-only cross-fitted one-shot learning for OTDR fault classes."""


@cli.command("train-fold")
@click.option("--holdout", type=click.IntRange(1, 7), multiple=True, required=True)
@_common_options
def train_fold_command(holdout, data_path, out_dir, device, epochs, batch_size, learning_rate, pair_count, calibration_epochs, calibration_pair_count, similarity_mode, support_draws, seed) -> None:
    if len(set(holdout)) != 2:
        raise click.ClickException("Provide exactly two distinct --holdout fault classes.")
    config = _training_config(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        pair_count=pair_count, calibration_epochs=calibration_epochs,
        calibration_pair_count=calibration_pair_count,
        similarity_mode=similarity_mode, seed=seed,
    )
    metrics = run_crossfit_fold(
        data_path=data_path,
        out_dir=out_dir,
        holdout=tuple(sorted(holdout)),
        device=_cuda(device),
        config=config,
        support_draws=support_draws,
    )
    click.echo(json.dumps(metrics, indent=2))


def _evaluation_options(function):
    options = [
        click.option("--fold-dir", type=click.Path(path_type=Path, exists=True), required=True),
        click.option("--data", "data_path", type=click.Path(path_type=Path, exists=True), default=None),
        click.option("--device", default="cuda:0", show_default=True),
        click.option("--regime", type=click.Choice(REGIMES), default="uniform_one_reference", show_default=True),
        click.option("--method", type=click.Choice(METHODS), default="learned", show_default=True),
        click.option("--operating-point", type=click.Choice(OPERATING_POINTS), default="balanced", show_default=True),
    ]
    for option in reversed(options):
        function = option(function)
    return function


def _saved_dataset(fold_dir: Path, data_path: Path | None) -> Path:
    if data_path is not None:
        return data_path
    metadata = json.loads((fold_dir / "metadata.json").read_text(encoding="utf-8"))
    return Path(metadata["dataset_path"])


@cli.command("evaluate-detection")
@_evaluation_options
def evaluate_detection_command(fold_dir, data_path, device, regime, method, operating_point) -> None:
    result = recompute_saved_configuration(
        fold_dir=fold_dir,
        data_path=_saved_dataset(fold_dir, data_path),
        regime=regime,
        method=method,
        operating_point=operating_point,
        device=_cuda(device),
        include_one_shot=False,
    )
    click.echo(json.dumps(result, indent=2))


@cli.command("evaluate-one-shot")
@_evaluation_options
def evaluate_one_shot_command(fold_dir, data_path, device, regime, method, operating_point) -> None:
    result = recompute_saved_configuration(
        fold_dir=fold_dir,
        data_path=_saved_dataset(fold_dir, data_path),
        regime=regime,
        method=method,
        operating_point=operating_point,
        device=_cuda(device),
        include_one_shot=True,
    )
    click.echo(json.dumps(result, indent=2))


@cli.command("enroll-reference")
@click.option("--fold-dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--class-id", type=click.IntRange(0, 7), required=True)
@click.option("--gallery-out", type=click.Path(path_type=Path), required=True)
@click.option("--regime", type=click.Choice(REGIMES), default="uniform_one_reference", show_default=True)
@click.option("--device", default="cuda:0", show_default=True)
def enroll_reference_command(fold_dir, input_path, class_id, gallery_out, regime, device) -> None:
    rows = pd.read_csv(input_path)
    embeddings = encode_saved_references(rows, fold_dir=fold_dir, device=_cuda(device))
    gallery_name = "gallery_uniform.pt" if regime == "uniform_one_reference" else "gallery_seen_rich.pt"
    gallery = ReferenceGallery.load(fold_dir / gallery_name)
    enrolled = gallery.enroll(embeddings, class_id=class_id, row_indices=torch.arange(len(rows)))
    enrolled.save(gallery_out)
    click.echo(json.dumps({"gallery": str(gallery_out), "enrolled_class": class_id, "enrolled_rows": len(rows)}, indent=2))


@cli.command("classify")
@click.option("--fold-dir", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--input", "input_path", type=click.Path(path_type=Path, exists=True), required=True)
@click.option("--gallery", "gallery_path", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--semantic-suggestions", type=click.Path(path_type=Path, exists=True), default=None)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--regime", type=click.Choice(REGIMES), default="uniform_one_reference", show_default=True)
@click.option("--method", type=click.Choice(METHODS), default="learned", show_default=True)
@click.option("--operating-point", type=click.Choice(OPERATING_POINTS), default="balanced", show_default=True)
def classify_command(fold_dir, input_path, gallery_path, semantic_suggestions, device, regime, method, operating_point) -> None:
    rows = pd.read_csv(input_path)
    output = classify_saved_frame(
        rows, fold_dir=fold_dir, gallery_path=gallery_path, regime=regime,
        method=method, operating_point=operating_point, device=_cuda(device),
    )
    if semantic_suggestions is not None:
        semantic = pd.read_csv(semantic_suggestions)
        if "predicted_class" not in semantic or len(semantic) != len(output):
            raise click.ClickException("Semantic suggestions must contain one predicted_class per input row.")
        suggestions, sources = attach_semantic_suggestions(
            torch.from_numpy(output["predicted_class"].to_numpy(dtype=np.int64, copy=True)),
            torch.from_numpy(semantic["predicted_class"].to_numpy(dtype=np.int64, copy=True)),
        )
        output["suggested_class"] = suggestions.numpy()
        output["decision_source"] = sources
    click.echo(output.to_json(orient="records", indent=2))


@cli.command("benchmark")
@click.option("--force", is_flag=True)
@_common_options
def benchmark_command(force, data_path, out_dir, device, epochs, batch_size, learning_rate, pair_count, calibration_epochs, calibration_pair_count, similarity_mode, support_draws, seed) -> None:
    cuda = _cuda(device)
    config = _training_config(
        epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
        pair_count=pair_count, calibration_epochs=calibration_epochs,
        calibration_pair_count=calibration_pair_count,
        similarity_mode=similarity_mode, seed=seed,
    )
    folds = []
    for holdout in fault_pairs():
        metrics_path = out_dir / f"fold_{holdout[0]:02d}_{holdout[1]:02d}" / "metrics.json"
        if metrics_path.exists() and not force:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        else:
            click.echo(f"[ONE-SHOT] Training fold {holdout[0]}-{holdout[1]} on {cuda}")
            metrics = run_crossfit_fold(
                data_path=data_path, out_dir=out_dir, holdout=holdout,
                device=cuda, config=config, support_draws=support_draws,
            )
        folds.append(metrics)
    summary = summarize_crossfit_benchmark(folds)
    save_json(out_dir / "benchmark_summary.json", summary)
    write_benchmark_tables(out_dir, summary)
    click.echo(json.dumps({key: value for key, value in summary.items() if key != "folds"}, indent=2))


if __name__ == "__main__":
    cli()
