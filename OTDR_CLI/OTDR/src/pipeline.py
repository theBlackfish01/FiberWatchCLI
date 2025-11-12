"""High-level OTDR training ➜ evaluation pipeline CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import click

from . import train as train_cli
from . import eval as eval_cli


def _checkpoint_path(out_dir: Path, classifier: str, anomaly_only: bool) -> Path:
    if classifier == "tcn":
        name = "tcn_anomaly.pt" if anomaly_only else "tcn_full.pt"
    elif classifier == "tcn_binary":
        name = "tcn_binary.pt"
    elif classifier == "tab":
        name = "tabnet.pt"
    else:
        name = "tst.pt"
    return out_dir / name


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--full-run",
    is_flag=True,
    help="Train all OTDR models and immediately evaluate the GRU-AE pipeline.",
)
@click.option(
    "--data",
    "data_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/OTDR_DATA.csv"),
    show_default=True,
    help="Path to the cleaned OTDR dataset (CSV or Parquet).",
)
@click.option(
    "--out-dir",
    type=str,
    default="models",
    show_default=True,
    help="Directory where training checkpoints and metadata will be written.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="cuda | cuda:0 | mps | cpu | leave empty for auto-detect.",
)
@click.option(
    "--classifier",
    type=click.Choice(["tcn", "tcn_binary", "tst", "tab"], case_sensitive=False),
    default="tcn",
    show_default=True,
    help="Classifier checkpoint to evaluate after training completes.",
)
@click.option(
    "--eval-dir",
    type=str,
    default=None,
    help="Optional outputs/ subfolder for evaluation artifacts (defaults to pipeline_<out-dir>).",
)
@click.option(
    "--tcn-anomaly-only/--tcn-all-data",
    "tcn_anomaly_only",
    default=False,
    help="Train and evaluate the anomaly-only TCN variant (Class != 0).",
)
@click.option(
    "--extra-feature",
    "extra_features",
    multiple=True,
    help=(
        "Optional additional feature columns to append to the default measurement "
        "set (repeat flag for multiple columns)."
    ),
)
@click.option(
    "--num-samples",
    type=click.IntRange(0, None),
    default=0,
    show_default=True,
    help="Random evaluation samples to visualise/explain; 0 skips explainability.",
)
@click.option(
    "--orchestrate-tst",
    is_flag=True,
    help="Chain binary ➜ anomaly-only TCN ➜ TST during evaluation (requires trained checkpoints).",
)
def main(
    full_run: bool,
    data_path: Path,
    out_dir: str,
    device: str | None,
    classifier: str,
    eval_dir: str | None,
    tcn_anomaly_only: bool,
    extra_features: Tuple[str, ...],
    num_samples: int,
    orchestrate_tst: bool,
) -> None:
    """Run ``src.train`` and ``src.eval`` sequentially with shared configuration."""

    if not full_run:
        raise click.UsageError("Specify --full-run to launch the training/evaluation pipeline.")

    out_dir_path = Path(out_dir)
    eval_dir_name = eval_dir or f"pipeline_{out_dir_path.name}"
    extras = tuple(dict.fromkeys(extra_features))

    click.echo("[PIPELINE] Starting training phase (mode=all)...")
    train_cli.main.callback(
        mode="all",
        data_path=data_path,
        out_dir=str(out_dir_path),
        device=device,
        tcn_anomaly_only=tcn_anomaly_only,
        extra_features=extras,
    )

    detector_path = out_dir_path / "gru_ae.pt"
    cls_path = _checkpoint_path(out_dir_path, classifier, tcn_anomaly_only)

    click.echo("[PIPELINE] Launching evaluation phase (mode=pipeline)...")
    eval_cli.main.callback(
        mode="pipeline",
        classifier=classifier,
        data_path=data_path,
        detector=detector_path,
        cls_path=cls_path,
        num_samples=num_samples,
        out_dir=eval_dir_name,
        device=device,
        tcn_anomaly_only=tcn_anomaly_only,
        orchestrate_tst=orchestrate_tst,
        extra_features=extras,
    )


if __name__ == "__main__":
    main()
