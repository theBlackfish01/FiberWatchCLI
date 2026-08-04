from __future__ import annotations

from dataclasses import fields
from itertools import combinations
import json
from pathlib import Path
import shutil

import click
import pandas as pd
import torch

from .model_functions.zero_shot import require_cuda
from .study_data import feature_signature, load_frame, validate_model_features
from .study_experiment import run_study_fold
from .study_state import StudyState, environment_metadata, file_sha256, stable_run_id, validate_run
from .study_sweep import run_sweep
from .study_training import ApproachAConfig, ApproachBConfig, ApproachCConfig
from .zero_shot_data import INPUT_COLUMNS, validate_zero_shot_frame


OTDR_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = OTDR_ROOT / "experiments" / "otdr_three_approach_study"
DATA_PATH = OTDR_ROOT / "src" / "data" / "OTDR_DATA.csv"
PHYSICS_PATH = OTDR_ROOT / "src" / "corpus" / "otdr_physics_prototypes.json"
DESCRIPTION_PATH = OTDR_ROOT / "src" / "corpus" / "zero_shot_fault_prototypes.json"
CONFIG_CLASSES = {"a": ApproachAConfig, "b": ApproachBConfig, "c": ApproachCConfig}


def _device(value: str) -> torch.device:
    try:
        return require_cuda(value)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _approaches(value: str) -> list[str]:
    return ["a", "b", "c"] if value == "all" else [value]


def _load_frozen(approach: str, seed: int):
    path = STUDY_ROOT / "configs" / f"approach_{approach}_frozen.json"
    if not path.exists():
        raise click.ClickException(f"Frozen config is missing: {path}. Run the sweep first.")
    payload = json.loads(path.read_text(encoding="utf-8"))["config"]
    payload["seed"] = seed
    allowed = {field.name for field in fields(CONFIG_CLASSES[approach])}
    return CONFIG_CLASSES[approach](**{key: value for key, value in payload.items() if key in allowed})


@click.group()
def cli() -> None:
    """CUDA-only, leakage-safe OTDR three-approach study."""


@cli.command("audit")
@click.option("--device", default="cuda:0", show_default=True)
def audit_command(device: str) -> None:
    cuda = _device(device)
    frame = load_frame(DATA_PATH)
    validate_zero_shot_frame(frame)
    validate_model_features(INPUT_COLUMNS)
    payload = {
        "dataset": str(DATA_PATH), "dataset_sha256": file_sha256(DATA_PATH), "rows": len(frame),
        "class_counts": {str(int(k)): int(v) for k, v in frame["Class"].value_counts().sort_index().items()},
        "feature_signature": feature_signature(), "features": INPUT_COLUMNS,
        "environment": environment_metadata(cuda),
        "disk_free_bytes": shutil.disk_usage(STUDY_ROOT).free,
    }
    click.echo(json.dumps(payload, indent=2))


@cli.command("sweep")
@click.option("--approach", type=click.Choice(["a", "b", "c", "all"]), default="all", show_default=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
def sweep_command(approach: str, device: str, resume: bool) -> None:
    cuda = _device(device)
    frame = load_frame(DATA_PATH)
    for name in _approaches(approach):
        click.echo(f"[SWEEP] Approach {name.upper()} on {cuda}")
        result = run_sweep(approach=name, frame=frame, device=cuda, study_root=STUDY_ROOT,
                           physics_path=PHYSICS_PATH, description_path=DESCRIPTION_PATH, resume=resume)
        click.echo(json.dumps(result, indent=2))


@cli.command("benchmark")
@click.option("--approach", type=click.Choice(["a", "b", "c", "all"]), default="all", show_default=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--support-draws", default=20, type=click.IntRange(1), show_default=True)
def benchmark_command(approach: str, device: str, resume: bool, support_draws: int) -> None:
    cuda = _device(device)
    frame = load_frame(DATA_PATH)
    state = StudyState(STUDY_ROOT)
    pairs = list(combinations(range(1, 8), 2))
    for name in _approaches(approach):
        for seed in (42, 123, 2026):
            config = _load_frozen(name, seed)
            for holdout in pairs:
                run_id = stable_run_id(name, holdout, seed, config)
                run_dir = STUDY_ROOT / "full_benchmark" / name / run_id
                valid, reason = validate_run(run_dir, {"run_id": run_id}) if resume else (False, "forced")
                if valid:
                    click.echo(f"[SKIP] {run_id}: validated")
                    continue
                click.echo(f"[RUN] {run_id} ({reason}) on {cuda}")
                metadata = {"run_id": run_id, "approach": name, "holdout": list(holdout), "seed": seed,
                            "config": vars(config), "phase": "full_benchmark"}
                with state.run(run_id, run_dir, metadata):
                    run_study_fold(
                        approach=name, frame=frame, data_path=DATA_PATH, run_dir=run_dir,
                        holdout=holdout, seed=seed, config=config, device=cuda,
                        physics_path=PHYSICS_PATH, description_path=DESCRIPTION_PATH,
                        study_root=STUDY_ROOT, support_draws=support_draws,
                    )
                torch.cuda.empty_cache()
    state.update(status="full_benchmark_complete", note="All requested benchmark runs completed or validated.")


@cli.command("analyze")
def analyze_command() -> None:
    from .study_analysis import analyze_study
    result = analyze_study(STUDY_ROOT)
    click.echo(json.dumps(result, indent=2))


if __name__ == "__main__":
    cli()
