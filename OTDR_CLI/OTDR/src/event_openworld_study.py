from __future__ import annotations

from dataclasses import fields
from itertools import combinations
import json
from pathlib import Path

import click
import numpy as np
import pandas as pd
import torch

from .event_openworld_audit import write_initial_audits
from .event_openworld_analysis import analyze_event_study
from .event_openworld_ablation import run_ablation_study
from .event_openworld_baselines import run_closed_set_sanity
from .event_openworld_data import attach_input_groups
from .event_openworld_experiment import event_openworld_source_manifest, run_event_openworld_fold
from .event_openworld_external import run_external_validation
from .event_openworld_pilot import run_integrity_pilots
from .event_openworld_sweep import run_neural_sweep, run_sgme_sweep
from .event_openworld_stress import run_stress_validation
from .event_openworld_training import ECConfig, PC2Config, SGMEConfig
from .model_functions.zero_shot import require_cuda
from .study_state import StudyState, atomic_json, file_sha256, stable_run_id, validate_run


OTDR_ROOT = Path(__file__).resolve().parents[1]
STUDY_ROOT = OTDR_ROOT / "experiments" / "otdr_event_openworld_study"
DATA_PATH = OTDR_ROOT / "src" / "data" / "OTDR_DATA.csv"
RECIPE_PATH = STUDY_ROOT / "configs" / "event_recipes.json"
CONFIG_CLASSES = {"ec": ECConfig, "pc2": PC2Config}


def _device(value: str) -> torch.device:
    try:
        return require_cuda(value)
    except (RuntimeError, ValueError) as exc:
        raise click.ClickException(str(exc)) from exc


def _load_frame() -> pd.DataFrame:
    return attach_input_groups(pd.read_csv(DATA_PATH))


def _load_frozen(approach: str, seed: int) -> ECConfig | PC2Config:
    path = STUDY_ROOT / "configs" / f"{approach}_frozen.json"
    if not path.exists():
        raise click.ClickException(f"Frozen config missing: {path}. Complete the sweep first.")
    payload = json.loads(path.read_text(encoding="utf-8"))["config"]
    payload["seed"] = seed
    cls = CONFIG_CLASSES[approach]
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def _load_sgme() -> SGMEConfig:
    path = STUDY_ROOT / "configs" / "sgme_frozen.json"
    if not path.exists():
        raise click.ClickException(f"Frozen SGME config missing: {path}. Complete the SGME sweep before any outer benchmark.")
    payload = json.loads(path.read_text(encoding="utf-8"))["config"]
    allowed = {field.name for field in fields(SGMEConfig)}
    return SGMEConfig(**{key: value for key, value in payload.items() if key in allowed})


@click.group()
def cli() -> None:
    """CUDA-only physics-constrained open-world OTDR study."""


@cli.command("audit")
@click.option("--device", default="cuda:0", show_default=True)
def audit_command(device: str) -> None:
    _device(device)
    write_initial_audits(STUDY_ROOT, DATA_PATH, device)
    click.echo("Initial local, CUDA, and external-data audits written.")


@cli.command("sanity")
@click.option("--device", default="cuda:0", show_default=True)
def sanity_command(device: str) -> None:
    cuda = _device(device)
    result = run_closed_set_sanity(_load_frame(), study_root=STUDY_ROOT, device=cuda)
    click.echo(json.dumps(result, indent=2))


@cli.command("external")
@click.option("--device", default="cuda:0", show_default=True)
def external_command(device: str) -> None:
    cuda = _device(device)
    result = run_external_validation(frame=_load_frame(), data_path=DATA_PATH, study_root=STUDY_ROOT,
                                     recipe_path=RECIPE_PATH, device=cuda)
    click.echo(json.dumps(result, indent=2))


@cli.command("stress")
@click.option("--device", default="cuda:0", show_default=True)
def stress_command(device: str) -> None:
    cuda = _device(device)
    result = run_stress_validation(frame=_load_frame(), study_root=STUDY_ROOT, recipe_path=RECIPE_PATH, device=cuda)
    click.echo(json.dumps({"rows": len(result["rows"]), "grid": result["grid"]}, indent=2))


@cli.command("pilot")
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
def pilot_command(device: str, resume: bool) -> None:
    cuda = _device(device)
    result = run_integrity_pilots(frame=_load_frame(), data_path=DATA_PATH, study_root=STUDY_ROOT,
                                  recipe_path=RECIPE_PATH, device=cuda, resume=resume)
    click.echo(json.dumps(result, indent=2))


@cli.command("ablate")
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
def ablate_command(device: str, resume: bool) -> None:
    cuda = _device(device)
    result = run_ablation_study(frame=_load_frame(), data_path=DATA_PATH, study_root=STUDY_ROOT,
                                recipe_path=RECIPE_PATH, device=cuda, resume=resume)
    click.echo(json.dumps({key: len(value) if isinstance(value, list) else value for key, value in result.items()}, indent=2))


@cli.command("sweep")
@click.option("--approach", type=click.Choice(["ec", "pc2", "sgme"]), required=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
def sweep_command(approach: str, device: str, resume: bool) -> None:
    cuda = _device(device)
    frame = _load_frame()
    click.echo(f"Starting {approach.upper()} nested CUDA sweep on {cuda}.")
    if approach == "sgme":
        result = run_sgme_sweep(frame=frame, study_root=STUDY_ROOT, recipe_path=RECIPE_PATH, device=cuda, resume=resume)
    else:
        result = run_neural_sweep(
            approach=approach, frame=frame, study_root=STUDY_ROOT,
            recipe_path=RECIPE_PATH, device=cuda, resume=resume,
        )
    StudyState(STUDY_ROOT).update(
        status=f"{approach}_sweep_complete",
        selected_configs={approach: result},
        note=f"{approach.upper()} finalist frozen from inner-only selection; no outer query evaluated.",
    )
    click.echo(json.dumps(result, indent=2))


@cli.command("benchmark")
@click.option("--approach", type=click.Choice(["ec", "pc2", "all"]), default="all", show_default=True)
@click.option("--device", default="cuda:0", show_default=True)
@click.option("--resume/--no-resume", default=True, show_default=True)
@click.option("--support-draws", default=20, type=click.IntRange(1), show_default=True)
@click.option("--continue-on-error/--fail-fast", default=True, show_default=True)
def benchmark_command(approach: str, device: str, resume: bool, support_draws: int, continue_on_error: bool) -> None:
    cuda = _device(device)
    if support_draws != 20:
        raise click.ClickException("The frozen full protocol requires exactly 20 support draws.")
    source = event_openworld_source_manifest()
    pilot_path = STUDY_ROOT / "tables" / "pilot_integrity.json"
    pilot = json.loads(pilot_path.read_text(encoding="utf-8")) if pilot_path.exists() else {}
    if (pilot.get("status") != "pass" or
            pilot.get("runtime_source_sha256") != source["runtime_source_sha256"]):
        raise click.ClickException("Pilot score/histogram integrity gate has not passed. Run the `pilot` command first.")
    frame = _load_frame()
    state = StudyState(STUDY_ROOT)
    sgme = _load_sgme()
    atomic_source = STUDY_ROOT / "configs" / "source_manifest.json"
    atomic_json(atomic_source, source)
    approaches = ["ec", "pc2"] if approach == "all" else [approach]
    failures = []
    for name in approaches:
        for seed in (42, 123, 2026, 7, 31415):
            config = _load_frozen(name, seed)
            for holdout in combinations(range(1, 8), 2):
                identity = {"model": config, "sgme": sgme if name == "ec" else None,
                            "runtime_source_sha256": source["runtime_source_sha256"]}
                run_id = stable_run_id(name, holdout, seed, identity)
                run_dir = STUDY_ROOT / "full_benchmark" / name / run_id
                valid, reason = validate_run(
                    run_dir, {"run_id": run_id, "runtime_source_sha256": source["runtime_source_sha256"]}
                ) if resume else (False, "forced")
                if valid:
                    click.echo(f"[SKIP] {run_id}: validated")
                    continue
                click.echo(f"[RUN] {run_id}: {reason}")
                metadata = {"run_id": run_id, "approach": name, "holdout": list(holdout), "seed": seed,
                            "config": identity, "phase": "full_benchmark"}
                try:
                    with state.run(run_id, run_dir, metadata):
                        run_event_openworld_fold(
                            approach=name, frame=frame, data_path=DATA_PATH, run_dir=run_dir,
                            holdout=holdout, seed=seed, config=config, sgme_config=sgme,
                            device=cuda, recipe_path=RECIPE_PATH, support_draws=support_draws,
                            run_sgme=name == "ec",
                        )
                except Exception as exc:
                    failures.append({"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"})
                    click.echo(f"[FAIL] {run_id}: {type(exc).__name__}: {exc}", err=True)
                    if not continue_on_error:
                        raise
                finally:
                    torch.cuda.empty_cache()
    if failures:
        state.update(status="benchmark_incomplete", note=f"Benchmark command ended with {len(failures)} failures; resume after repair.")
        raise click.ClickException(f"{len(failures)} runs failed. See failures.jsonl.")
    state.update(status="full_benchmark_complete", note="All selected full benchmark runs completed or hash-validated.")


@cli.command("validate")
def validate_command() -> None:
    failures = []
    count = 0
    observed = set()
    source = event_openworld_source_manifest()
    for manifest in (STUDY_ROOT / "full_benchmark").rglob("manifest.json"):
        count += 1
        valid, reason = validate_run(manifest.parent, {
            "runtime_source_sha256": source["runtime_source_sha256"],
            "cuda_verified": True,
        })
        if not valid:
            failures.append({"run": str(manifest.parent), "reason": reason})
            continue
        manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
        observed.add((manifest_payload.get("approach"), tuple(manifest_payload.get("holdout", [])),
                      manifest_payload.get("seed")))
        metrics = json.loads((manifest.parent / "metrics.json").read_text(encoding="utf-8"))
        expected_graph_rows = 240 if metrics.get("approach") == "ec" else 0
        if (len(metrics.get("inductive_enrollment", [])) != 60 or
                len(metrics.get("raw_baselines", [])) != 60 or
                len(metrics.get("sgme_enrollment", [])) != expected_graph_rows):
            failures.append({"run": str(manifest.parent), "reason": "support-draw/shot/buffer matrix incomplete"})
            continue
        split = json.loads((manifest.parent / "split_manifest.json").read_text(encoding="utf-8"))
        if split.get("dataset_sha256") != file_sha256(DATA_PATH):
            failures.append({"run": str(manifest.parent), "reason": "dataset hash mismatch"})
            continue
        exact = STUDY_ROOT / split["exact_group_manifest"]
        if not exact.is_file() or file_sha256(exact) != split["exact_group_manifest_sha256"]:
            failures.append({"run": str(manifest.parent), "reason": "exact split manifest missing or hash mismatch"})
            continue
        try:
            with np.load(exact) as split_groups, np.load(manifest.parent / "enrollment_groups.npz") as enrollment:
                reference = {bytes(value) for value in split_groups["reference_pool"]}
                adaptation = {bytes(value) for value in split_groups["adaptation_pool"]}
                query = {bytes(value) for value in split_groups["query"]}
                for name in enrollment.files:
                    values = {bytes(value) for value in enrollment[name]}
                    allowed = reference if name.startswith("support_") else adaptation
                    if not values.issubset(allowed) or values & query:
                        failures.append({"run": str(manifest.parent), "reason": f"enrollment isolation failed: {name}"})
                        break
        except (OSError, ValueError, KeyError) as exc:
            failures.append({"run": str(manifest.parent), "reason": f"enrollment manifest unreadable: {exc}"})
    if count != 210:
        failures.append({"run": "full_benchmark", "reason": f"expected 210 runs, found {count}"})
    expected = {
        (approach, pair, seed)
        for approach in ("ec", "pc2")
        for pair in combinations(range(1, 8), 2)
        for seed in (42, 123, 2026, 7, 31415)
    }
    if observed != expected:
        failures.append({"run": "full_benchmark", "reason":
                         f"run matrix mismatch: missing={len(expected - observed)}, extra={len(observed - expected)}"})
    click.echo(json.dumps({"runs": count, "expected_runs": 210,
                           "valid": count - sum(item["run"] != "full_benchmark" for item in failures),
                           "failures": failures}, indent=2))
    if failures:
        raise click.ClickException("Artifact validation failed.")


@cli.command("analyze")
def analyze_command() -> None:
    click.echo(json.dumps(analyze_event_study(STUDY_ROOT), indent=2))


if __name__ == "__main__":
    cli()
