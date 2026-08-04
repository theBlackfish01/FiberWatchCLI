from __future__ import annotations

"""Command line orchestrator for the feature-assisted OTDR lifecycle study."""

import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
from typing import Iterable

import pandas as pd
import torch

from .lifecycle_data import data_audit
from .lifecycle_experiment import CFEConfig, FoldExperimentConfig, SCODConfig, execute_fold
from .lifecycle_external import run_external_lifecycle_validation
from .lifecycle_training import LifecycleTrainingConfig
from .lifecycle_sweep import (
    freeze_default_posthoc_finalists,
    run_posthoc_sweeps,
    run_representation_sweep,
)
from .lifecycle_stress import run_stress_validation
from .lifecycle_analysis import analyze_lifecycle
from .lifecycle_ablation import run_kpsc_ablation
from .lifecycle_tabpfn import run_tabpfn_pilot
from .lifecycle_posthoc import run_calibration_enrichment
from .lifecycle_synthesis import synthesize_auxiliary_results
from .lifecycle_completion import completion_audit
from .model_functions.lifecycle import LifecycleModelConfig
from .model_functions.zero_shot import require_cuda
from .study_state import atomic_json, environment_metadata


MODULE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = MODULE_ROOT / "src" / "data" / "OTDR_DATA.csv"
STUDY_ROOT = MODULE_ROOT / "experiments" / "otdr_feature_assisted_lifecycle_study"
OUTER_SEEDS = (42, 123, 2026, 7, 31415)
PILOT_PAIRS = ((1, 2), (3, 5), (6, 7))
ALL_PAIRS = tuple(itertools.combinations(range(1, 8), 2))


def load_frame() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def run_audit(device_name: str = "cuda:0") -> dict[str, object]:
    device = require_cuda(device_name)
    result = data_audit(load_frame(), data_path=DATA_PATH)
    result["cuda"] = {
        **environment_metadata(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "available": torch.cuda.is_available(),
    }
    atomic_json(STUDY_ROOT / "data_audit.json", result)
    return result


def _smoke_config(pair: tuple[int, int], seed: int, regime: str, device: str) -> FoldExperimentConfig:
    return FoldExperimentConfig(
        holdout=pair,
        seed=seed,
        regime=regime,
        device=device,
        stage="smoke",
        model=LifecycleModelConfig(width=24, embedding_dim=24, context_width=12, blocks=2),
        training=LifecycleTrainingConfig(
            epochs=1,
            steps_per_epoch=2,
            batch_size=128,
            patience=1,
            seed=seed,
        ),
        scod=SCODConfig(prototypes_per_class=2, knn_k=3),
        cfe=CFEConfig(shots=(1,), draws=1),
    )


def _pilot_config(pair: tuple[int, int], seed: int, regime: str, device: str) -> FoldExperimentConfig:
    return FoldExperimentConfig(
        holdout=pair,
        seed=seed,
        regime=regime,
        device=device,
        stage="pilots",
        model=LifecycleModelConfig(width=64, embedding_dim=64, context_width=32, blocks=3),
        training=LifecycleTrainingConfig(
            epochs=12,
            steps_per_epoch=50,
            batch_size=256,
            patience=4,
            seed=seed,
        ),
    )


def _load_frozen_config(pair: tuple[int, int], seed: int, regime: str, device: str) -> FoldExperimentConfig:
    path = STUDY_ROOT / "configs" / "finalists.json"
    if not path.exists():
        raise FileNotFoundError("Finalists are not frozen; run staged selection before the full benchmark.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    model = LifecycleModelConfig(**payload["shared_backbone"])
    training = LifecycleTrainingConfig(**{**payload["training"], "seed": seed})
    scod = SCODConfig(**payload["kpsc"])
    cfe_payload = dict(payload["cfe"])
    cfe_payload["shots"] = tuple(cfe_payload.get("shots", (1, 3, 5)))
    cfe = CFEConfig(**cfe_payload)
    return FoldExperimentConfig(
        holdout=pair,
        seed=seed,
        regime=regime,
        device=device,
        stage="full_benchmark",
        model=model,
        training=training,
        scod=scod,
        cfe=cfe,
    )


def run_configs(configs: Iterable[FoldExperimentConfig]) -> list[dict[str, object]]:
    frame = load_frame()
    results = []
    for config in configs:
        results.append(
            execute_fold(
                frame=frame,
                data_path=DATA_PATH,
                study_root=STUDY_ROOT,
                repository_root=REPOSITORY_ROOT,
                config=config,
            )
        )
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit_parser = subparsers.add_parser("audit")
    audit_parser.add_argument("--device", default="cuda:0")
    sweep_parser = subparsers.add_parser("sweep")
    sweep_parser.add_argument("--device", default="cuda:0")
    posthoc_parser = subparsers.add_parser("posthoc")
    posthoc_parser.add_argument("--device", default="cuda:0")
    external_parser = subparsers.add_parser("external")
    external_parser.add_argument("--device", default="cuda:0")
    stress_parser = subparsers.add_parser("stress")
    stress_parser.add_argument("--device", default="cuda:0")
    analyze_parser = subparsers.add_parser("analyze")
    analyze_parser.add_argument("--partial", action="store_true")
    analyze_parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=5000,
        help="Hierarchical bootstrap draws (use a smaller value only for diagnostics).",
    )
    analyze_parser.add_argument(
        "--regime", choices=("full", "trace_only", "summary_only"), default="full"
    )
    enrich_parser = subparsers.add_parser("enrich")
    enrich_parser.add_argument("--device", default="cuda:0")
    enrich_parser.add_argument(
        "--regime", choices=("full", "trace_only", "summary_only"), default="full"
    )
    enrich_parser.add_argument("--partial", action="store_true")
    enrich_parser.add_argument("--expected-runs", type=int)
    analyze_parser.add_argument(
        "--expected-runs",
        type=int,
        help="Override the complete-matrix expectation (default: 105).",
    )
    ablation_parser = subparsers.add_parser("ablate")
    ablation_parser.add_argument("--device", default="cuda:0")
    tabpfn_parser = subparsers.add_parser("tabpfn")
    tabpfn_parser.add_argument("--device", default="cuda:0")
    tabpfn_parser.add_argument("--draws", type=int, default=20)
    subparsers.add_parser("synthesize")
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="Write and print the audit without raising on unmet checks.",
    )
    for name in ("smoke", "pilot", "full"):
        command = subparsers.add_parser(name)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--regime", choices=("full", "trace_only", "summary_only"), default="full")
        command.add_argument("--pair", nargs=2, type=int)
        command.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    if args.command == "audit":
        print(json.dumps(run_audit(args.device), indent=2))
        return
    if args.command == "sweep":
        result = run_representation_sweep(
            frame=load_frame(), study_root=STUDY_ROOT, device=args.device
        )
        freeze_default_posthoc_finalists(STUDY_ROOT, result)
        finalists = run_posthoc_sweeps(
            frame=load_frame(), study_root=STUDY_ROOT,
            representation_result=result, device=args.device,
        )
        print(json.dumps({"representation": result, "finalists": finalists}, indent=2))
        return
    if args.command == "posthoc":
        path = STUDY_ROOT / "sweeps" / "representation_finalist.json"
        if not path.exists():
            raise FileNotFoundError("Representation finalist does not exist.")
        finalists = run_posthoc_sweeps(
            frame=load_frame(), study_root=STUDY_ROOT,
            representation_result=json.loads(path.read_text(encoding="utf-8")),
            device=args.device,
        )
        print(json.dumps(finalists, indent=2))
        return
    if args.command == "external":
        path = STUDY_ROOT / "configs" / "finalists.json"
        if not path.exists():
            raise FileNotFoundError("Finalists are not frozen.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = run_external_lifecycle_validation(
            frame=load_frame(),
            data_path=DATA_PATH,
            external_root=MODULE_ROOT / "experiments" / "otdr_event_openworld_study" / "external" / "dataset",
            study_root=STUDY_ROOT,
            model_config=LifecycleModelConfig(**payload["shared_backbone"]),
            training_config=LifecycleTrainingConfig(**payload["training"]),
            device=args.device,
        )
        print(json.dumps({"variants": list(metrics["zero_target"])}, indent=2))
        return
    if args.command == "stress":
        path = STUDY_ROOT / "configs" / "finalists.json"
        if not path.exists():
            raise FileNotFoundError("Finalists are not frozen.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = run_stress_validation(
            frame=load_frame(), study_root=STUDY_ROOT,
            model_config=LifecycleModelConfig(**payload["shared_backbone"]),
            training_config=LifecycleTrainingConfig(**payload["training"]),
            device=args.device,
        )
        print(json.dumps({"rows": len(metrics["rows"])}, indent=2))
        return
    if args.command == "analyze":
        metrics = analyze_lifecycle(
            STUDY_ROOT,
            regime=args.regime,
            require_complete=not args.partial,
            expected_runs=args.expected_runs,
            bootstrap_iterations=args.bootstrap_iterations,
        )
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "enrich":
        metrics = run_calibration_enrichment(
            frame=load_frame(),
            study_root=STUDY_ROOT,
            regime=args.regime,
            device=args.device,
            require_complete=not args.partial,
            expected_runs=args.expected_runs,
        )
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "ablate":
        path = STUDY_ROOT / "configs" / "finalists.json"
        if not path.exists():
            raise FileNotFoundError("Finalists are not frozen.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        metrics = run_kpsc_ablation(
            frame=load_frame(), study_root=STUDY_ROOT,
            model_config=LifecycleModelConfig(**payload["shared_backbone"]),
            training_config=LifecycleTrainingConfig(**payload["training"]),
            device=args.device,
        )
        print(json.dumps({"rows": len(metrics["rows"])}, indent=2))
        return
    if args.command == "tabpfn":
        metrics = run_tabpfn_pilot(
            frame=load_frame(), study_root=STUDY_ROOT,
            device=args.device, draws=args.draws,
        )
        print(json.dumps({"rows": len(metrics["rows"])}, indent=2))
        return
    if args.command == "synthesize":
        metrics = synthesize_auxiliary_results(STUDY_ROOT)
        print(json.dumps(metrics, indent=2))
        return
    if args.command == "verify":
        audit = completion_audit(STUDY_ROOT)
        print(json.dumps(audit, indent=2))
        if not audit["complete"] and not args.allow_incomplete:
            raise RuntimeError("Lifecycle study completion audit failed.")
        return
    require_cuda(args.device)
    if args.command == "smoke":
        pair = tuple(args.pair or (1, 2))
        seed = args.seed or 42
        configs = [_smoke_config(pair, seed, args.regime, args.device)]
    elif args.command == "pilot":
        pairs = [tuple(args.pair)] if args.pair else list(PILOT_PAIRS)
        seeds = [args.seed] if args.seed is not None else [42]
        configs = [_pilot_config(pair, seed, args.regime, args.device) for pair in pairs for seed in seeds]
    else:
        pairs = [tuple(args.pair)] if args.pair else list(ALL_PAIRS)
        seeds = [args.seed] if args.seed is not None else list(OUTER_SEEDS)
        configs = [_load_frozen_config(pair, seed, args.regime, args.device) for pair in pairs for seed in seeds]
    results = run_configs(configs)
    print(json.dumps({"completed": len(results), "run_ids": [value["run_id"] for value in results]}, indent=2))


if __name__ == "__main__":
    main()
