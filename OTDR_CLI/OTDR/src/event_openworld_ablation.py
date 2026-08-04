from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .event_openworld_data import attach_input_groups
from .event_openworld_experiment import event_openworld_source_manifest, run_event_openworld_fold
from .event_openworld_training import ECConfig, PC2Config, SGMEConfig
from .model_functions.zero_shot import require_cuda
from .study_state import config_hash, validate_run


ABLATION_PAIRS = ((1, 2), (3, 5), (6, 7))


def _load_config(study_root: Path, approach: str, cls: type[ECConfig] | type[PC2Config]) -> ECConfig | PC2Config:
    payload = json.loads((study_root / "configs" / f"{approach}_frozen.json").read_text(encoding="utf-8"))["config"]
    return cls(**payload)


def ec_ablations(finalist: ECConfig) -> dict[str, ECConfig]:
    return {
        "no_canonicalization": replace(finalist, canonicalize=False),
        "hard_alignment": replace(finalist, soft_alignment=False),
        "no_derivatives": replace(finalist, derivative_channels=False),
        "no_global_branch": replace(finalist, global_branch=False),
        "deterministic_factors": replace(finalist, deterministic_factors=True),
        "no_residual": replace(finalist, residual=False, residual_penalty=0),
        "point_recipes": replace(finalist, recipe_mode="point"),
        "tcn_backbone": replace(finalist, backbone="tcn"),
        "shapelet_backbone": replace(finalist, backbone="shapelet"),
    }


def pc2_ablations(finalist: PC2Config) -> dict[str, PC2Config]:
    rows = {
        "no_outlier_exposure": replace(finalist, outlier_mode="none"),
        "generic_corruption_oe": replace(finalist, outlier_mode="generic"),
        "virtual_feature_outliers": replace(finalist, outlier_mode="virtual_feature"),
        "ce_only": replace(finalist, outlier_mode="none", factor_weight=0, named_weight=0, oe_weight=0, cvar_weight=0),
        "energy_without_cvar": replace(finalist, cvar_weight=0),
        "global_calibration": replace(finalist, calibration="global"),
        "mondrian_calibration": replace(finalist, calibration="mondrian"),
        "normalized_calibration": replace(finalist, calibration="normalized"),
    }
    for atom in ("narrow_reflection", "abrupt_step", "smooth_or_broad_attenuation", "terminal_drop",
                 "reflection_dead_zone", "irregular_mixture", "multi_event"):
        rows[f"without_atom_{atom}"] = replace(finalist, atom_ablation=atom)
    return rows


def sgme_ablations(finalist: SGMEConfig) -> dict[str, SGMEConfig]:
    return {
        "graph_no_semantic_guards": replace(finalist, semantic_threshold=0, augmentation_threshold=0,
                                             seen_rejection_threshold=-1, agreement_threshold=0),
        "graph_semantic_only": replace(finalist, augmentation_threshold=0, agreement_threshold=0),
        "graph_semantic_augmentation": replace(finalist, agreement_threshold=0),
        "prototype_covariance_off": replace(finalist, covariance=False),
        "no_abstention": replace(finalist, abstention_quantile=0),
    }


def run_ablation_study(
    *, frame: pd.DataFrame, data_path: Path, study_root: Path, recipe_path: Path, device: torch.device,
    resume: bool = True,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    frame = attach_input_groups(frame) if "_input_group" not in frame else frame
    ec_final = _load_config(study_root, "ec", ECConfig)
    pc2_final = _load_config(study_root, "pc2", PC2Config)
    sgme_payload = json.loads((study_root / "configs" / "sgme_frozen.json").read_text(encoding="utf-8"))["config"]
    sgme_final = SGMEConfig(**sgme_payload)
    source = event_openworld_source_manifest()
    completed, skipped, failures = [], [], []
    groups: list[tuple[str, str, ECConfig | PC2Config, SGMEConfig, bool]] = []
    groups.extend(("ec", name, config, sgme_final, False) for name, config in ec_ablations(ec_final).items())
    groups.extend(("pc2", name, config, sgme_final, False) for name, config in pc2_ablations(pc2_final).items())
    groups.extend(("ec", f"sgme_{name}", ec_final, config, True) for name, config in sgme_ablations(sgme_final).items())
    for approach, name, config, graph_config, run_sgme in groups:
        for holdout in ABLATION_PAIRS:
            identity = {"name": name, "model": asdict(config), "sgme": asdict(graph_config) if run_sgme else None,
                        "runtime_source_sha256": source["runtime_source_sha256"]}
            run_id = f"{approach}-{name}-{holdout[0]}_{holdout[1]}-{config_hash(identity)}"
            run_dir = study_root / "stress" / "ablations" / approach / run_id
            valid, _ = validate_run(run_dir, {
                "run_id": run_id,
                "runtime_source_sha256": source["runtime_source_sha256"],
            }) if resume else (False, "forced")
            if valid:
                skipped.append(run_id); continue
            try:
                run_event_openworld_fold(
                    approach=approach, frame=frame, data_path=data_path, run_dir=run_dir,
                    holdout=holdout, seed=42, config=replace(config, seed=42), sgme_config=graph_config,
                    device=device, recipe_path=recipe_path, support_draws=3,
                    shots=(1, 3, 5) if not run_sgme else (1,),
                    adaptation_buffers=(0, 32, 128, 512) if run_sgme else (0,), run_sgme=run_sgme,
                )
                manifest_path = run_dir / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["run_id"] = run_id
                from .study_state import atomic_json
                atomic_json(manifest_path, manifest)
                completed.append(run_id)
            except Exception as exc:
                failures.append({"run_id": run_id, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                torch.cuda.empty_cache()
    result = {"completed": completed, "skipped": skipped, "failures": failures, "pairs": [list(value) for value in ABLATION_PAIRS]}
    from .study_state import atomic_json
    atomic_json(study_root / "tables" / "ablation_run_summary.json", result)
    if failures:
        raise RuntimeError(f"{len(failures)} ablation runs failed; resume after repair.")
    return result
