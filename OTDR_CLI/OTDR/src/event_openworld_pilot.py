from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_curve

from .event_openworld_data import attach_input_groups
from .event_openworld_experiment import event_openworld_source_manifest, run_event_openworld_fold
from .event_openworld_training import ECConfig, PC2Config, SGMEConfig
from .model_functions.zero_shot import require_cuda
from .study_state import atomic_json, validate_run


PILOT_PAIRS = ((1, 2), (3, 5), (6, 7))


def _load(study_root: Path, name: str, cls: type[Any]) -> Any:
    payload = json.loads((study_root / "configs" / f"{name}_frozen.json").read_text(encoding="utf-8"))["config"]
    allowed = {field.name for field in fields(cls)}
    return cls(**{key: value for key, value in payload.items() if key in allowed})


def run_integrity_pilots(
    *, frame: pd.DataFrame, data_path: Path, study_root: Path, recipe_path: Path, device: torch.device,
    resume: bool = True,
) -> dict[str, Any]:
    device = require_cuda(str(device))
    frame = attach_input_groups(frame) if "_input_group" not in frame else frame
    sgme = _load(study_root, "sgme", SGMEConfig)
    configs = {"ec": _load(study_root, "ec", ECConfig), "pc2": _load(study_root, "pc2", PC2Config)}
    source = event_openworld_source_manifest()
    atomic_json(study_root / "configs" / "source_manifest.json", source)
    rows, warnings, failures = [], [], []
    for approach, config in configs.items():
        for pair in PILOT_PAIRS:
            run_id = f"pilot-{approach}-{pair[0]}_{pair[1]}"
            run_dir = study_root / "stress" / "integrity_pilots" / run_id
            valid, _ = validate_run(
                run_dir, {"run_id": run_id, "runtime_source_sha256": source["runtime_source_sha256"]}
            ) if resume else (False, "forced")
            if not valid:
                result = run_event_openworld_fold(
                    approach=approach, frame=frame, data_path=data_path, run_dir=run_dir,
                    holdout=pair, seed=42, config=config, sgme_config=sgme, device=device,
                    recipe_path=recipe_path, support_draws=3, shots=(1, 3, 5),
                    adaptation_buffers=(0, 128), run_sgme=approach == "ec",
                )
                manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
                manifest["run_id"] = run_id
                atomic_json(run_dir / "manifest.json", manifest)
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            arrays = np.load(run_dir / "predictions.npz")
            labels, score = arrays["labels"], arrays["novelty_score"]
            unknown = np.isin(labels, pair)
            normal = labels == 0
            seen_fault = (~unknown) & (~normal)
            finite = bool(np.isfinite(score).all() and np.isfinite(arrays["logits"]).all())
            unique_score = int(len(np.unique(score)))
            strict_distribution = metrics["semantic"]["strict_prediction_distribution"]
            operating_point = metrics["zero_day"]["operating_points"]["far_0.010"]
            row = {"approach": approach, "holdout": list(pair), "finite": finite, "unique_score_values": unique_score,
                   "auroc": metrics["zero_day"]["auroc"],
                   "observed_normal_far_1pct": operating_point["observed_normal_far"],
                   "unknown_recall_1pct": operating_point["unknown_recall"],
                   "known_acceptance_1pct": operating_point["known_acceptance"],
                   "strict_distribution": strict_distribution}
            rows.append(row)
            if not finite or unique_score < 100:
                failures.append({"run_id": run_id, "reason": "non-finite or degenerate novelty score"})
            if any(strict_distribution.get(str(class_id), 0) == 0 for class_id in pair):
                warnings.append({"run_id": run_id, "reason": "strict semantic class collapse; preserve and continue"})
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            axes[0].hist(score[normal], bins=80, density=True, alpha=.5, label="normal")
            axes[0].hist(score[seen_fault], bins=80, density=True, alpha=.5, label="seen fault")
            axes[0].hist(score[unknown], bins=80, density=True, alpha=.5, label="held-out fault")
            axes[0].set(title=f"{approach.upper()} pair {pair}: scores", xlabel="novelty", ylabel="density"); axes[0].legend()
            fpr, tpr, _ = roc_curve(unknown, score)
            normal_or_unknown = normal | unknown
            normal_fpr, normal_tpr, _ = roc_curve(unknown[normal_or_unknown], score[normal_or_unknown])
            axes[1].plot(normal_fpr, normal_tpr, label="raw score: normal negative")
            axes[1].plot(fpr, tpr, label="raw score: all known negative")
            axes[1].scatter(
                operating_point["observed_normal_far"], operating_point["unknown_recall"],
                marker="x", s=55, linewidths=2, color="black", label="calibrated 1% point", zorder=5,
            )
            axes[1].set(xlim=(0, .05), ylim=(0, 1), xlabel="false-positive rate", ylabel="unknown recall",
                        title="Low-FAR ROC")
            axes[1].legend()
            fig.tight_layout(); fig.savefig(study_root / "plots" / f"pilot_{approach}_{pair[0]}_{pair[1]}.png", dpi=180)
            plt.close(fig)
    result = {"schema_version": 1, "status": "pass" if not failures else "fail", "rows": rows,
              "warnings": warnings, "failures": failures,
              "runtime_source_sha256": source["runtime_source_sha256"],
              "note": "Class collapse is a scientific result and warning, not an integrity failure."}
    atomic_json(study_root / "tables" / "pilot_integrity.json", result)
    if failures:
        raise RuntimeError(f"Pilot integrity failed for {len(failures)} runs.")
    return result
