from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

from .study_data import group_stratified_inner_splits, prepare_fold
from .study_experiment import DensityScorer, class_prototypes, cosine_scores
from .study_metrics import classification_metrics, harmonic, macro_class_accuracy
from .study_semantics import semantic_prototypes
from .study_state import StudyState, atomic_json, config_hash, stable_run_id, validate_run, write_manifest
from .study_training import (
    ApproachAConfig, ApproachBConfig, ApproachCConfig, encode,
    train_approach_a, train_approach_b, train_approach_c,
)
from .zero_shot_training import save_json


PILOT_FOLDS = [(1, 2), (3, 5), (6, 7)]


def _sample_unique(rows: list[dict[str, Any]], count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(rows))
    return [rows[index] for index in order[:count]]


def candidate_configs(approach: str, *, seed: int = 20260716) -> list[Any]:
    if approach == "a":
        rows = [dict(embedding_dim=e, dropout=d, learning_rate=lr, supcon_weight=s,
                     hard_negative_weight=h, aggregation=a, temperature=t)
                for e, d, lr, s, h, a, t in product(
                    [64, 128, 256], [0.05, 0.15, 0.30], [1e-4, 3e-4, 1e-3],
                    [0.0, 0.25, 0.5], [0.0, 0.1], ["prototype", "equal", "medoid", "random"], [0.07, 0.15])]
        selected = _sample_unique(rows, 24, seed)
        # Guarantee core objective and gallery ablations are represented.
        selected[:4] = [
            dict(embedding_dim=128, dropout=0.10, learning_rate=3e-4, supcon_weight=0.25, hard_negative_weight=0.1, aggregation=value, temperature=0.10)
            for value in ("prototype", "equal", "medoid", "random")]
        return [ApproachAConfig(epochs=3, steps_per_epoch=24, seed=42, **row) for row in selected]
    if approach == "b":
        rows = [dict(latent_dim=e, dropout=d, learning_rate=lr, temperature=t, prototype_weight=p,
                     attribute_weight=a, supcon_weight=s, prototype_mode=m, seen_penalty_grid_max=g)
                for e, d, lr, t, p, a, s, m, g in product(
                    [64, 128, 256], [0.05, 0.15, 0.30], [1e-4, 3e-4, 1e-3], [0.05, 0.10, 0.20],
                    [0.5, 1.0], [0.25, 0.75, 1.5], [0.0, 0.2], ["physics", "combined", "text"], [1.0, 3.0])]
        selected = _sample_unique(rows, 24, seed + 1)
        selected[:3] = [
            dict(latent_dim=128, dropout=0.10, learning_rate=3e-4, temperature=0.08, prototype_weight=1.0,
                 attribute_weight=0.5, supcon_weight=0.1, prototype_mode=value, seen_penalty_grid_max=3.0)
            for value in ("physics", "combined", "text")]
        return [ApproachBConfig(epochs=3, steps_per_epoch=24, seed=42, **row) for row in selected]
    if approach == "c":
        rows = [dict(embedding_dim=e, dropout=d, learning_rate=lr, temperature=t, mask_ratio=m,
                     reconstruction_weight=r, contrastive_weight=c, noise_std=n, scale_std=s,
                     density=den, knn_k=k, covariance_shrinkage=sh)
                for e, d, lr, t, m, r, c, n, s, den, k, sh in product(
                    [64, 128, 256], [0.05, 0.15, 0.30], [1e-4, 3e-4, 1e-3], [0.07, 0.15],
                    [0.10, 0.20, 0.30], [0.5, 1.0, 2.0], [0.5, 1.0, 2.0], [0.01, 0.03],
                    [0.01, 0.05], ["mahalanobis", "knn"], [5, 15], [0.05, 0.2, 0.5])]
        selected = _sample_unique(rows, 24, seed + 2)
        selected[:2] = [
            dict(embedding_dim=128, dropout=0.10, learning_rate=3e-4, temperature=0.10, mask_ratio=0.15,
                 reconstruction_weight=1.0, contrastive_weight=1.0, noise_std=0.02, scale_std=0.03,
                 density=value, knn_k=10, covariance_shrinkage=0.1)
            for value in ("mahalanobis", "knn")]
        return [ApproachCConfig(epochs=3, steps_per_epoch=24, seed=42, **row) for row in selected]
    raise ValueError(f"Unknown approach {approach}")


def _with_budget(config: Any, *, epochs: int, steps: int, seed: int) -> Any:
    return replace(config, epochs=epochs, steps_per_epoch=steps, seed=seed)


def _cap_by_class(x: torch.Tensor, y: torch.Tensor, *, limit: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(seed)
    indices = []
    for class_id in sorted(int(value) for value in y.unique()):
        candidates = np.flatnonzero(y.numpy() == class_id)
        indices.extend(rng.choice(candidates, min(limit, len(candidates)), replace=False).tolist())
    selected = torch.tensor(sorted(indices), dtype=torch.long)
    return x[selected], y[selected]


def _pseudo_split(prepared, holdout: tuple[int, int], inner_fold: int) -> tuple[tuple[int, int], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    splits = list(group_stratified_inner_splits(prepared.outer.train, n_splits=3, seed=prepared.outer.holdout[0] * 100 + prepared.outer.holdout[1]))
    train_indices, validation_indices = splits[inner_fold]
    inner_train_x, inner_train_y = prepared.train_x[train_indices], prepared.train_y[train_indices]
    inner_validation_x, inner_validation_y = prepared.train_x[validation_indices], prepared.train_y[validation_indices]
    seen_faults = sorted(set(int(value) for value in inner_train_y.unique()) - {0})
    pseudo = (seen_faults[0], seen_faults[-1])
    train_mask = ~torch.isin(inner_train_y, torch.tensor(pseudo))
    known_mask = ~torch.isin(inner_validation_y, torch.tensor(pseudo))
    return pseudo, inner_train_x[train_mask], inner_train_y[train_mask], inner_validation_x[known_mask], inner_validation_y[known_mask], inner_validation_x[~known_mask], inner_validation_y[~known_mask]


def inner_trial(*, approach: str, frame: pd.DataFrame, holdout: tuple[int, int], config: Any, device: torch.device,
                physics_path: Path, description_path: Path, study_root: Path, inner_fold: int) -> tuple[dict[str, float], torch.nn.Module, dict[str, Any]]:
    prepared = prepare_fold(frame, holdout=holdout, seed=config.seed)
    pseudo, train_x, train_y, known_x, known_y, pseudo_x, pseudo_y = _pseudo_split(prepared, holdout, inner_fold)
    evaluation_train_x, evaluation_train_y = _cap_by_class(train_x, train_y, limit=2048, seed=config.seed + inner_fold)
    known_x, known_y = _cap_by_class(known_x, known_y, limit=2048, seed=config.seed + 100 + inner_fold)
    pseudo_x, pseudo_y = _cap_by_class(pseudo_x, pseudo_y, limit=2048, seed=config.seed + 200 + inner_fold)
    seen_ids = sorted(int(value) for value in train_y.unique())
    semantic = None
    if approach == "a":
        model, training = train_approach_a(train_x, train_y, known_x, known_y, device=device, config=config)
    elif approach == "b":
        _, _, semantic = semantic_prototypes(mode=config.prototype_mode, physics_path=physics_path,
                                             description_path=description_path, text_model="sentence-transformers/all-mpnet-base-v2",
                                             device=device, cache_dir=study_root / "cache")
        model, training = train_approach_b(train_x, train_y, known_x, known_y, semantic, device=device, config=config)
    else:
        model, training = train_approach_c(train_x, device=device, config=config)
    train_z = encode(model, evaluation_train_x, device=device, kind=approach)
    known_z = encode(model, known_x, device=device, kind=approach)
    pseudo_z = encode(model, pseudo_x, device=device, kind=approach)
    if approach == "c":
        scorer = DensityScorer(train_z, evaluation_train_y, seen_ids, density=config.density,
                               shrinkage=config.covariance_shrinkage, knn_k=config.knn_k)
        known_conf = -scorer.distances(known_z).min(1)
        unknown_conf = -scorer.distances(pseudo_z).min(1)
    elif approach == "b":
        known_conf = cosine_scores(known_z, semantic[seen_ids]).max(1)
        unknown_conf = cosine_scores(pseudo_z, semantic[seen_ids]).max(1)
    else:
        base = class_prototypes(train_z, evaluation_train_y, seen_ids, strategy=config.aggregation, seed=config.seed)
        known_conf = cosine_scores(known_z, base).max(1)
        unknown_conf = cosine_scores(pseudo_z, base).max(1)
    detection_auroc = float(roc_auc_score(np.r_[np.ones(len(known_conf)), np.zeros(len(unknown_conf))], np.r_[known_conf, unknown_conf]))
    if approach == "b":
        strict_scores = cosine_scores(pseudo_z, semantic[list(pseudo)])
        strict_pred = np.asarray([pseudo[index] for index in strict_scores.argmax(1)])
        strict_balanced = float(balanced_accuracy_score(pseudo_y.numpy(), strict_pred))
        all_z = torch.cat([known_z, pseudo_z])
        all_y = np.concatenate([known_y.numpy(), pseudo_y.numpy()])
        pred = cosine_scores(all_z, semantic).argmax(1)
        seen_acc = macro_class_accuracy(all_y, pred, seen_ids)
        unseen_acc = macro_class_accuracy(all_y, pred, pseudo)
        post_h = harmonic(seen_acc, unseen_acc)
        objective = 0.7 * strict_balanced + 0.3 * post_h
    else:
        chosen = []
        pseudo_labels = pseudo_y.numpy()
        for value in pseudo:
            chosen.append(int(np.flatnonzero(pseudo_labels == value)[0]))
        chosen = np.asarray(chosen)
        support_z, support_y = pseudo_z[chosen], pseudo_y[chosen]
        query_mask = np.ones(len(pseudo_z), dtype=bool)
        query_mask[chosen] = False
        query_z, query_y = pseudo_z[query_mask], pseudo_y[query_mask]
        base = class_prototypes(train_z, evaluation_train_y, seen_ids, strategy="prototype")
        enrolled = class_prototypes(support_z, support_y, pseudo, strategy="prototype")
        class_ids = seen_ids + list(pseudo)
        all_z = torch.cat([known_z, query_z])
        all_y = np.concatenate([known_y.numpy(), query_y.numpy()])
        pred = np.asarray([class_ids[index] for index in cosine_scores(all_z, torch.cat([base, enrolled])).argmax(1)])
        seen_acc = macro_class_accuracy(all_y, pred, seen_ids)
        unseen_acc = macro_class_accuracy(all_y, pred, pseudo)
        post_h = harmonic(seen_acc, unseen_acc)
        strict_balanced = float("nan")
        objective = 0.5 * detection_auroc + 0.5 * post_h if approach == "a" else 0.7 * detection_auroc + 0.3 * post_h
    metrics = {"objective": float(objective), "detection_auroc": detection_auroc,
               "post_harmonic_mean": float(post_h), "strict_balanced_accuracy": float(strict_balanced),
               "pseudo_holdout": list(pseudo), "inner_fold": inner_fold}
    return metrics, model, training


def run_sweep(*, approach: str, frame: pd.DataFrame, device: torch.device, study_root: Path,
              physics_path: Path, description_path: Path, resume: bool = True) -> dict[str, Any]:
    state = StudyState(study_root)
    candidates = candidate_configs(approach)
    candidate_ids = {config_hash(asdict(config)): config for config in candidates}
    stages = [
        ("initial", 3, 24, [(PILOT_FOLDS[0], inner) for inner in range(3)], 24, 8),
        ("intermediate", 8, 48, [(fold, inner) for inner, fold in enumerate(PILOT_FOLDS)], 8, 3),
        ("final", 16, 72, [(fold, inner) for fold in PILOT_FOLDS for inner in range(3)], 3, 1),
    ]
    active = list(candidate_ids)
    all_rows: list[dict[str, Any]] = []
    for stage, epochs, steps, fold_specs, limit, keep in stages:
        active = active[:limit]
        stage_rows = []
        for candidate_id in active:
            base = candidate_ids[candidate_id]
            fold_scores = []
            for fold_index, (holdout, inner_fold) in enumerate(fold_specs):
                config = _with_budget(base, epochs=epochs, steps=steps, seed=42 + fold_index)
                trial_id = f"sweep-{approach}-{stage}-{candidate_id}-{holdout[0]:02d}_{holdout[1]:02d}-i{inner_fold}"
                run_dir = study_root / "sweeps" / approach / stage / trial_id
                valid, _ = validate_run(run_dir, {"run_id": trial_id}) if resume else (False, "forced")
                if valid:
                    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
                else:
                    metadata = {"run_id": trial_id, "approach": approach, "stage": stage, "candidate_id": candidate_id,
                                "holdout": list(holdout), "inner_fold": inner_fold, "config": asdict(config)}
                    with state.run(trial_id, run_dir, metadata):
                        metrics, model, training = inner_trial(
                            approach=approach, frame=frame, holdout=holdout, config=config, device=device,
                            physics_path=physics_path, description_path=description_path, study_root=study_root, inner_fold=inner_fold,
                        )
                        metrics["training"] = training
                        save_json(run_dir / "metrics.json", metrics)
                        save_json(run_dir / "config.json", asdict(config))
                        write_manifest(run_dir, metadata)
                        del model
                        torch.cuda.empty_cache()
                fold_scores.append(float(metrics["objective"]))
                row = {"approach": approach, "stage": stage, "candidate_id": candidate_id,
                       "holdout": f"{holdout[0]}-{holdout[1]}", **{k: v for k, v in metrics.items() if k != "training"}}
                all_rows.append(row)
            stage_rows.append({"candidate_id": candidate_id, "mean_objective": float(np.mean(fold_scores)),
                               "min_objective": float(np.min(fold_scores)), "fold_scores": fold_scores})
        stage_rows.sort(key=lambda row: (row["mean_objective"], row["min_objective"]), reverse=True)
        atomic_json(study_root / "sweeps" / approach / f"{stage}_ranking.json", stage_rows)
        active = [row["candidate_id"] for row in stage_rows[:keep]]
    winner_id = active[0]
    winner = _with_budget(candidate_ids[winner_id], epochs=16, steps=72, seed=42)
    frozen = {"approach": approach, "candidate_id": winner_id, "config": asdict(winner), "selection": "successive_halving_inner_only"}
    atomic_json(study_root / "configs" / f"approach_{approach}_frozen.json", frozen)
    pd.DataFrame(all_rows).to_csv(study_root / "sweeps" / approach / "all_trials.csv", index=False)
    state.update(selected_configs={approach: frozen}, note=f"Froze approach {approach} candidate {winner_id} after inner-only pilot selection.")
    return frozen
