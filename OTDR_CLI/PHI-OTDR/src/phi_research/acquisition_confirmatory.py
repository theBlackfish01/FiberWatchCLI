"""Frozen bidirectional acquisition-era open-world confirmation for Phi-OTDR.

This module is intentionally parameter-light.  Its choices are locked in
``config/acquisition_confirmatory_v2.json`` before target-query access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .conditional_alignment import conditional_center_alignment
from .data_contract import CLASS_NAMES, canonical_json_hash
from .embedding_gallery import FeatureEncoder, _encode, _seed, _session_sampler
from .metrics import classification_metrics, harmonic_mean, open_set_metrics
from .session_distribution import (
    _descriptor_matrix,
    _session_windows,
    class_scores,
    feature_masks,
    select_support,
)


def verify_lock(config: Path, sidecar: Path) -> str:
    payload = json.loads(config.read_text(encoding="utf-8"))
    actual = canonical_json_hash(payload)
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    if actual != expected:
        raise ValueError(f"Confirmatory lock hash mismatch: {actual} != {expected}")
    if payload.get("final_query_used_at_freeze") is not False:
        raise ValueError("Confirmatory lock does not attest an unused query at freeze")
    return actual


def class_conditional_conformal(
    calibration_scores: np.ndarray,
    calibration_labels: np.ndarray,
    query_scores: np.ndarray,
    class_ids: list[int],
    *,
    alpha: float = 0.05,
) -> dict[str, np.ndarray]:
    """Mondrian p-values from true-class calibration nonconformity scores."""
    p_values = np.zeros_like(query_scores, dtype=float)
    counts = np.zeros(len(class_ids), dtype=np.int64)
    for column, class_id in enumerate(class_ids):
        reference = -calibration_scores[calibration_labels == class_id, column]
        counts[column] = len(reference)
        query_nonconformity = -query_scores[:, column]
        p_values[:, column] = (
            1.0 + np.sum(reference[None, :] >= query_nonconformity[:, None], axis=1)
        ) / (len(reference) + 1.0)
    included = p_values > alpha
    return {
        "p_values": p_values,
        "included": included,
        "set_sizes": included.sum(axis=1),
        "calibration_counts": counts,
    }


def _fold_seed(seed: int, holdout: int, method: str) -> int:
    value = hashlib.sha256(f"{seed}|{holdout}|{method}".encode()).digest()
    return int.from_bytes(value[:4], "little")


def _train_encoder(
    x: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    partitions: np.ndarray,
    dates: np.ndarray,
    seen: list[int],
    *,
    seed: int,
    weight: float,
    output_dir: Path,
) -> tuple[np.ndarray, dict[str, object]]:
    """Train only on source development data, then encode all frozen rows."""
    _seed(seed)
    device = torch.device("cuda")
    train = (partitions == "source_train") & np.isin(y, seen)
    validation = (partitions == "source_validation") & np.isin(y, seen)
    scaler = StandardScaler().fit(x[train])
    x_train = scaler.transform(x[train]).astype(np.float32)
    x_validation = scaler.transform(x[validation]).astype(np.float32)
    class_map = {class_id: index for index, class_id in enumerate(seen)}
    mapped_train = np.asarray([class_map[int(value)] for value in y[train]], dtype=np.int64)
    mapped_validation = np.asarray([class_map[int(value)] for value in y[validation]], dtype=np.int64)
    checkpoint = output_dir / "best_model.pt"
    result_path = output_dir / "training.json"
    scaler_path = output_dir / "scaler.joblib"
    if checkpoint.exists() and result_path.exists() and scaler_path.exists():
        scaler = joblib.load(scaler_path)
        model = FeatureEncoder(x.shape[1], len(seen)).to(device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        represented = _encode(model, scaler.transform(x).astype(np.float32), device)
        return represented, json.loads(result_path.read_text(encoding="utf-8"))
    domain_names = sorted(set(dates[train].tolist()))
    domain_map = {value: index for index, value in enumerate(domain_names)}
    domains = np.asarray([domain_map[value] for value in dates[train]], dtype=np.int64)
    dataset = TensorDataset(
        torch.from_numpy(x_train), torch.from_numpy(mapped_train), torch.from_numpy(domains)
    )
    sampler = _session_sampler(mapped_train, sessions[train], seed)
    loader = DataLoader(dataset, batch_size=512, sampler=sampler, num_workers=0, pin_memory=True)
    model = FeatureEncoder(x.shape[1], len(seen)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)
    amp = torch.amp.GradScaler("cuda")
    best_f1, best_epoch, stale = -1.0, 0, 0
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, float]] = []
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    for epoch in range(1, 41):
        model.train()
        losses, alignments = [], []
        for values, targets, domain in loader:
            values = values.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            domain = domain.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                embedding, logits = model(values)
                alignment = conditional_center_alignment(embedding, targets, domain)
                loss = criterion(logits, targets) + weight * alignment
            amp.scale(loss).backward()
            amp.step(optimizer)
            amp.update()
            losses.append(float(loss.detach().cpu()))
            alignments.append(float(alignment.detach().cpu()))
        model.eval()
        with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
            _, logits = model(torch.from_numpy(x_validation).to(device))
        predicted = logits.argmax(1).cpu().numpy()
        score = float(f1_score(mapped_validation, predicted, average="macro", zero_division=0))
        history.append({"epoch": epoch, "loss": float(np.mean(losses)),
                        "alignment_loss": float(np.mean(alignments)), "validation_macro_f1": score})
        if score > best_f1:
            best_f1, best_epoch, stale = score, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= 6:
                break
    elapsed = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    represented = _encode(model, scaler.transform(x).astype(np.float32), device)
    joblib.dump(scaler, scaler_path)
    metadata = {
        "weight": weight, "seed": seed, "best_epoch": best_epoch,
        "validation_window_macro_f1": best_f1, "epochs_completed": len(history),
        "training_seconds": elapsed, "history": history,
        "cuda": {"required": True, "device_name": torch.cuda.get_device_name(0),
                 "torch": torch.__version__, "runtime": torch.version.cuda,
                 "precision": "float16 autocast", "peak_bytes": int(torch.cuda.max_memory_allocated())},
    }
    result_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return represented, metadata


def _represent_classical(
    x: np.ndarray, y: np.ndarray, partitions: np.ndarray, seen: list[int], seed: int
) -> np.ndarray:
    train = (partitions == "source_train") & np.isin(y, seen)
    scaler = StandardScaler().fit(x[train])
    scaled_train = scaler.transform(x[train])
    components = min(24, scaled_train.shape[1], len(scaled_train) - 1)
    pca = PCA(n_components=components, random_state=seed).fit(scaled_train)
    return pca.transform(scaler.transform(x)).astype(np.float32)


def _evaluate_representation(
    represented: np.ndarray,
    y: np.ndarray,
    sessions: np.ndarray,
    partitions: np.ndarray,
    window_ids: np.ndarray,
    *,
    seen: list[int],
    holdout: int,
    seed: int,
    support_draws: int,
) -> dict[str, object]:
    # The frozen random projections turn empirical session distributions into
    # equal-length quantile vectors, approximating sliced Wasserstein geometry.
    rng = np.random.default_rng(_fold_seed(seed, holdout, "projections"))
    projections = rng.normal(size=(24, represented.shape[1]))
    projections /= np.maximum(np.linalg.norm(projections, axis=1, keepdims=True), 1e-12)
    rows = _session_windows(
        represented, y, sessions, partitions, window_ids,
        np.ones(len(y), dtype=bool),
    )
    mean_x, session_y, session_ids = _descriptor_matrix(rows, "mean", projections)
    distribution_x, _, _ = _descriptor_matrix(rows, "sliced_wasserstein", projections)
    session_partition = np.asarray([row.partition for row in rows])
    source_train = (session_partition == "source_train") & np.isin(session_y, seen)
    source_calibration = (session_partition == "source_calibration") & np.isin(session_y, seen)
    target_calibration = (session_partition == "target_calibration") & np.isin(session_y, seen)
    support = (session_partition == "target_support") & (session_y == holdout)
    query = session_partition == "target_query"
    gallery_scores = class_scores(mean_x[query], mean_x[source_train], session_y[source_train], seen)
    predicted = np.asarray(seen)[np.argmax(gallery_scores, axis=1)]
    confidence = np.max(gallery_scores, axis=1)
    final_true = session_y[query]
    known = final_true != holdout
    detection: dict[str, object] = {}
    conformal_by_mode: dict[str, np.ndarray] = {}
    for mode, selected in (("target_calibration_primary", target_calibration),
                           ("source_calibration_strict", source_calibration)):
        calibration_scores = class_scores(mean_x[selected], mean_x[source_train], session_y[source_train], seen)
        threshold = float(np.quantile(np.max(calibration_scores, axis=1), 0.05, method="higher"))
        detection[mode] = open_set_metrics(
            confidence, known, predicted == final_true, threshold=threshold
        )
        conformal = class_conditional_conformal(
            calibration_scores, session_y[selected], gallery_scores, seen, alpha=0.05
        )
        conformal_by_mode[mode] = conformal["p_values"]
        set_sizes = conformal["set_sizes"]
        detection[mode]["conformal"] = {
            "calibration_counts": conformal["calibration_counts"].tolist(),
            "empty_set_unknown_recall": float(np.mean(set_sizes[~known] == 0)),
            "known_nonempty_coverage": float(np.mean(set_sizes[known] > 0)),
            "known_singleton_rate": float(np.mean(set_sizes[known] == 1)),
            "mean_set_size": float(np.mean(set_sizes)),
        }

    post: dict[str, object] = {}
    candidates = distribution_x[support]
    candidate_indices = np.flatnonzero(support)
    for shot in (1, 3, 5):
        strategies: dict[str, object] = {}
        for strategy in ("medoid", "random"):
            draws = 1 if strategy == "medoid" else support_draws
            metrics: list[dict[str, float]] = []
            chosen_sessions: list[list[str]] = []
            for draw in range(draws):
                indices = select_support(
                    candidates, distribution_x[source_train], strategy=strategy, shot=shot,
                    seed=_fold_seed(seed + draw, holdout, f"{strategy}_{shot}"),
                )
                selected = candidate_indices[indices]
                gallery_x = np.concatenate((distribution_x[source_train], distribution_x[selected]))
                gallery_y = np.concatenate((session_y[source_train], session_y[selected]))
                scores = class_scores(distribution_x[query], gallery_x, gallery_y, seen + [holdout])
                post_predicted = np.asarray(seen + [holdout])[np.argmax(scores, axis=1)]
                base_accuracy = float(np.mean(post_predicted[known] == final_true[known]))
                enrolled_recall = float(np.mean(post_predicted[~known] == holdout))
                metrics.append({"base_accuracy": base_accuracy, "enrolled_recall": enrolled_recall,
                                "enrollment_h": harmonic_mean(base_accuracy, enrolled_recall)})
                chosen_sessions.append(session_ids[selected].astype(str).tolist())
            strategies[strategy] = {
                "draws": draws,
                "mean": {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]},
                "standard_deviation": {
                    key: float(np.std([row[key] for row in metrics], ddof=1)) if draws > 1 else 0.0
                    for key in metrics[0]
                },
                "minimum_enrollment_h": float(np.min([row["enrollment_h"] for row in metrics])),
                "support_sessions": chosen_sessions,
                "draw_metrics": metrics,
            }
        post[str(shot)] = strategies
    return {
        "holdout_class": CLASS_NAMES[holdout], "holdout_class_id": holdout,
        "query_sessions": int(np.sum(query)), "known_query_sessions": int(np.sum(known)),
        "unknown_query_sessions": int(np.sum(~known)), "support_candidates": int(np.sum(support)),
        "detection": detection, "enrollment": post,
        "query_classification": classification_metrics(final_true[known], predicted[known]),
        "query_records": [
            {
                "session_id": str(session_id), "true_class_id": int(true),
                "true_class": CLASS_NAMES[int(true)], "is_known": bool(known_value),
                "predicted_seen_class_id": int(prediction),
                "predicted_seen_class": CLASS_NAMES[int(prediction)],
                "confidence": float(confidence_value),
                "class_ids": seen,
                "class_scores": score_row.astype(float).tolist(),
                "target_calibration_conformal_pvalues": conformal_by_mode[
                    "target_calibration_primary"
                ][index].astype(float).tolist(),
                "source_calibration_conformal_pvalues": conformal_by_mode[
                    "source_calibration_strict"
                ][index].astype(float).tolist(),
            }
            for index, (session_id, true, known_value, prediction, confidence_value, score_row)
            in enumerate(zip(session_ids[query], final_true, known, predicted, confidence, gallery_scores, strict=True))
        ],
    }


def _summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    summary: dict[str, object] = {}
    for method in sorted(set(str(row["method"]) for row in rows)):
        selected = [row for row in rows if row["method"] == method]
        method_summary: dict[str, object] = {"folds": len(selected)}
        for metric in ("unknown_auroc", "detection_h", "oscr", "known_classification_accuracy"):
            values = np.asarray([row["detection"]["target_calibration_primary"][metric] for row in selected])
            method_summary[metric] = {"mean": float(values.mean()), "std": float(values.std(ddof=1)),
                                      "minimum": float(values.min())}
        for shot in (1, 3, 5):
            for strategy in ("medoid", "random"):
                values = np.asarray([row["enrollment"][str(shot)][strategy]["mean"]["enrollment_h"]
                                     for row in selected])
                method_summary[f"{shot}shot_{strategy}_enrollment_h"] = {
                    "mean": float(values.mean()), "std": float(values.std(ddof=1)), "minimum": float(values.min())
                }
        summary[method] = method_summary
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-features", type=Path, required=True)
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--support-draws", type=int, default=30)
    args = parser.parse_args()
    lock_hash = verify_lock(args.config, args.hash)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for aligned confirmatory models")
    development = np.load(args.development_features, allow_pickle=False)
    final = np.load(args.query_features, allow_pickle=False)
    if "target_query" in set(development["partitions"].astype(str).tolist()):
        raise ValueError("Development bundle contains target query")
    if set(final["partitions"].astype(str).tolist()) != {"target_query"}:
        raise ValueError("Query bundle is not exclusively target_query")
    if not np.array_equal(development["feature_names"], final["feature_names"]):
        raise ValueError("Feature schemas disagree")
    keys = ("features", "labels", "sessions", "partitions", "window_ids", "date_tokens")
    combined = {key: np.concatenate((development[key], final[key])) for key in keys}
    names = development["feature_names"].astype(str)
    mask = feature_masks(names)["dynamics"]
    x = combined["features"][:, mask].astype(np.float32)
    y = combined["labels"].astype(np.int64)
    sessions = combined["sessions"].astype(str)
    partitions = combined["partitions"].astype(str)
    window_ids = combined["window_ids"].astype(np.int32)
    dates = combined["date_tokens"].astype(str)
    rows: list[dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    for seed in (20260805, 20260806, 20260807):
        for holdout in range(len(CLASS_NAMES)):
            seen = [value for value in range(len(CLASS_NAMES)) if value != holdout]
            classical = _represent_classical(x, y, partitions, seen, seed)
            result = _evaluate_representation(
                classical, y, sessions, partitions, window_ids, seen=seen, holdout=holdout,
                seed=seed, support_draws=args.support_draws,
            )
            result.update({"method": "pca24", "seed": seed})
            rows.append(result)
            for weight in (0.0, 100.0):
                model_dir = args.output_dir / "models" / f"seed{seed}" / CLASS_NAMES[holdout] / f"weight_{weight:g}"
                represented, training = _train_encoder(
                    x, y, sessions, partitions, dates, seen, seed=seed, weight=weight,
                    output_dir=model_dir,
                )
                result = _evaluate_representation(
                    represented, y, sessions, partitions, window_ids, seen=seen, holdout=holdout,
                    seed=seed, support_draws=args.support_draws,
                )
                result.update({"method": f"aligned_weight_{weight:g}", "seed": seed,
                               "training": {key: training[key] for key in (
                                   "best_epoch", "validation_window_macro_f1", "training_seconds", "cuda")}})
                rows.append(result)
                print(f"seed={seed} holdout={CLASS_NAMES[holdout]} weight={weight:g} "
                      f"H={result['detection']['target_calibration_primary']['detection_h']:.3f} "
                      f"5shot={result['enrollment']['5']['medoid']['mean']['enrollment_h']:.3f}", flush=True)
    payload = {
        "schema_version": "phi-acquisition-confirmatory-results-v2",
        "lock_hash": lock_hash, "final_query_used": True,
        "elapsed_seconds": time.perf_counter() - started,
        "cuda": {"required": True, "device": torch.cuda.get_device_name(0),
                 "torch": torch.__version__, "runtime": torch.version.cuda},
        "rows": rows, "summary": _summarize(rows),
    }
    (args.output_dir / "confirmatory_open_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
