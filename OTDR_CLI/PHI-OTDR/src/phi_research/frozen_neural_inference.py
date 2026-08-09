"""CUDA-only inference for frozen acquisition-era neural checkpoints.

The full recovered target partition is evaluated once.  Metrics for the
original audited inventory are then derived from the same predictions so the
historical and recovered-data cohorts cannot diverge through separate model
runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .data_contract import CLASS_NAMES, canonical_json_hash
from .metrics import aggregate_session_predictions, classification_metrics
from .neural_baseline import _evaluate, build_model, seed_everything
from .neural_data import ManifestWindowDataset


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_inventory_paths(path: Path) -> set[str]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "rel_path" not in reader.fieldnames:
            raise ValueError(f"Inventory has no rel_path column: {path}")
        rows = {str(row["rel_path"]).replace("\\", "/") for row in reader}
    if not rows:
        raise ValueError(f"Inventory is empty: {path}")
    return rows


def path_fingerprint(paths: np.ndarray) -> str:
    ordered = "\n".join(sorted(str(value).replace("\\", "/") for value in paths))
    return hashlib.sha256(ordered.encode("utf-8")).hexdigest()


def cohort_summary(
    y_true: np.ndarray,
    predictions: np.ndarray,
    probabilities: np.ndarray,
    sessions: np.ndarray,
    paths: np.ndarray,
    mask: np.ndarray,
) -> tuple[dict[str, object], dict[str, np.ndarray]]:
    selected = np.asarray(mask, dtype=bool)
    if selected.shape != y_true.shape or not np.any(selected):
        raise ValueError("Cohort mask must select at least one evaluated window")
    cohort_true = y_true[selected]
    cohort_pred = predictions[selected]
    cohort_probabilities = probabilities[selected]
    cohort_sessions = sessions[selected]
    cohort_paths = paths[selected]
    session_true, session_pred, session_ids = aggregate_session_predictions(
        cohort_true, cohort_sessions, probabilities=cohort_probabilities
    )
    class_counts = Counter(int(value) for value in cohort_true)
    summary = {
        "window_metrics": classification_metrics(cohort_true, cohort_pred),
        "session_metrics": classification_metrics(session_true, session_pred),
        "window_count": int(len(cohort_true)),
        "session_count": int(len(session_ids)),
        "class_window_counts": {
            CLASS_NAMES[class_id]: int(class_counts.get(class_id, 0))
            for class_id in range(len(CLASS_NAMES))
        },
        "rel_path_set_sha256": path_fingerprint(cohort_paths),
    }
    arrays = {
        "session_ids": np.asarray(session_ids),
        "session_true": session_true,
        "session_pred": session_pred,
    }
    return summary, arrays


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--complete-audit", type=Path, required=True)
    parser.add_argument("--reference-inventory", type=Path, required=True)
    parser.add_argument("--reference-audit", type=Path, required=True)
    parser.add_argument("--dataset-contract", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--partition", default="target_query")
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is mandatory for frozen Phi-OTDR neural inference")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")

    complete_audit = _load_json(args.complete_audit)
    reference_audit = _load_json(args.reference_audit)
    manifest = _load_json(args.manifest)
    checkpoint_sha256 = file_sha256(args.checkpoint)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    required_checkpoint_fields = {
        "state_dict", "model", "model_config", "normalization", "temporal_pool", "seed",
        "train_partition", "validation_partition",
    }
    missing_fields = sorted(required_checkpoint_fields - set(saved))
    if missing_fields:
        raise ValueError(f"Checkpoint lacks required fields: {missing_fields}")
    if saved["train_partition"] != "source_train" or saved["validation_partition"] != "source_validation":
        raise ValueError("Checkpoint was not selected with the frozen source-era partitions")

    model_name = str(saved["model"])
    model, runtime_config = build_model(model_name)
    if canonical_json_hash(runtime_config) != canonical_json_hash(saved["model_config"]):
        raise ValueError("Runtime architecture differs from checkpoint model_config")

    seed = int(saved["seed"])
    seed_everything(seed)
    device = torch.device("cuda")
    model.load_state_dict(saved["state_dict"])
    model = model.to(device)

    dataset = ManifestWindowDataset(
        args.data_root,
        args.manifest,
        (args.partition,),
        normalization=str(saved["normalization"]),
        temporal_pool=int(saved["temporal_pool"]),
    )
    if not dataset.samples:
        raise ValueError(f"No samples found for partition {args.partition!r}")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    evaluation = _evaluate(model, model_name, loader, device, nn.CrossEntropyLoss())
    elapsed = time.perf_counter() - started
    peak_memory = int(torch.cuda.max_memory_allocated(device))

    paths = evaluation["paths"].astype(str)
    reference_paths = load_inventory_paths(args.reference_inventory)
    reference_mask = np.asarray(
        [path.replace("\\", "/") in reference_paths for path in paths], dtype=bool
    )
    expanded_mask = np.ones(len(paths), dtype=bool)
    recovered_mask = ~reference_mask
    common_args = (
        evaluation["y_true"],
        evaluation["predictions"],
        evaluation["probabilities"],
        evaluation["sessions"].astype(str),
        paths,
    )
    reference_summary, reference_sessions = cohort_summary(*common_args, reference_mask)
    expanded_summary, expanded_sessions = cohort_summary(*common_args, expanded_mask)
    recovered_counts = Counter(int(value) for value in evaluation["y_true"][recovered_mask])

    contract = None
    if args.dataset_contract is not None:
        contract = {
            "path": args.dataset_contract.as_posix(),
            "sha256": file_sha256(args.dataset_contract),
            "payload_sha256": canonical_json_hash(_load_json(args.dataset_contract)),
        }
    artifact = {
        "schema_version": 1,
        "protocol": "frozen acquisition-era neural target inference",
        "inference_only": True,
        "model_selection_or_fitting_performed": False,
        "partition": args.partition,
        "direction": manifest.get("direction"),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "checkpoint": {
            "path": args.checkpoint.as_posix(),
            "sha256": checkpoint_sha256,
            "model": model_name,
            "model_config": saved["model_config"],
            "normalization": saved["normalization"],
            "temporal_pool": int(saved["temporal_pool"]),
            "seed": seed,
            "selected_epoch": int(saved.get("epoch", -1)),
            "source_validation_macro_f1": float(saved.get("validation_macro_f1", float("nan"))),
        },
        "dataset": {
            "complete_fingerprint_sha256": complete_audit.get("dataset_fingerprint_sha256"),
            "complete_audit_sha256": file_sha256(args.complete_audit),
            "reference_fingerprint_sha256": reference_audit.get("dataset_fingerprint_sha256"),
            "reference_audit_sha256": file_sha256(args.reference_audit),
            "reference_inventory_sha256": file_sha256(args.reference_inventory),
            "dataset_contract": contract,
        },
        "cohorts": {
            "reference_inventory": reference_summary,
            "complete_recovered": expanded_summary,
        },
        "recovered_query_delta": {
            "window_count": int(np.sum(recovered_mask)),
            "session_count": int(len(set(evaluation["sessions"][recovered_mask].astype(str)))),
            "class_window_counts": {
                CLASS_NAMES[class_id]: int(recovered_counts.get(class_id, 0))
                for class_id in range(len(CLASS_NAMES))
            },
            "rel_paths": sorted(paths[recovered_mask].tolist()),
        },
        "inference": {
            "loss_on_complete_query": float(evaluation["loss"]),
            "seconds": elapsed,
            "peak_cuda_memory_bytes": peak_memory,
            "batch_size": args.batch_size,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(0),
            "cuda_capability": list(torch.cuda.get_device_capability(0)),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
            "pid": os.getpid(),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "frozen_target_inference.json"
    prediction_path = args.output_dir / "frozen_target_predictions.npz"
    result_path.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    np.savez_compressed(
        prediction_path,
        y_true=evaluation["y_true"],
        logits=evaluation["logits"],
        probabilities=evaluation["probabilities"],
        predictions=evaluation["predictions"],
        sessions=evaluation["sessions"],
        rel_paths=paths,
        reference_inventory_mask=reference_mask,
        recovered_query_mask=recovered_mask,
        expanded_session_ids=expanded_sessions["session_ids"],
        expanded_session_true=expanded_sessions["session_true"],
        expanded_session_pred=expanded_sessions["session_pred"],
        reference_session_ids=reference_sessions["session_ids"],
        reference_session_true=reference_sessions["session_true"],
        reference_session_pred=reference_sessions["session_pred"],
    )
    print(json.dumps({
        "result": result_path.as_posix(),
        "predictions": prediction_path.as_posix(),
        "model": model_name,
        "direction": manifest.get("direction"),
        "reference": reference_summary,
        "complete": expanded_summary,
        "recovered_query_windows": int(np.sum(recovered_mask)),
        "inference_seconds": elapsed,
        "peak_cuda_memory_bytes": peak_memory,
    }, indent=2))


if __name__ == "__main__":
    main()
