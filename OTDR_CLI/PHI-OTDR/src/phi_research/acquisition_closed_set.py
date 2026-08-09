"""Frozen closed-set feature controls on acquisition-era target queries."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import numpy as np

from .acquisition_confirmatory import verify_lock
from .confirmatory_supervised import evaluate_probabilities


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--query-features", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    lock_hash = verify_lock(args.config, args.hash)
    query = np.load(args.query_features, allow_pickle=False)
    if set(query["partitions"].astype(str).tolist()) != {"target_query"}:
        raise ValueError("Input is not exclusively target_query")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: dict[str, object] = {}
    for view in ("amplitude", "dynamics", "full"):
        for estimator in ("logistic", "extra_trees", "hist_gradient_boosting"):
            key = f"{view}__{estimator}"
            saved = joblib.load(args.model_dir / f"{key}.joblib")
            if not np.array_equal(saved["feature_names"].astype(str), query["feature_names"].astype(str)):
                raise ValueError(f"Feature schema mismatch for {key}")
            started = time.perf_counter()
            probabilities = saved["model"].predict_proba(query["features"][:, saved["feature_mask"]])
            result, arrays = evaluate_probabilities(
                query["labels"].astype(np.int64), query["sessions"].astype(str), probabilities
            )
            result["inference_seconds"] = time.perf_counter() - started
            results[key] = result
            np.savez_compressed(args.output_dir / f"{key}_predictions.npz", **arrays)
            print(f"{key}: session F1={result['session_metrics']['macro_f1']:.4f}", flush=True)
    payload = {
        "schema_version": "phi-acquisition-closed-set-v2", "lock_hash": lock_hash,
        "final_query_used": True, "query_sessions": int(len(np.unique(query["sessions"].astype(str)))),
        "query_windows": int(len(query["labels"])), "results": results,
        "best_by_session_macro_f1_for_description_only": max(
            results, key=lambda key: results[key]["session_metrics"]["macro_f1"]
        ),
        "selection_warning": "All nine frozen controls are reported; the descriptive maximum is not a new selected model.",
    }
    (args.output_dir / "confirmatory_closed_set_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
