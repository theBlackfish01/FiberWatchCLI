"""Session-safe conventional baselines for deterministic signal features."""

from __future__ import annotations

import argparse
import copy
import json
import platform
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import aggregate_session_predictions, classification_metrics


def _feature_masks(names: np.ndarray) -> dict[str, np.ndarray]:
    names = names.astype(str)
    amplitude_prefixes = ("raw_",)
    amplitude_globals = {"global_mean", "global_std", "global_range"}
    amplitude = np.asarray([
        name.startswith(amplitude_prefixes) or name in amplitude_globals for name in names
    ])
    dynamics = ~amplitude
    return {
        "amplitude": amplitude,
        "dynamics": dynamics,
        "full": np.ones(len(names), dtype=bool),
    }


def _models(seed: int) -> dict[str, object]:
    return {
        "dummy_prior": DummyClassifier(strategy="prior"),
        "logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=1.0,
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=400,
            max_features="sqrt",
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=250,
            max_leaf_nodes=31,
            l2_regularization=1.0,
            class_weight="balanced",
            random_state=seed,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument("--train-partition", default="train")
    parser.add_argument("--validation-partition", default="validation")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bundle = np.load(args.features, allow_pickle=False)
    features = bundle["features"]
    labels = bundle["labels"]
    sessions = bundle["sessions"].astype(str)
    partitions = bundle["partitions"].astype(str)
    feature_names = bundle["feature_names"].astype(str)
    if np.any(partitions == "final_query"):
        raise ValueError("Development baseline must not receive final_query features")
    train = partitions == args.train_partition
    validation = partitions == args.validation_partition
    if not np.any(train) or not np.any(validation):
        raise ValueError(
            f"Empty train/validation selection: {args.train_partition!r}, "
            f"{args.validation_partition!r}; available={sorted(set(partitions.tolist()))}"
        )
    masks = _feature_masks(feature_names)
    model_factories = _models(args.seed)
    results: list[dict[str, object]] = []
    predictions: dict[str, np.ndarray] = {}

    for ablation, feature_mask in masks.items():
        for model_name, prototype in model_factories.items():
            model = copy.deepcopy(prototype)
            started = time.perf_counter()
            model.fit(features[train][:, feature_mask], labels[train])
            fit_seconds = time.perf_counter() - started
            started = time.perf_counter()
            predicted = model.predict(features[validation][:, feature_mask])
            probabilities = model.predict_proba(features[validation][:, feature_mask])
            inference_seconds = time.perf_counter() - started
            window_metrics = classification_metrics(labels[validation], predicted)
            session_true, session_pred, ordered_sessions = aggregate_session_predictions(
                labels[validation], sessions[validation], probabilities=probabilities
            )
            session_metrics = classification_metrics(session_true, session_pred)
            key = f"{ablation}__{model_name}"
            predictions[f"{key}__window_pred"] = predicted.astype(np.int64)
            predictions[f"{key}__window_prob"] = probabilities.astype(np.float32)
            predictions[f"{key}__session_true"] = session_true
            predictions[f"{key}__session_pred"] = session_pred
            results.append(
                {
                    "key": key,
                    "ablation": ablation,
                    "model": model_name,
                    "feature_count": int(np.sum(feature_mask)),
                    "fit_seconds": fit_seconds,
                    "validation_inference_seconds": inference_seconds,
                    "window_metrics": window_metrics,
                    "session_metrics": session_metrics,
                    "validation_sessions": ordered_sessions,
                }
            )
            joblib.dump(
                {"model": model, "feature_mask": feature_mask, "feature_names": feature_names},
                args.output_dir / f"{key}.joblib",
            )
            print(
                f"[{key}] window_macro_f1={window_metrics['macro_f1']:.4f} "
                f"session_macro_f1={session_metrics['macro_f1']:.4f}",
                flush=True,
            )

    ranked = sorted(
        results,
        key=lambda row: (
            float(row["session_metrics"]["macro_f1"]),
            float(row["window_metrics"]["macro_f1"]),
        ),
        reverse=True,
    )
    payload = {
        "protocol": "session-safe development; selected train fit and validation evaluation only",
        "seed": args.seed,
        "train_partition": args.train_partition,
        "validation_partition": args.validation_partition,
        "train_windows": int(np.sum(train)),
        "validation_windows": int(np.sum(validation)),
        "train_sessions": int(len(np.unique(sessions[train]))),
        "validation_sessions": int(len(np.unique(sessions[validation]))),
        "selected_by_validation": ranked[0]["key"],
        "ranking": [row["key"] for row in ranked],
        "results": results,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    (args.output_dir / "development_results.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    np.savez_compressed(
        args.output_dir / "validation_predictions.npz",
        y_true=labels[validation],
        sessions=sessions[validation],
        **predictions,
    )
    print(json.dumps({"selected": ranked[0]["key"], "ranking": payload["ranking"]}, indent=2))


if __name__ == "__main__":
    main()
