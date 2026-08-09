"""Statistical and nuisance-probe analysis for frozen Phi-OTDR v3 spatial runs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_contract import CLASS_NAMES, canonical_json_hash


PRIMARY = ("registered_position", "temporal_difference_energy", "dynamics")
CONTROLS = {
    "absolute_dynamics": ("absolute", "none", "dynamics"),
    "absolute_fused": ("absolute", "none", "fused"),
    "invariant_fused": ("invariant", "none", "fused"),
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _spec(row: dict[str, str]) -> tuple[str, str, str]:
    return row["view"], row["estimator"], row["ablation"]


def _macro(labels: np.ndarray, predictions: np.ndarray) -> float:
    return float(f1_score(labels, predictions, average="macro", zero_division=0))


def _stratified_bootstrap(
    labels: np.ndarray,
    primary: np.ndarray,
    control: np.ndarray,
    *,
    draws: int = 5000,
    seed: int = 20260808,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    class_indices = [np.flatnonzero(labels == class_id) for class_id in range(len(CLASS_NAMES))]
    observed = _macro(labels, primary) - _macro(labels, control)
    deltas = np.empty(draws, dtype=np.float64)
    for draw in range(draws):
        sampled = np.concatenate(
            [rng.choice(indices, size=len(indices), replace=True) for indices in class_indices]
        )
        deltas[draw] = _macro(labels[sampled], primary[sampled]) - _macro(labels[sampled], control[sampled])
    return {
        "observed_delta_macro_f1": observed,
        "ci95_low": float(np.quantile(deltas, 0.025)),
        "ci95_high": float(np.quantile(deltas, 0.975)),
        "bootstrap_probability_delta_le_zero": float(np.mean(deltas <= 0.0)),
        "draws": draws,
    }


def _prediction_map(rows: Sequence[dict[str, str]], direction: str, spec: tuple[str, str, str]) -> dict[str, tuple[int, int]]:
    return {
        row["session_id"]: (int(row["true_label"]), int(row["predicted_label"]))
        for row in rows
        if row["direction"] == direction and _spec(row) == spec
    }


def _paired_comparisons(prediction_path: Path, model_name: str) -> list[dict[str, object]]:
    rows = [row for row in _read_csv(prediction_path) if row["model"] == model_name]
    directions = sorted({row["direction"] for row in rows})
    output = []
    for direction in directions:
        primary = _prediction_map(rows, direction, PRIMARY)
        for control_name, control_spec in CONTROLS.items():
            control = _prediction_map(rows, direction, control_spec)
            shared = sorted(set(primary) & set(control))
            if not shared:
                continue
            labels = np.asarray([primary[session][0] for session in shared], dtype=np.int64)
            if any(primary[session][0] != control[session][0] for session in shared):
                raise ValueError("Paired prediction labels disagree")
            comparison = _stratified_bootstrap(
                labels,
                np.asarray([primary[session][1] for session in shared]),
                np.asarray([control[session][1] for session in shared]),
            )
            output.append(
                {
                    "model": model_name,
                    "direction": direction,
                    "primary": ":".join(PRIMARY),
                    "control": control_name,
                    "sessions": len(shared),
                    **comparison,
                }
            )
    return output


def _bh_qvalues(p_values: Sequence[float]) -> list[float]:
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    ranked = values[order] * len(values) / np.arange(1, len(values) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty_like(values)
    result[order] = np.clip(ranked, 0.0, 1.0)
    return result.tolist()


def _metadata(bundle: dict[str, np.ndarray]) -> dict[str, dict[str, object]]:
    result = {}
    for index, session in enumerate(bundle["sessions"].astype(str)):
        if session not in result:
            result[session] = {
                "label": int(bundle["labels"][index]),
                "era": str(bundle["eras"][index]),
                "date": str(bundle["date_tokens"][index]),
                "source": str(bundle["source_tokens"][index]),
            }
    return result


def _era_probe(features: np.ndarray, sessions: np.ndarray, metadata: dict[str, dict[str, object]]) -> dict[str, object]:
    era = np.asarray([0 if metadata[s]["era"] == "january" else 1 for s in sessions], dtype=np.int64)
    dates = np.asarray([metadata[s]["date"] for s in sessions])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=20260808)
    folds = []
    predictions = np.full(len(era), -1, dtype=np.int64)
    for fold, (train, test) in enumerate(splitter.split(features, era, groups=dates)):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="lbfgs"),
        )
        model.fit(features[train], era[train])
        predictions[test] = model.predict(features[test])
        folds.append(
            {
                "fold": fold,
                "test_dates": sorted(set(dates[test].tolist())),
                "test_sessions": int(len(test)),
                "balanced_accuracy": float(balanced_accuracy_score(era[test], predictions[test])),
            }
        )
    if np.any(predictions < 0):
        raise AssertionError("Era probe did not predict every session")
    return {
        "grouping": "five-fold stratified group CV by acquisition date",
        "balanced_accuracy": float(balanced_accuracy_score(era, predictions)),
        "folds": folds,
    }


def _source_block_stress(
    features: np.ndarray,
    sessions: np.ndarray,
    metadata: dict[str, dict[str, object]],
    manifest: dict[str, object],
) -> dict[str, object]:
    rows = {str(row["session_id"]): row for row in manifest["sessions"]}
    mask = np.asarray(
        [rows[session]["partition"] in {"source_train", "source_validation"} for session in sessions]
    )
    x = features[mask]
    local_sessions = sessions[mask]
    labels = np.asarray([metadata[s]["label"] for s in local_sessions], dtype=np.int64)
    dates = np.asarray([metadata[s]["date"] for s in local_sessions])
    folds = []
    weighted_scores = []
    logo = LeaveOneGroupOut()
    for train, test in logo.split(x, labels, groups=dates):
        train_classes = set(labels[train].tolist())
        eligible = np.asarray([label in train_classes for label in labels[test]])
        impossible = sorted(set(labels[test][~eligible].tolist()))
        if not np.any(eligible) or len(train_classes) < 2:
            continue
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=3000, solver="lbfgs"),
        )
        model.fit(x[train], labels[train])
        prediction = model.predict(x[test][eligible])
        score = _macro(labels[test][eligible], prediction)
        weighted_scores.extend([score] * int(np.sum(eligible)))
        folds.append(
            {
                "held_date": str(dates[test][0]),
                "eligible_sessions": int(np.sum(eligible)),
                "total_sessions": int(len(test)),
                "impossible_unseen_classes": [CLASS_NAMES[i] for i in impossible],
                "eligible_macro_f1": score,
            }
        )
    return {
        "warning": (
            "Some classes occur on only one source date; their held-date fold is zero-shot and is "
            "excluded from the supervised score but reported as impossible."
        ),
        "eligible_session_weighted_macro_f1": float(np.mean(weighted_scores)),
        "eligible_sessions_across_folds": len(weighted_scores),
        "folds": folds,
    }


def analyze(
    *,
    bundle_path: Path,
    session_view_dir: Path,
    manifests: Sequence[Path],
    logistic_predictions: Path,
    tree_predictions: Path,
    output_path: Path,
) -> dict[str, object]:
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    metadata = _metadata(bundle)
    comparisons = _paired_comparisons(logistic_predictions, "logistic")
    comparisons.extend(_paired_comparisons(tree_predictions, "hist_gradient"))
    q_values = _bh_qvalues([row["bootstrap_probability_delta_le_zero"] for row in comparisons])
    for row, q_value in zip(comparisons, q_values, strict=True):
        row["bh_qvalue_one_sided"] = q_value
    manifest_payloads = [json.loads(path.read_text(encoding="utf-8")) for path in manifests]
    view_specs = {
        "absolute_dynamics": ("absolute", "none", "dynamics"),
        "invariant_fused": ("invariant", "none", "fused"),
        "registered_position_difference_dynamics": PRIMARY,
    }
    probes = {}
    block_stress = defaultdict(dict)
    for name, spec in view_specs.items():
        cache_path = session_view_dir / f"{spec[0]}__{spec[1]}__{spec[2]}.npz"
        with np.load(cache_path, allow_pickle=False) as cached:
            features = cached["features"]
            sessions = cached["sessions"].astype(str)
        probes[name] = _era_probe(features, sessions, metadata)
        for manifest in manifest_payloads:
            direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
            block_stress[name][direction] = _source_block_stress(
                features, sessions, metadata, manifest
            )

    names = bundle["feature_names"].astype(str).tolist()
    center_summary = {}
    for profile in ("difference", "robust", "spectral", "consensus"):
        centers = bundle["features"][:, names.index(f"{profile}_center")]
        confidence = bundle["features"][:, names.index(f"{profile}_confidence")]
        center_summary[profile] = {
            "window_center_median": float(np.median(centers)),
            "window_center_q10": float(np.quantile(centers, 0.10)),
            "window_center_q90": float(np.quantile(centers, 0.90)),
            "confidence_median": float(np.median(confidence)),
            "active_fraction_confidence_ge_0_02": float(np.mean(confidence >= 0.02)),
            "large_shift_fraction_abs_ge_2_channels": float(
                np.mean((confidence >= 0.02) & (np.abs(5.5 - centers) >= 2.0))
            ),
        }
    payload: dict[str, object] = {
        "schema_version": 1,
        "evidence_status": "retrospective development",
        "primary_spec": ":".join(PRIMARY),
        "paired_session_bootstrap": comparisons,
        "era_probes": probes,
        "source_leave_date_out_stress": dict(block_stress),
        "registration_center_diagnostics": center_summary,
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--session-view-dir", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--logistic-predictions", type=Path, required=True)
    parser.add_argument("--tree-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(
        bundle_path=args.bundle,
        session_view_dir=args.session_view_dir,
        manifests=args.manifests,
        logistic_predictions=args.logistic_predictions,
        tree_predictions=args.tree_predictions,
        output_path=args.output,
    )
    print(json.dumps({"payload_sha256": result["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
