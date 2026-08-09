"""Source-only channel-shift augmentation control for Phi-OTDR v3."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_contract import canonical_json_hash
from .morphology_features import aggregate_sessions, transform_view
from .spatial_experiment import _metrics, _select_temperature, _temperature


def _session_features(
    bundle: dict[str, np.ndarray], *, ablation: str, shift: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = bundle["feature_names"].astype(str).tolist()
    rows = [
        transform_view(
            row,
            names,
            view="absolute",
            estimator="multi_estimator_consensus",
            ablation=ablation,
            forced_shift=shift,
        )[0]
        for row in bundle["features"]
    ]
    features, sessions, _ = aggregate_sessions(
        np.asarray(rows), bundle["sessions"].astype(str), bundle["window_ids"]
    )
    first = {session: i for i, session in enumerate(bundle["sessions"].astype(str))}
    labels = np.asarray([bundle["labels"][first[session]] for session in sessions], dtype=np.int64)
    return features, sessions, labels


def run(
    *,
    bundle_path: Path,
    manifests: Sequence[Path],
    protocol_path: Path,
    protocol_hash_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    protocol_hash = protocol_hash_path.read_text(encoding="utf-8").split()[0]
    if canonical_json_hash(protocol) != protocol_hash:
        raise ValueError("V3 protocol hash mismatch")
    with np.load(bundle_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "session_shift_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    results = []
    started = time.perf_counter()
    for ablation in ("dynamics", "fused"):
        shifted = {}
        sessions = labels = None
        for shift in (-2.0, -1.0, 0.0, 1.0, 2.0):
            cache_path = cache_dir / f"{ablation}_shift_{shift:+.0f}.npz"
            if cache_path.is_file():
                with np.load(cache_path, allow_pickle=False) as cached:
                    features = cached["features"]
                    current_sessions = cached["sessions"].astype(str)
                    current_labels = cached["labels"].astype(np.int64)
            else:
                features, current_sessions, current_labels = _session_features(
                    bundle, ablation=ablation, shift=shift
                )
                np.savez_compressed(
                    cache_path,
                    features=features,
                    sessions=current_sessions,
                    labels=current_labels,
                )
            if sessions is None:
                sessions, labels = current_sessions, current_labels
            elif not np.array_equal(sessions, current_sessions) or not np.array_equal(labels, current_labels):
                raise AssertionError("Shifted session caches are misaligned")
            shifted[shift] = features
        assert sessions is not None and labels is not None
        for manifest_path in manifests:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
            partitions = np.asarray([session_rows[session]["partition"] for session in sessions])
            direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
            train = partitions == "source_train"
            validation = partitions == "source_validation"
            calibration = partitions == "source_calibration"
            query = partitions == "target_query"
            for augmentation_name, shifts in (
                ("plus_minus_1", (-1.0, 0.0, 1.0)),
                ("plus_minus_1_2", (-2.0, -1.0, 0.0, 1.0, 2.0)),
            ):
                x_train = np.concatenate([shifted[shift][train] for shift in shifts])
                y_train = np.concatenate([labels[train] for _ in shifts])
                selection = []
                best = None
                for c in (0.01, 0.1, 1.0, 10.0):
                    model = make_pipeline(
                        StandardScaler(),
                        LogisticRegression(
                            C=c,
                            class_weight="balanced",
                            max_iter=4000,
                            solver="lbfgs",
                            random_state=20260808,
                        ),
                    )
                    model.fit(x_train, y_train)
                    score = f1_score(
                        labels[validation],
                        model.predict(shifted[0.0][validation]),
                        average="macro",
                        zero_division=0,
                    )
                    selection.append({"C": c, "source_validation_macro_f1": float(score)})
                    candidate = (float(score), -c)
                    if best is None or candidate > best[0]:
                        best = (candidate, c)
                assert best is not None
                c = best[1]
                development = train | validation
                final_x = np.concatenate([shifted[shift][development] for shift in shifts])
                final_y = np.concatenate([labels[development] for _ in shifts])
                final = make_pipeline(
                    StandardScaler(),
                    LogisticRegression(
                        C=c,
                        class_weight="balanced",
                        max_iter=4000,
                        solver="lbfgs",
                        random_state=20260808,
                    ),
                )
                final.fit(final_x, final_y)
                calibration_probs = final.predict_proba(shifted[0.0][calibration])
                temperature = _select_temperature(labels[calibration], calibration_probs)
                query_probs = _temperature(final.predict_proba(shifted[0.0][query]), temperature)
                results.append(
                    {
                        "direction": direction,
                        "manifest": manifest_path.name,
                        "ablation": ablation,
                        "augmentation": augmentation_name,
                        "training_shifts": list(shifts),
                        "selected_C": c,
                        "selection_trace": selection,
                        "temperature": temperature,
                        "target_query_retrospective": _metrics(labels[query], query_probs),
                        "selection_used_target_query": False,
                    }
                )
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "source-only channel-shift augmentation v3",
        "evidence_status": "retrospective development",
        "protocol_sha256": protocol_hash,
        "elapsed_seconds": time.perf_counter() - started,
        "results": results,
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "shift_augmentation_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--protocol-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        bundle_path=args.bundle,
        manifests=args.manifests,
        protocol_path=args.protocol,
        protocol_hash_path=args.protocol_hash,
        output_dir=args.output_dir,
    )
    print(json.dumps({"payload_sha256": result["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
