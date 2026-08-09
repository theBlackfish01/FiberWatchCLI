"""Evaluate frozen CUDA Siamese session embeddings in the v3 enrollment protocol."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

from .data_contract import CLASS_NAMES, canonical_json_hash
from .distributional_enrollment_v3 import (
    _class_distances,
    _classification_metrics,
    _detection_metrics,
    _seed,
    select_support,
)


METHOD = "cuda_supervised_siamese_session_embedding"


def run(
    *,
    siamese_summary_path: Path,
    manifests: Sequence[Path],
    config_path: Path,
    config_hash_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    summary = json.loads(siamese_summary_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = config_hash_path.read_text(encoding="utf-8").split()[0]
    if canonical_json_hash(config) != expected_hash or summary["config_sha256"] != expected_hash:
        raise ValueError("Siamese enrollment/config hash mismatch")
    manifest_by_direction = {}
    for path in manifests:
        payload = json.loads(path.read_text(encoding="utf-8"))
        direction = f"{payload['direction']['source']}_to_{payload['direction']['target']}"
        manifest_by_direction[direction] = payload
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = []
    detections = []
    support_rows = []
    prediction_rows = []
    started = time.perf_counter()
    for trained in summary["runs"]:
        direction = trained["direction"]
        manifest = manifest_by_direction[direction]
        holdout = int(trained["heldout_class_id"])
        seed = int(trained["seed"])
        with np.load(trained["embedding_path"], allow_pickle=False) as source:
            descriptor = source["embeddings"].astype(np.float32)
            sessions = source["sessions"].astype(str)
            labels = source["labels"].astype(np.int64)
        session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
        partitions = np.asarray([session_rows[session]["partition"] for session in sessions])
        base_mask = (partitions == "source_train") & (labels != holdout)
        calibration_mask = (partitions == "target_calibration") & (labels != holdout)
        query_mask = partitions == "target_query"
        support_mask = (partitions == "target_support") & (labels == holdout)
        base_gallery = descriptor[base_mask]
        base_labels = labels[base_mask]
        candidates = descriptor[support_mask]
        candidate_sessions = sessions[support_mask]
        query_sessions = sessions[query_mask]
        calibration_scores = _class_distances(
            descriptor[calibration_mask],
            base_gallery,
            base_labels,
            None,
            holdout,
            method=METHOD,
            neighbors=int(config["distribution"]["session_gallery_neighbors"]),
        )
        query_scores_pre = _class_distances(
            descriptor[query_mask],
            base_gallery,
            base_labels,
            None,
            holdout,
            method=METHOD,
            neighbors=int(config["distribution"]["session_gallery_neighbors"]),
        )
        seen_columns = [class_id for class_id in range(len(CLASS_NAMES)) if class_id != holdout]
        detection = _detection_metrics(
            np.min(calibration_scores[:, seen_columns], axis=1),
            np.min(query_scores_pre[:, seen_columns], axis=1),
            labels[query_mask],
            holdout,
            float(config["detection"]["known_acceptance_quantile"]),
        )
        detections.append(
            {
                "direction": direction,
                "heldout_class": CLASS_NAMES[holdout],
                "method": METHOD,
                "seed": seed,
                **detection,
            }
        )
        for shot in config["support"]["shots"]:
            selector_draws = [("random", draw) for draw in range(int(config["support"]["random_draws"]))]
            selector_draws.extend((selector, 0) for selector in ("medoid", "k_center", "pool_coverage"))
            for selector, draw in selector_draws:
                support_seed = _seed(20260808, direction, holdout, shot, selector, draw)
                selected_local = select_support(
                    candidates,
                    selector=selector,
                    shot=int(shot),
                    seed=support_seed,
                )
                selected_support = candidates[selected_local]
                selected_sessions = candidate_sessions[selected_local]
                if np.any(np.isin(selected_sessions, query_sessions)):
                    raise AssertionError("Siamese support/query session overlap")
                for rank, session in enumerate(selected_sessions):
                    support_rows.append(
                        {
                            "direction": direction,
                            "heldout_class": CLASS_NAMES[holdout],
                            "method": METHOD,
                            "siamese_seed": seed,
                            "shot": int(shot),
                            "selector": selector,
                            "draw": draw,
                            "support_seed": support_seed,
                            "rank": rank,
                            "session_id": session,
                        }
                    )
                scores = _class_distances(
                    descriptor[query_mask],
                    base_gallery,
                    base_labels,
                    selected_support,
                    holdout,
                    method=METHOD,
                    neighbors=int(config["distribution"]["session_gallery_neighbors"]),
                )
                prediction = np.argmin(scores, axis=1)
                metrics = _classification_metrics(labels[query_mask], prediction, holdout)
                episode_id = canonical_json_hash(
                    {
                        "direction": direction,
                        "holdout": holdout,
                        "method": METHOD,
                        "siamese_seed": seed,
                        "shot": shot,
                        "selector": selector,
                        "draw": draw,
                        "support": selected_sessions.tolist(),
                    }
                )
                episodes.append(
                    {
                        "episode_id": episode_id,
                        "direction": direction,
                        "heldout_class": CLASS_NAMES[holdout],
                        "method": METHOD,
                        "siamese_seed": seed,
                        "shot": int(shot),
                        "selector": selector,
                        "draw": draw,
                        "support_sessions": selected_sessions.tolist(),
                        **metrics,
                    }
                )
                for index, session in enumerate(query_sessions):
                    prediction_rows.append(
                        {
                            "episode_id": episode_id,
                            "session_id": session,
                            "true_label": int(labels[query_mask][index]),
                            "predicted_label": int(prediction[index]),
                            "true_class": CLASS_NAMES[int(labels[query_mask][index])],
                            "predicted_class": CLASS_NAMES[int(prediction[index])],
                            "minimum_distance": float(np.min(scores[index])),
                        }
                    )
        print(
            f"[SIAMESE ENROLLMENT] {direction} holdout={CLASS_NAMES[holdout]} seed={seed}",
            flush=True,
        )
    groups: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for row in episodes:
        if row["selector"] == "random":
            groups[(row["direction"], row["shot"])].append(row)
    random_summary = []
    for (direction, shot), rows in sorted(groups.items()):
        h = np.asarray([row["enrollment_h"] for row in rows])
        f1 = np.asarray([row["session_macro_f1"] for row in rows])
        random_summary.append(
            {
                "direction": direction,
                "method": METHOD,
                "shot": shot,
                "heldout_classes": len(set(row["heldout_class"] for row in rows)),
                "siamese_seeds": len(set(row["siamese_seed"] for row in rows)),
                "support_draw_episodes": len(rows),
                "enrollment_h_mean": float(np.mean(h)),
                "enrollment_h_std": float(np.std(h, ddof=1)),
                "enrollment_h_worst": float(np.min(h)),
                "macro_f1_mean": float(np.mean(f1)),
                "macro_f1_worst": float(np.min(f1)),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "CUDA Siamese session enrollment v3",
        "evidence_status": "retrospective development",
        "config_sha256": expected_hash,
        "siamese_summary_sha256": summary["payload_sha256"],
        "episode_count": len(episodes),
        "detection_fold_seed_count": len(detections),
        "elapsed_seconds": time.perf_counter() - started,
        "pre_enrollment_detection": detections,
        "random_draw_summary": random_summary,
        "episodes": episodes,
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "siamese_enrollment_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    for filename, rows in (("support_draws.csv", support_rows), ("query_predictions.csv.gz", prediction_rows)):
        opener = gzip.open if filename.endswith(".gz") else open
        mode = "wt" if filename.endswith(".gz") else "w"
        with opener(output_dir / filename, mode, newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--siamese-summary", type=Path, required=True)
    parser.add_argument("--manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--config-hash", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        siamese_summary_path=args.siamese_summary,
        manifests=args.manifests,
        config_path=args.config,
        config_hash_path=args.config_hash,
        output_dir=args.output_dir,
    )
    print(json.dumps({"episode_count": result["episode_count"], "payload_sha256": result["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
