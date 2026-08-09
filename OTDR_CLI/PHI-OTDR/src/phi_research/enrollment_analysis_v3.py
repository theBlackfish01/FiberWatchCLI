"""Cluster-aware synthesis of retrospective PHI-OTDR v3 enrollment results."""

from __future__ import annotations

import argparse
import csv
import itertools
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from .data_contract import CLASS_NAMES, canonical_json_hash


CLASSICAL_METHODS = (
    "class_prototype",
    "registered_distribution_hybrid",
    "sliced_wasserstein_session_gallery",
)
SIAMESE_METHOD = "cuda_supervised_siamese_session_embedding"
METHOD_COMPARISONS = (
    ("registered_distribution_hybrid", "class_prototype"),
    ("sliced_wasserstein_session_gallery", "class_prototype"),
    ("sliced_wasserstein_session_gallery", "registered_distribution_hybrid"),
    (SIAMESE_METHOD, "class_prototype"),
    (SIAMESE_METHOD, "registered_distribution_hybrid"),
    (SIAMESE_METHOD, "sliced_wasserstein_session_gallery"),
)
METRICS = ("enrollment_h", "session_macro_f1", "worst_class_recall")


def _direction_key(value: object) -> str:
    if not isinstance(value, dict) or set(value) != {"source", "target"}:
        raise ValueError(f"Invalid acquisition direction: {value!r}")
    key = f"{value['source']}_to_{value['target']}"
    if key not in {"january_to_april_may", "april_may_to_january"}:
        raise ValueError(f"Unsupported acquisition direction: {key}")
    return key


def _load_hashed_payload(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = str(payload.pop("payload_sha256"))
    calculated = canonical_json_hash(payload)
    if stored != calculated:
        raise ValueError(f"Payload hash mismatch for {path}: {stored} != {calculated}")
    payload["payload_sha256"] = stored
    return payload


def exact_sign_flip_p(class_effects: Iterable[float]) -> float:
    """Exact two-sided randomization p-value over independent class effects."""

    effects = np.asarray(list(class_effects), dtype=np.float64)
    if effects.ndim != 1 or effects.size == 0 or not np.all(np.isfinite(effects)):
        raise ValueError("class_effects must be a non-empty finite vector")
    observed = abs(float(np.mean(effects)))
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=effects.size):
        null.append(abs(float(np.mean(effects * np.asarray(signs)))))
    return float(np.mean(np.asarray(null) >= observed - 1e-15))


def bh_qvalues(p_values: Iterable[float]) -> list[float]:
    values = np.asarray(list(p_values), dtype=np.float64)
    if values.ndim != 1 or not np.all((values >= 0.0) & (values <= 1.0)):
        raise ValueError("p-values must lie in [0, 1]")
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0.0, 1.0)
    return [float(value) for value in output]


def _episode_key(row: dict[str, object]) -> tuple[str, str, int, int]:
    return (
        str(row["direction"]),
        str(row["heldout_class"]),
        int(row["shot"]),
        int(row["draw"]),
    )


def _method_rows(
    classical: dict[str, object], siamese: dict[str, object]
) -> dict[str, dict[tuple[str, str, int, int], dict[str, object]]]:
    methods: dict[str, dict[tuple[str, str, int, int], dict[str, object]]] = {
        method: {} for method in (*CLASSICAL_METHODS, SIAMESE_METHOD)
    }
    for raw in classical["episodes"]:
        row = dict(raw)
        if row["selector"] != "random":
            continue
        method = str(row["method"])
        key = _episode_key(row)
        if key in methods[method]:
            raise ValueError(f"Duplicate classical episode for {method} {key}")
        methods[method][key] = row

    grouped_siamese: dict[tuple[str, str, int, int], list[dict[str, object]]] = defaultdict(list)
    for raw in siamese["episodes"]:
        row = dict(raw)
        if row["selector"] == "random":
            grouped_siamese[_episode_key(row)].append(row)
    for key, rows in grouped_siamese.items():
        supports = {tuple(row["support_sessions"]) for row in rows}
        if len(rows) != 3 or len(supports) != 1:
            raise ValueError(f"Expected three Siamese seeds with identical support for {key}")
        aggregate = dict(rows[0])
        aggregate.pop("siamese_seed", None)
        aggregate["siamese_seeds"] = sorted(int(row["siamese_seed"]) for row in rows)
        for metric in METRICS + ("base_class_accuracy", "enrolled_class_recall", "balanced_accuracy"):
            aggregate[metric] = float(np.mean([float(row[metric]) for row in rows]))
        methods[SIAMESE_METHOD][key] = aggregate

    expected = 2 * len(CLASS_NAMES) * 3 * 30
    for method, rows in methods.items():
        if len(rows) != expected:
            raise ValueError(f"Expected {expected} random episodes for {method}, found {len(rows)}")
    return methods


def paired_cluster_comparison(
    left: dict[tuple[str, str, int, int], dict[str, object]],
    right: dict[tuple[str, str, int, int], dict[str, object]],
    *,
    direction: str,
    shot: int,
    metric: str,
    seed: int,
    bootstrap_draws: int = 20_000,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Pair support-identical draws, then aggregate to held-out-class clusters."""

    class_rows: list[dict[str, object]] = []
    all_draw_effects: list[float] = []
    for heldout in CLASS_NAMES:
        draw_effects = []
        for draw in range(30):
            key = (direction, heldout, shot, draw)
            left_row = left[key]
            right_row = right[key]
            if tuple(left_row["support_sessions"]) != tuple(right_row["support_sessions"]):
                raise ValueError(f"Unpaired support sessions for {key}")
            draw_effects.append(float(left_row[metric]) - float(right_row[metric]))
        class_rows.append(
            {
                "direction": direction,
                "shot": shot,
                "metric": metric,
                "heldout_class": heldout,
                "mean_effect": float(np.mean(draw_effects)),
                "median_effect": float(np.median(draw_effects)),
                "draw_win_fraction": float(np.mean(np.asarray(draw_effects) > 0.0)),
                "draw_tie_fraction": float(np.mean(np.asarray(draw_effects) == 0.0)),
                "draw_count": len(draw_effects),
            }
        )
        all_draw_effects.extend(draw_effects)

    class_effects = np.asarray([float(row["mean_effect"]) for row in class_rows])
    rng = np.random.default_rng(seed)
    bootstrap = np.mean(
        rng.choice(class_effects, size=(bootstrap_draws, class_effects.size), replace=True), axis=1
    )
    sd = float(np.std(class_effects, ddof=1))
    result = {
        "direction": direction,
        "shot": shot,
        "metric": metric,
        "mean_effect": float(np.mean(class_effects)),
        "median_class_effect": float(np.median(class_effects)),
        "ci95_low": float(np.quantile(bootstrap, 0.025)),
        "ci95_high": float(np.quantile(bootstrap, 0.975)),
        "exact_sign_flip_two_sided_p": exact_sign_flip_p(class_effects),
        "class_win_fraction": float(np.mean(class_effects > 0.0)),
        "class_tie_fraction": float(np.mean(class_effects == 0.0)),
        "draw_win_fraction": float(np.mean(np.asarray(all_draw_effects) > 0.0)),
        "standardized_class_effect": float(np.mean(class_effects) / sd) if sd > 0 else None,
        "heldout_class_clusters": int(class_effects.size),
        "paired_random_draws": len(all_draw_effects),
        "bootstrap_draws": bootstrap_draws,
    }
    return result, class_rows


def _selector_summary(payloads: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, object]]] = defaultdict(list)
    for payload in payloads:
        for raw in payload["episodes"]:
            row = dict(raw)
            grouped[
                (str(row["direction"]), str(row["method"]), int(row["shot"]), str(row["selector"]))
            ].append(row)
    output = []
    for (direction, method, shot, selector), rows in sorted(grouped.items()):
        # Siamese rows contain three seeds. Equal weighting across class-seed episodes is intentional here.
        output.append(
            {
                "direction": direction,
                "method": method,
                "shot": shot,
                "selector": selector,
                "episode_count": len(rows),
                "enrollment_h_mean": float(np.mean([float(row["enrollment_h"]) for row in rows])),
                "macro_f1_mean": float(np.mean([float(row["session_macro_f1"]) for row in rows])),
                "worst_class_recall_mean": float(
                    np.mean([float(row["worst_class_recall"]) for row in rows])
                ),
                "enrollment_h_min": float(np.min([float(row["enrollment_h"]) for row in rows])),
            }
        )
    return output


def _candidate_sessions(manifest: dict[str, object]) -> dict[str, set[str]]:
    output = {name: set() for name in CLASS_NAMES}
    for row in manifest["sessions"]:
        if row["partition"] == "target_support":
            output[str(row["class_name"])].add(str(row["session_id"]))
    if any(len(values) != 7 for values in output.values()):
        raise ValueError("Every target-support class must have exactly seven sessions")
    return output


def _coverage_rows(
    payloads: Iterable[dict[str, object]], manifests: dict[str, dict[str, object]]
) -> list[dict[str, object]]:
    candidates = {direction: _candidate_sessions(manifest) for direction, manifest in manifests.items()}
    grouped: dict[tuple[str, str, str, int, str], list[tuple[str, ...]]] = defaultdict(list)
    for payload in payloads:
        for raw in payload["episodes"]:
            row = dict(raw)
            grouped[
                (
                    str(row["direction"]),
                    str(row["heldout_class"]),
                    str(row["method"]),
                    int(row["shot"]),
                    str(row["selector"]),
                )
            ].append(tuple(str(value) for value in row["support_sessions"]))

    output = []
    for (direction, heldout, method, shot, selector), supports in sorted(grouped.items()):
        frequency = Counter(session for support in supports for session in support)
        pool = candidates[direction][heldout]
        if not set(frequency).issubset(pool):
            raise ValueError(f"Support outside candidate pool for {direction} {heldout} {method}")
        counts = np.asarray([frequency.get(session, 0) for session in sorted(pool)], dtype=np.float64)
        probabilities = counts / max(float(np.sum(counts)), 1.0)
        nonzero = probabilities[probabilities > 0]
        entropy = -float(np.sum(nonzero * np.log(nonzero))) / np.log(len(pool))
        output.append(
            {
                "direction": direction,
                "heldout_class": heldout,
                "method": method,
                "shot": shot,
                "selector": selector,
                "episodes": len(supports),
                "candidate_sessions": len(pool),
                "unique_sessions_selected": len(frequency),
                "candidate_pool_coverage": len(frequency) / len(pool),
                "selection_frequency_min": int(np.min(counts)),
                "selection_frequency_max": int(np.max(counts)),
                "normalized_selection_entropy": entropy,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze(
    classical_path: Path,
    siamese_path: Path,
    manifest_paths: Iterable[Path],
    output_dir: Path,
) -> dict[str, object]:
    classical = _load_hashed_payload(classical_path)
    siamese = _load_hashed_payload(siamese_path)
    manifests = {}
    for path in manifest_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        stored = str(payload.pop("manifest_sha256"))
        if stored != canonical_json_hash(payload):
            raise ValueError(f"Manifest hash mismatch for {path}")
        payload["manifest_sha256"] = stored
        manifests[_direction_key(payload["direction"])] = payload
    if set(manifests) != {"january_to_april_may", "april_may_to_january"}:
        raise ValueError("Both acquisition directions are required")

    methods = _method_rows(classical, siamese)
    comparisons = []
    class_effect_rows = []
    sequence = 0
    for metric in METRICS:
        for direction in sorted(manifests):
            for shot in (1, 3, 5):
                for left_name, right_name in METHOD_COMPARISONS:
                    comparison, class_rows = paired_cluster_comparison(
                        methods[left_name],
                        methods[right_name],
                        direction=direction,
                        shot=shot,
                        metric=metric,
                        seed=20260808 + sequence,
                    )
                    sequence += 1
                    comparison["left_method"] = left_name
                    comparison["right_method"] = right_name
                    comparison["comparison"] = f"{left_name}_minus_{right_name}"
                    for row in class_rows:
                        row["left_method"] = left_name
                        row["right_method"] = right_name
                        row["comparison"] = comparison["comparison"]
                    comparisons.append(comparison)
                    class_effect_rows.extend(class_rows)

    for metric in METRICS:
        positions = [index for index, row in enumerate(comparisons) if row["metric"] == metric]
        qvalues = bh_qvalues(
            [float(comparisons[index]["exact_sign_flip_two_sided_p"]) for index in positions]
        )
        for index, qvalue in zip(positions, qvalues, strict=True):
            comparisons[index]["bh_qvalue_within_metric_family"] = qvalue

    selector_summary = _selector_summary((classical, siamese))
    coverage = _coverage_rows((classical, siamese), manifests)
    _write_csv(output_dir / "paired_comparisons.csv", comparisons)
    _write_csv(output_dir / "heldout_class_effects.csv", class_effect_rows)
    _write_csv(output_dir / "selector_summary.csv", selector_summary)
    _write_csv(output_dir / "support_pool_coverage.csv", coverage)

    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 paired enrollment synthesis",
        "evidence_status": "retrospective development; not independent confirmation",
        "statistical_unit": "held-out event class after pairing identical target-support draws",
        "independent_clusters": len(CLASS_NAMES),
        "pairing": "same direction, held-out class, shot, draw, and exact support-session set",
        "siamese_seed_policy": "average the three frozen Siamese seeds within an identical support episode",
        "bootstrap": "20,000 resamples of the six held-out-class mean effects",
        "test": "exact two-sided sign-flip randomization over six held-out-class mean effects",
        "multiplicity": "Benjamini-Hochberg within each metric family across 36 comparisons",
        "inputs": {
            "classical_payload_sha256": classical["payload_sha256"],
            "siamese_payload_sha256": siamese["payload_sha256"],
            "manifest_sha256": {
                direction: manifest["manifest_sha256"] for direction, manifest in manifests.items()
            },
        },
        "comparisons": comparisons,
        "selector_summary": selector_summary,
        "support_pool_coverage": coverage,
        "limitations": [
            "Only six held-out-class clusters are available, so exact p-values are coarse.",
            "Support-selection geometry is method-specific for deterministic selectors; only random draws are paired across methods.",
            "Candidate-pool coverage measures session coverage, not coverage of unknown real-world morphologies.",
            "Target-query outcomes are historical retrospective evidence.",
        ],
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "enrollment_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classical", type=Path, required=True)
    parser.add_argument("--siamese", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.classical, args.siamese, args.manifest, args.output_dir)
    print(
        json.dumps(
            {
                "comparisons": len(result["comparisons"]),
                "payload_sha256": result["payload_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
