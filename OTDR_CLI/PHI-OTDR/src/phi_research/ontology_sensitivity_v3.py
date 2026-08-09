"""Post-hoc quiet-background exclusion sensitivity for PHI-OTDR v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from .data_contract import CLASS_NAMES, canonical_json_hash
from .era_contract import verify_acquisition_manifest
from .spatial_experiment import _metrics, _model_grid, _select_temperature, _temperature


PRIMARY = ("registered_position", "temporal_difference_energy", "dynamics", "logistic")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _fit_fixed(
    features: np.ndarray,
    labels: np.ndarray,
    partitions: np.ndarray,
    excluded: np.ndarray,
    *,
    selected_params: dict[str, object],
) -> tuple[object, float, dict[str, object], np.ndarray, np.ndarray]:
    train = (partitions == "source_train") & ~excluded
    validation = (partitions == "source_validation") & ~excluded
    calibration = (partitions == "source_calibration") & ~excluded
    query = (partitions == "target_query") & ~excluded
    development = train | validation
    model = next(model for params, model in _model_grid("logistic") if params == selected_params)
    model.fit(features[development], labels[development])
    temperature = _select_temperature(
        labels[calibration], model.predict_proba(features[calibration])
    )
    validation_probs = _temperature(model.predict_proba(features[validation]), temperature)
    query_probs = _temperature(model.predict_proba(features[query]), temperature)
    return model, temperature, _metrics(labels[validation], validation_probs), query, query_probs


def run(
    *,
    session_view_path: Path,
    spatial_results_path: Path,
    manifests: Iterable[Path],
    ontology_csv_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    with np.load(session_view_path, allow_pickle=False) as source:
        features = source["features"].astype(np.float64)
        sessions = source["sessions"].astype(str)
        labels = source["labels"].astype(np.int64)
    if features.shape[0] != 441 or len(np.unique(sessions)) != 441:
        raise ValueError("Expected complete 441-session view")
    spatial = json.loads(spatial_results_path.read_text(encoding="utf-8"))
    stored_spatial_hash = str(spatial.pop("payload_sha256"))
    if stored_spatial_hash != canonical_json_hash(spatial):
        raise ValueError("Spatial results hash mismatch")
    spatial["payload_sha256"] = stored_spatial_hash

    with ontology_csv_path.open(newline="", encoding="utf-8") as handle:
        ontology_rows = list(csv.DictReader(handle))
    ambiguous_sessions = {
        row["session_id"]
        for row in ontology_rows
        if row["nearby_non_target_activity"].lower() == "true"
    }
    if len(ambiguous_sessions) != 9:
        raise ValueError("Expected nine nearby-activity background sessions")
    ambiguous_mask = np.isin(sessions, sorted(ambiguous_sessions))

    results = []
    prediction_rows = []
    manifest_hashes = {}
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        verification = verify_acquisition_manifest(manifest)
        direction = f"{manifest['direction']['source']}_to_{manifest['direction']['target']}"
        manifest_hashes[direction] = verification["manifest_sha256"]
        session_rows = {str(row["session_id"]): row for row in manifest["sessions"]}
        partitions = np.asarray([session_rows[session]["partition"] for session in sessions])
        primary = next(
            row
            for row in spatial["results"]
            if row["direction"] == direction
            and (row["view"], row["estimator"], row["ablation"], row["model"]) == PRIMARY
        )
        selected_params = dict(primary["selected_params"])

        _, control_temperature, control_validation, control_query, control_probs = _fit_fixed(
            features,
            labels,
            partitions,
            np.zeros(len(sessions), dtype=bool),
            selected_params=selected_params,
        )
        control_metrics = _metrics(labels[control_query], control_probs)
        if not np.isclose(
            control_metrics["macro_f1"], primary["target_query_retrospective"]["macro_f1"], atol=1e-12
        ):
            raise ValueError("Fixed sensitivity control does not reproduce primary spatial result")

        quiet_query = control_query & ~ambiguous_mask
        control_quiet_probs = control_probs[~ambiguous_mask[control_query]]
        quiet_control_metrics = _metrics(labels[quiet_query], control_quiet_probs)

        _, sensitivity_temperature, sensitivity_validation, sensitivity_query, sensitivity_probs = _fit_fixed(
            features,
            labels,
            partitions,
            ambiguous_mask,
            selected_params=selected_params,
        )
        sensitivity_metrics = _metrics(labels[sensitivity_query], sensitivity_probs)
        if not np.array_equal(sensitivity_query, quiet_query):
            raise ValueError("Sensitivity query must equal the quiet-only official query")

        affected = {
            partition: int(np.sum(ambiguous_mask & (partitions == partition)))
            for partition in (
                "source_train",
                "source_validation",
                "source_calibration",
                "target_support",
                "target_calibration",
                "target_query",
            )
        }
        results.append(
            {
                "direction": direction,
                "selected_params_frozen_from_primary": selected_params,
                "affected_ambiguous_sessions_by_partition": affected,
                "official_control_temperature": control_temperature,
                "sensitivity_temperature": sensitivity_temperature,
                "official_control_source_validation": control_validation,
                "sensitivity_source_validation": sensitivity_validation,
                "official_control_target_query": control_metrics,
                "official_model_quiet_query_only": quiet_control_metrics,
                "ontology_excluded_refit_quiet_query": sensitivity_metrics,
                "query_composition_delta_macro_f1": float(
                    quiet_control_metrics["macro_f1"] - control_metrics["macro_f1"]
                ),
                "source_refit_delta_on_same_quiet_query_macro_f1": float(
                    sensitivity_metrics["macro_f1"] - quiet_control_metrics["macro_f1"]
                ),
            }
        )

        control_by_session = {
            session: control_probs[index]
            for index, session in enumerate(sessions[control_query])
            if not ambiguous_mask[np.flatnonzero(control_query)[index]]
        }
        for index, session in enumerate(sessions[sensitivity_query]):
            control = control_by_session[session]
            sensitivity = sensitivity_probs[index]
            row = {
                "direction": direction,
                "session_id": session,
                "true_label": int(labels[sensitivity_query][index]),
                "true_class": CLASS_NAMES[int(labels[sensitivity_query][index])],
                "control_predicted_class": CLASS_NAMES[int(np.argmax(control))],
                "sensitivity_predicted_class": CLASS_NAMES[int(np.argmax(sensitivity))],
            }
            for class_id, name in enumerate(CLASS_NAMES):
                row[f"control_prob_{name}"] = float(control[class_id])
                row[f"sensitivity_prob_{name}"] = float(sensitivity[class_id])
            prediction_rows.append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "ontology_sensitivity_predictions.csv"
    _write_csv(prediction_path, prediction_rows)
    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 post-hoc quiet-background exclusion sensitivity",
        "evidence_status": "exploratory retrospective sensitivity; not a relabelled primary analysis",
        "input_hashes": {
            "session_view_sha256": _sha256(session_view_path),
            "spatial_results_payload_sha256": stored_spatial_hash,
            "ontology_csv_sha256": _sha256(ontology_csv_path),
            "manifest_sha256": manifest_hashes,
        },
        "ambiguous_background_sessions": sorted(ambiguous_sessions),
        "policy": {
            "labels_changed": False,
            "hyperparameters_reselected": False,
            "query_correctness_used_for_selection": False,
            "interpretation": "Exclusion tests whether the weak composite background definition materially changes the frozen primary model; it does not establish that excluded labels are wrong.",
        },
        "results": results,
        "prediction_sha256": _sha256(prediction_path),
        "limitations": [
            "The nine excluded sessions were identified post hoc from filenames.",
            "All nine occur on one April/May date, so ontology and acquisition effects cannot be separated.",
            "January-to-April/May changes target-query composition; April/May-to-January changes source fitting.",
        ],
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "ontology_sensitivity.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-view", type=Path, required=True)
    parser.add_argument("--spatial-results", type=Path, required=True)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--ontology-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(
        session_view_path=args.session_view,
        spatial_results_path=args.spatial_results,
        manifests=args.manifest,
        ontology_csv_path=args.ontology_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps({"payload_sha256": result["payload_sha256"], "results": len(result["results"])}, indent=2))


if __name__ == "__main__":
    main()
