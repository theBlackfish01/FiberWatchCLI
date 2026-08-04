from __future__ import annotations

"""Strict completion audit for the confirmatory OTDR TabPFN study."""

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Any

import torch

from .lifecycle_experiment import _git_metadata
from .study_state import atomic_json, file_sha256, utc_now, validate_run
from .tabpfn_full_study import (
    DATA_PATH,
    FROZEN_PROTOCOL_SHA256,
    METHODS,
    REPOSITORY_ROOT,
    PROTOCOL_PATH,
    STUDY_ROOT,
    load_protocol,
)


REQUIRED_FILES = (
    "failures.jsonl",
    "experiment_registry.jsonl",
    "state.json",
    "TEST_VALIDATION.json",
    "PLOT_INSPECTION.json",
    "SOURCE_SNAPSHOT.json",
    "ANALYSIS_SOURCE_SNAPSHOT.json",
)
REQUIRED_DIRECTORIES = (
    "configs",
    "pilots",
    "full_benchmark",
    "summary_only",
    "incremental_memory_pilot",
    "tables",
    "plots",
)

SOURCE_FILES = (
    "OTDR_CLI/OTDR/src/tabpfn_full_study.py",
    "OTDR_CLI/OTDR/src/tabpfn_full_analysis.py",
    "OTDR_CLI/OTDR/src/tabpfn_full_completion.py",
    "OTDR_CLI/OTDR/src/tabpfn_incremental_memory.py",
    "OTDR_CLI/OTDR/src/lifecycle_tabpfn.py",
    "OTDR_CLI/OTDR/src/lifecycle_data.py",
    "OTDR_CLI/OTDR/src/lifecycle_baselines.py",
    "OTDR_CLI/OTDR/src/lifecycle_enrollment.py",
    "OTDR_CLI/OTDR/src/lifecycle_metrics.py",
    "OTDR_CLI/OTDR/src/lifecycle_training.py",
    "OTDR_CLI/OTDR/src/study_state.py",
    "OTDR_CLI/OTDR/src/event_openworld_data.py",
    "OTDR_CLI/OTDR/src/model_functions/lifecycle.py",
    "OTDR_CLI/OTDR/tests/test_lifecycle.py",
    "OTDR_CLI/OTDR/tests/test_tabpfn_incremental_memory.py",
)


def _units(stage: str) -> list[Path]:
    return sorted((STUDY_ROOT / stage).glob("pair_*/seed_*"))


def write_source_snapshot() -> dict[str, Any]:
    from tabpfn.model_loading import resolve_model_path

    checkpoint, _, model_name, _ = resolve_model_path(
        None, "classifier", "v2"
    )
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot source missing: {path}")
        source_hashes[relative] = file_sha256(path)
    legacy_source_records = []
    for metrics_path in sorted(
        (STUDY_ROOT / "full_benchmark").glob("pair_*/seed_*/metrics.json")
    ):
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        if metrics.get("source") and metrics["source"] not in legacy_source_records:
            legacy_source_records.append(metrics["source"])
    snapshot = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "timing": (
            "Named snapshot captured before the schema-v3 full-matrix rerun. "
            "The initial schema-v1 units already persisted per-unit Git revision, "
            "dirty-patch hash, and dirty-source-content hash before each outer run. "
            "Schema-v2 was a bounded evidence-format validation that exposed "
            "float32 probability tie and ECE-boundary ambiguity; schema-v3 stores "
            "probabilities losslessly as float64 for exact reconstruction."
        ),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "dataset_path": str(DATA_PATH),
        "dataset_sha256": file_sha256(DATA_PATH),
        "tabpfn_package_version": importlib.metadata.version("tabpfn"),
        "tabpfn_model_name": model_name,
        "tabpfn_checkpoint_path": str(checkpoint),
        "tabpfn_checkpoint_sha256": file_sha256(checkpoint),
        "tabpfn_checkpoint_size_bytes": checkpoint.stat().st_size,
        "python": {
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
        },
        "git": _git_metadata(REPOSITORY_ROOT),
        "source_file_sha256": source_hashes,
        "initial_per_unit_source_records": legacy_source_records,
        "frozen_source_artifacts": {
            "lifecycle_finalists_sha256": file_sha256(
                REPOSITORY_ROOT
                / "OTDR_CLI/OTDR/experiments/otdr_feature_assisted_lifecycle_study/configs/finalists.json"
            ),
            "representative_tabpfn_metrics_sha256": file_sha256(
                REPOSITORY_ROOT
                / "OTDR_CLI/OTDR/experiments/otdr_feature_assisted_lifecycle_study/baselines/tabpfn_v2/metrics.json"
            ),
        },
        "protocol_deviation": (
            "The named SOURCE_SNAPSHOT.json was not written before the first "
            "schema-v1 outer unit. This is a documentation-timing deviation, "
            "not a policy change: configs/protocol.json was hash-frozen first, "
            "and every schema-v1 unit persisted source revision/dirty hashes. "
            "The final schema-v3 rerun begins only after this updated named snapshot."
        ),
    }
    atomic_json(STUDY_ROOT / "SOURCE_SNAPSHOT.json", snapshot)
    return snapshot


def write_analysis_source_snapshot() -> dict[str, Any]:
    """Capture the exact post-hoc/analysis source without rewriting the run snapshot."""
    from tabpfn.model_loading import resolve_model_path

    checkpoint, _, model_name, _ = resolve_model_path(None, "classifier", "v2")
    source_hashes = {}
    for relative in SOURCE_FILES:
        path = REPOSITORY_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Analysis source missing: {path}")
        source_hashes[relative] = file_sha256(path)
    snapshot = {
        "schema_version": 1,
        "captured_at": utc_now(),
        "purpose": (
            "Exact source snapshot for final reconstruction, reporting, and the "
            "post-confirmatory fixed-base-memory sensitivity. SOURCE_SNAPSHOT.json "
            "remains the immutable execution-era snapshot."
        ),
        "protocol_sha256": file_sha256(PROTOCOL_PATH),
        "dataset_sha256": file_sha256(DATA_PATH),
        "pilot_config_sha256": file_sha256(
            STUDY_ROOT / "incremental_memory_pilot" / "config.json"
        ),
        "tabpfn_package_version": importlib.metadata.version("tabpfn"),
        "tabpfn_model_name": model_name,
        "tabpfn_checkpoint_path": str(checkpoint),
        "tabpfn_checkpoint_sha256": file_sha256(checkpoint),
        "source_file_sha256": source_hashes,
        "git": _git_metadata(REPOSITORY_ROOT),
    }
    atomic_json(STUDY_ROOT / "ANALYSIS_SOURCE_SNAPSHOT.json", snapshot)
    return snapshot


def _unit_audit(
    *,
    stage: str,
    expected_units: int,
    expected_regime: str,
) -> dict[str, Any]:
    method_rows = {method: 0 for method in METHODS}
    invalid = []
    cuda_failures = []
    shape_failures = []
    for unit in _units(stage):
        valid, reason = validate_run(
            unit,
            expected={
                "protocol_sha256": FROZEN_PROTOCOL_SHA256,
                "evidence_schema": 3,
            },
        )
        if not valid:
            invalid.append({"unit": str(unit), "reason": reason})
            continue
        metrics = json.loads((unit / "metrics.json").read_text(encoding="utf-8"))
        cuda = metrics.get("cuda_diagnostics") or {}
        if (
            metrics.get("device") != "cuda:0"
            or metrics.get("environment", {}).get("device") != "cuda:0"
            or metrics.get("regime") != expected_regime
            or cuda.get("actual_device") != "cuda:0"
            or int(cuda.get("peak_allocated_bytes", 0)) <= 0
            or not cuda.get("compute_capability")
        ):
            cuda_failures.append(metrics.get("run_id"))
        expected_context_rows = (
            int(metrics["requested_draws"])
            * len(metrics["requested_shots"])
            * 3
        )
        if (
            not metrics.get("config_hash")
            or len(metrics.get("context_sensitivity_rows", []))
            != expected_context_rows
            or not (unit / "prediction_evidence.npz").is_file()
        ):
            shape_failures.append(
                {
                    "run_id": metrics["run_id"],
                    "failure": "schema_v3_evidence_or_context_rows",
                    "expected_context_rows": expected_context_rows,
                    "actual_context_rows": len(
                        metrics.get("context_sensitivity_rows", [])
                    ),
                }
            )
        unit_counts: dict[str, int] = {}
        for row in metrics["rows"]:
            method = row["method"]
            method_rows[method] = method_rows.get(method, 0) + 1
            unit_counts[method] = unit_counts.get(method, 0) + 1
        expected_per_present_method = (
            len(metrics["requested_shots"]) * int(metrics["requested_draws"])
        )
        if any(
            count != expected_per_present_method
            for count in unit_counts.values()
        ):
            shape_failures.append(
                {
                    "run_id": metrics["run_id"],
                    "counts": unit_counts,
                    "expected": expected_per_present_method,
                }
            )
    units = _units(stage)
    return {
        "stage": stage,
        "expected_units": expected_units,
        "discovered_units": len(units),
        "invalid_units": invalid,
        "cuda_or_regime_failures": cuda_failures,
        "row_shape_failures": shape_failures,
        "method_rows": method_rows,
        "passed": (
            len(units) == expected_units
            and not invalid
            and not cuda_failures
            and not shape_failures
        ),
    }


def completion_audit() -> dict[str, Any]:
    protocol = load_protocol()
    checks: dict[str, Any] = {}
    checks["protocol"] = {
        "expected_sha256": FROZEN_PROTOCOL_SHA256,
        "actual_sha256": file_sha256(PROTOCOL_PATH),
        "pairs": len(protocol["pairs"]),
        "seeds": len(protocol["seeds"]),
        "passed": (
            file_sha256(PROTOCOL_PATH) == FROZEN_PROTOCOL_SHA256
            and len(protocol["pairs"]) == 21
            and len(protocol["seeds"]) == 5
        ),
    }
    checks["full_benchmark"] = _unit_audit(
        stage="full_benchmark", expected_units=105, expected_regime="full"
    )
    checks["summary_only"] = _unit_audit(
        stage="summary_only",
        expected_units=105,
        expected_regime="summary_only",
    )
    full_rows = checks["full_benchmark"]["method_rows"]
    checks["primary_row_matrix"] = {
        "expected_per_method": 6300,
        "actual": full_rows,
        "passed": all(full_rows.get(method) == 6300 for method in METHODS),
    }
    summary_rows = checks["summary_only"]["method_rows"]
    summary_common = (
        "tabpfn_v2",
        "raw_cosine_1nn",
        "raw_euclidean_1nn",
        "raw_mahalanobis_1nn",
        "logistic_regression",
        "linear_svm",
        "shrinkage_lda",
    )
    summary_cfe = (
        "cfe_finalist",
        "cfe_uncalibrated_mean",
        "encoder_cosine_1nn",
    )
    checks["summary_row_matrix"] = {
        "expected_common_per_method": 6300,
        "expected_cfe_per_method": 1260,
        "actual": summary_rows,
        "passed": (
            all(summary_rows.get(method) == 6300 for method in summary_common)
            and all(summary_rows.get(method) == 1260 for method in summary_cfe)
        ),
    }
    missing = [
        name for name in REQUIRED_FILES if not (STUDY_ROOT / name).is_file()
    ]
    missing_directories = [
        name
        for name in REQUIRED_DIRECTORIES
        if not (STUDY_ROOT / name).is_dir()
    ]
    checks["required_artifacts"] = {
        "required": list(REQUIRED_FILES),
        "missing": missing,
        "required_directories": list(REQUIRED_DIRECTORIES),
        "missing_directories": missing_directories,
        "passed": not missing and not missing_directories,
    }
    reconstruction_checks = {}
    for regime in ("full", "summary_only"):
        path = STUDY_ROOT / "tables" / regime / "metric_reconstruction.json"
        reconstruction_checks[regime] = (
            path.is_file()
            and json.loads(path.read_text(encoding="utf-8")).get("passed") is True
        )
    checks["metric_reconstruction"] = {
        **reconstruction_checks,
        "passed": all(reconstruction_checks.values()),
    }
    analysis_manifests = {}
    for regime in ("full", "summary_only"):
        path = STUDY_ROOT / "tables" / regime / "analysis_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
        analysis_manifests[regime] = {
            "exists": path.is_file(),
            "units": payload.get("valid_units"),
            "rows": payload.get("metric_reconstruction", {}).get("rows_reconstructed"),
            "passed": (
                path.is_file()
                and payload.get("valid_units") == 105
                and payload.get("metric_reconstruction", {}).get("passed") is True
                and payload.get("group_manifest_audit", {}).get("passed") is True
            ),
        }
    feature_path = STUDY_ROOT / "tables" / "feature_regime_comparison_manifest.json"
    feature_payload = (
        json.loads(feature_path.read_text(encoding="utf-8"))
        if feature_path.is_file()
        else {}
    )
    checks["analysis_manifests"] = {
        **analysis_manifests,
        "feature_regime_comparison": feature_payload,
        "passed": (
            all(value["passed"] for value in analysis_manifests.values())
            and feature_path.is_file()
            and (STUDY_ROOT / "tables" / "feature_regime_comparison.csv").is_file()
            and feature_payload.get("independence_unit")
            == "pair/seed after support-draw averaging"
        ),
    }
    pilot_path = STUDY_ROOT / "incremental_memory_pilot" / "analysis_manifest.json"
    pilot_payload = (
        json.loads(pilot_path.read_text(encoding="utf-8"))
        if pilot_path.is_file()
        else {}
    )
    pilot_audit = pilot_payload.get("group_and_comparator_audit", {})
    checks["incremental_memory_pilot"] = {
        "payload": pilot_payload,
        "passed": (
            pilot_path.is_file()
            and pilot_payload.get("units") == 21
            and pilot_payload.get("rows") == 2520
            and pilot_payload.get("reconstruction", {}).get("passed") is True
            and pilot_audit.get("passed") is True
            and pilot_audit.get("cuda_units") == 21
            and pilot_audit.get("matched_frozen_query_and_support_units") == 21
            and pilot_audit.get("pre_enrollment_reconstructed_units") == 21
            and (STUDY_ROOT / "incremental_memory_pilot" / "REPORT.md").is_file()
        ),
    }
    validation_path = STUDY_ROOT / "TEST_VALIDATION.json"
    plot_path = STUDY_ROOT / "PLOT_INSPECTION.json"
    test_validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else {}
    )
    plot_inspection = (
        json.loads(plot_path.read_text(encoding="utf-8"))
        if plot_path.is_file()
        else {}
    )
    checks["tests"] = {
        "payload": test_validation,
        "passed": test_validation.get("passed") is True,
    }
    checks["plots"] = {
        "payload": plot_inspection,
        "passed": plot_inspection.get("passed") is True,
    }
    state = json.loads((STUDY_ROOT / "state.json").read_text(encoding="utf-8"))
    checks["state"] = {
        "status": state.get("status"),
        "active_failed_units": state.get("failed_units"),
        "passed": state.get("status") == "complete" and not state.get("failed_units"),
    }
    result = {
        "schema_version": 1,
        "checks": checks,
        "passed": all(check["passed"] for check in checks.values()),
        "interpretation": (
            "Completion is proven only when every named check is true; historical "
            "technical-smoke failures may remain in failures.jsonl if explicitly "
            "resolved and no active unit is failed."
        ),
    }
    atomic_json(STUDY_ROOT / "COMPLETION_AUDIT.json", result)
    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-source-snapshot", action="store_true")
    parser.add_argument("--write-analysis-source-snapshot", action="store_true")
    args = parser.parse_args(argv)
    if args.write_source_snapshot and args.write_analysis_source_snapshot:
        parser.error("Choose only one snapshot mode.")
    if args.write_source_snapshot:
        result = write_source_snapshot()
    elif args.write_analysis_source_snapshot:
        result = write_analysis_source_snapshot()
    else:
        result = completion_audit()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
