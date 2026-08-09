"""Requirement-level completion audit for the PHI-OTDR v3 research program."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import IO, Any

from .data_contract import canonical_json_hash


EXPECTED_FINGERPRINT = "df38c1a11ac10481cc376eb791d1bdc699b967041a0f8b89c9f46b11027ddfe1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _integrity(path: Path, field: str) -> tuple[bool, str, str]:
    payload = _load(path)
    stored = str(payload.pop(field, ""))
    calculated = canonical_json_hash(payload)
    return stored == calculated, stored, calculated


def _sidecar_integrity(json_path: Path, sidecar_path: Path) -> dict[str, Any]:
    expected = sidecar_path.read_text(encoding="utf-8").split()[0]
    text = json_path.read_text(encoding="utf-8")
    lf_payload = json.loads(text.replace("\r\n", "\n"))
    crlf_text = text.replace("\r\n", "\n").replace("\n", "\r\n")
    crlf_payload = json.loads(crlf_text)
    lf_hash = canonical_json_hash(lf_payload)
    crlf_hash = canonical_json_hash(crlf_payload)
    return {
        "json": json_path.name,
        "sidecar": sidecar_path.name,
        "expected": expected,
        "lf_hash": lf_hash,
        "simulated_crlf_hash": crlf_hash,
        "status": "pass" if expected == lf_hash == crlf_hash else "fail",
    }


def _open_csv(path: Path) -> IO[str]:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", newline="")
    return path.open("r", encoding="utf-8", newline="")


def _session_membership(path: Path, allowed: dict[str, set[str]]) -> dict[str, Any]:
    rows = 0
    invalid: list[dict[str, str]] = []
    allowed_union = set().union(*allowed.values())
    with _open_csv(path) as handle:
        for row in csv.DictReader(handle):
            rows += 1
            direction = row.get("direction", "encoded_by_episode_id")
            session_id = row["session_id"]
            valid = (
                session_id in allowed_union
                if direction == "encoded_by_episode_id"
                else direction in allowed and session_id in allowed[direction]
            )
            if not valid:
                if len(invalid) < 20:
                    invalid.append({"direction": direction, "session_id": session_id})
    return {
        "path": path.as_posix(),
        "rows": rows,
        "invalid_examples": invalid,
        "status": "pass" if rows > 0 and not invalid else "fail",
    }


def _run(command: list[str], cwd: Path) -> dict[str, Any]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
        "status": "pass" if completed.returncode == 0 else "fail",
    }


def _checkpoints(phi_root: Path, summaries: list[Path]) -> dict[str, Any]:
    rows = []
    for summary_path in summaries:
        for run in _load(summary_path)["runs"]:
            checkpoint = phi_root / run["checkpoint_path"]
            observed = _sha256(checkpoint) if checkpoint.is_file() else None
            rows.append(
                {
                    "run_name": run["run_name"],
                    "exists": checkpoint.is_file(),
                    "expected_sha256": run["checkpoint_sha256"],
                    "observed_sha256": observed,
                    "status": "pass"
                    if checkpoint.is_file() and observed == run["checkpoint_sha256"]
                    else "fail",
                }
            )
    return {
        "count": len(rows),
        "failed": [row for row in rows if row["status"] != "pass"],
        "status": "pass" if rows and all(row["status"] == "pass" for row in rows) else "fail",
    }


def audit(phi_root: Path, *, run_commands: bool = True) -> dict[str, Any]:
    v3 = phi_root / "experiments/phi_research_v3"
    config_dir = phi_root / "config/v3"
    checks: dict[str, Any] = {}

    audit_payload = _load(v3 / "inventory/dataset_audit.json")
    checks["complete_dataset"] = {
        "listed": audit_payload["listed_file_count"],
        "actual": audit_payload["actual_mat_file_count"],
        "readable": audit_payload["readable_file_count"],
        "unreadable": audit_payload["unreadable_file_count"],
        "sessions": audit_payload["session_count"],
        "duplicate_groups": audit_payload["duplicate_array_group_count"],
        "fingerprint": audit_payload["dataset_fingerprint_sha256"],
    }
    checks["complete_dataset"]["status"] = (
        "pass"
        if checks["complete_dataset"]
        | {"status": "pass"}
        == {
            "listed": 15419,
            "actual": 15419,
            "readable": 15418,
            "unreadable": 1,
            "sessions": 441,
            "duplicate_groups": 1,
            "fingerprint": EXPECTED_FINGERPRINT,
            "status": "pass",
        }
        else "fail"
    )

    sidecars = []
    for sidecar in sorted(config_dir.glob("*.sha256")):
        sidecars.append(_sidecar_integrity(sidecar.with_suffix(".json"), sidecar))
    checks["frozen_config_hashes_and_crlf"] = {
        "files": sidecars,
        "status": "pass" if len(sidecars) == 5 and all(row["status"] == "pass" for row in sidecars) else "fail",
    }

    manifests: dict[str, dict[str, Any]] = {}
    manifest_checks = []
    support_ids: dict[str, set[str]] = {}
    query_ids: dict[str, set[str]] = {}
    for path in (
        config_dir / "acquisition_january_to_april_may_v3.json",
        config_dir / "acquisition_april_may_to_january_v3.json",
    ):
        manifest = _load(path)
        direction_spec = manifest["direction"]
        direction = f"{direction_spec['source']}_to_{direction_spec['target']}"
        manifests[direction] = manifest
        valid_hash, stored, calculated = _integrity(path, "manifest_sha256")
        seen: set[str] = set()
        duplicates = []
        partition_counts: dict[str, int] = defaultdict(int)
        for row in manifest["sessions"]:
            session_id = str(row["session_id"])
            if session_id in seen:
                duplicates.append(session_id)
            seen.add(session_id)
            partition_counts[str(row["partition"])] += 1
        support_ids[direction] = {
            str(row["session_id"]) for row in manifest["sessions"] if row["partition"] == "target_support"
        }
        query_ids[direction] = {
            str(row["session_id"]) for row in manifest["sessions"] if row["partition"] == "target_query"
        }
        manifest_checks.append(
            {
                "path": path.name,
                "direction": direction,
                "manifest_sha256": stored,
                "calculated_sha256": calculated,
                "session_count": len(seen),
                "duplicate_sessions": duplicates,
                "support_query_overlap": sorted(support_ids[direction] & query_ids[direction]),
                "partition_counts": dict(sorted(partition_counts.items())),
                "external_confirmation_required": manifest.get("external_confirmation_required"),
                "status": "pass"
                if valid_hash
                and len(seen) == 441
                and not duplicates
                and not support_ids[direction] & query_ids[direction]
                and manifest.get("external_confirmation_required") is True
                else "fail",
            }
        )
    checks["session_safe_manifests"] = {
        "manifests": manifest_checks,
        "status": "pass" if len(manifest_checks) == 2 and all(row["status"] == "pass" for row in manifest_checks) else "fail",
    }

    support_files = [
        v3 / "distributional_enrollment/support_draws.csv",
        v3 / "siamese_enrollment/support_draws.csv",
        v3 / "morphology_attributes/support_draws.csv",
    ]
    prediction_files = [
        v3 / "distributional_enrollment/query_predictions.csv.gz",
        v3 / "siamese_enrollment/query_predictions.csv.gz",
        v3 / "morphology_attributes/enrollment_query_predictions.csv.gz",
        v3 / "morphology_attributes/classification_predictions.csv",
        v3 / "morphology_attributes/retrieval_predictions.csv",
        v3 / "spatial_logistic/spatial_target_predictions.csv",
        v3 / "neural_full/neural_target_predictions.csv",
        v3 / "extended_morphology/classification_predictions.csv",
    ]
    membership = {
        "support": [_session_membership(path, support_ids) for path in support_files],
        "query": [_session_membership(path, query_ids) for path in prediction_files],
    }
    checks["support_query_membership"] = {
        **membership,
        "status": "pass"
        if all(row["status"] == "pass" for rows in membership.values() for row in rows)
        else "fail",
    }

    integrity_rows = []
    for path in sorted(v3.rglob("*.json")):
        payload = _load(path)
        field = "payload_sha256" if "payload_sha256" in payload else None
        if field is None:
            continue
        valid, stored, calculated = _integrity(path, field)
        integrity_rows.append(
            {
                "path": path.relative_to(phi_root).as_posix(),
                "stored": stored,
                "calculated": calculated,
                "status": "pass" if valid else "fail",
            }
        )
    checks["machine_readable_payload_integrity"] = {
        "checked": len(integrity_rows),
        "failed": [row for row in integrity_rows if row["status"] != "pass"],
        "status": "pass" if integrity_rows and all(row["status"] == "pass" for row in integrity_rows) else "fail",
    }

    spatial = _load(v3 / "spatial_logistic/spatial_results.json")
    neural = _load(v3 / "neural_full/neural_summary.json")
    classical = _load(v3 / "distributional_enrollment/distributional_enrollment_results.json")
    siamese_models = _load(v3 / "siamese_full/siamese_summary.json")
    siamese_enrollment = _load(v3 / "siamese_enrollment/siamese_enrollment_results.json")
    morphology = _load(v3 / "morphology_attributes/morphology_attributes_results.json")
    extended = _load(v3 / "extended_morphology/extended_morphology_results.json")
    checks["experiment_cardinality"] = {
        "spatial_results": spatial["result_count"],
        "session_neural_runs": neural["run_count"],
        "classical_enrollment_episodes": classical["episode_count"],
        "siamese_model_runs": siamese_models["run_count"],
        "siamese_enrollment_episodes": siamese_enrollment["episode_count"],
        "morphology_enrollment_episodes": morphology["enrollment"]["episode_count"],
        "morphology_windows": morphology["window_count"],
        "morphology_sessions": morphology["session_count"],
        "extended_morphology_results": len(extended["results"]),
        "extended_morphology_windows": extended["cache"]["window_count"],
    }
    checks["experiment_cardinality"]["status"] = (
        "pass"
        if checks["experiment_cardinality"]
        | {"status": "pass"}
        == {
            "spatial_results": 84,
            "session_neural_runs": 24,
            "classical_enrollment_episodes": 3564,
            "siamese_model_runs": 36,
            "siamese_enrollment_episodes": 3564,
            "morphology_enrollment_episodes": 2376,
            "morphology_windows": 15418,
            "morphology_sessions": 441,
            "extended_morphology_results": 4,
            "extended_morphology_windows": 15418,
            "status": "pass",
        }
        else "fail"
    )

    neural_runs = list(neural["runs"]) + list(siamese_models["runs"])
    seed_groups: dict[str, set[int]] = defaultdict(set)
    cuda_failures = []
    for row in neural_runs:
        group = "|".join(
            [
                str(row["direction"]),
                str(row.get("view", {}).get("name", row.get("heldout_class", "siamese"))
                    if isinstance(row.get("view"), dict)
                    else row.get("view", row.get("heldout_class", "siamese"))),
                str(row.get("architecture", "siamese")),
                str(row.get("heldout_class", "all_classes")),
            ]
        )
        seed_groups[group].add(int(row["seed"]))
        cuda = row.get("cuda", {})
        device_name = cuda.get("device_name", cuda.get("device"))
        if (
            "NVIDIA GeForce RTX 4060 Laptop GPU" not in str(device_name)
            or not str(cuda.get("torch_version", "")).endswith("+cu128")
            or int(row.get("peak_cuda_memory_bytes", 0)) <= 0
        ):
            cuda_failures.append(row["run_name"])
    seed_failures = {name: sorted(values) for name, values in seed_groups.items() if len(values) != 3}
    checks["cuda_and_neural_seeds"] = {
        "run_count": len(neural_runs),
        "cuda_failures": cuda_failures,
        "seed_group_count": len(seed_groups),
        "seed_failures": seed_failures,
        "status": "pass" if len(neural_runs) == 60 and not cuda_failures and not seed_failures else "fail",
    }
    checks["checkpoint_integrity"] = _checkpoints(
        phi_root,
        [v3 / "neural_full/neural_summary.json", v3 / "siamese_full/siamese_summary.json"],
    )

    selection_failures = [
        f"spatial:{index}"
        for index, row in enumerate(spatial["results"])
        if row.get("selection_used_target_query") is not False
    ]
    selection_failures.extend(
        f"neural:{row['run_name']}"
        for row in neural["runs"]
        if row.get("selection_used_target_query") is not False
    )
    selection_failures.extend(
        f"morphology:{row['direction']}:{row['view']}"
        for row in morphology["classification_results"]
        if row.get("selection_used_target_query") is not False
    )
    selection_failures.extend(
        f"extended:{row['direction']}:{row['view']}"
        for row in extended["results"]
        if row.get("selection_used_target_query") is not False
    )
    evidence_payloads = [
        spatial,
        neural,
        classical,
        siamese_models,
        siamese_enrollment,
        morphology,
        extended,
    ]
    confirmatory_labels = [
        str(payload.get("evidence_status"))
        for payload in evidence_payloads
        if "confirmatory" in str(payload.get("evidence_status", "")).lower()
        and "not confirmatory" not in str(payload.get("evidence_status", "")).lower()
    ]
    checks["retrospective_evidence_and_selection"] = {
        "target_query_selection_failures": selection_failures,
        "confirmatory_labels": confirmatory_labels,
        "status": "pass" if not selection_failures and not confirmatory_labels else "fail",
    }

    required = [
        v3 / "FINAL_TECHNICAL_REPORT.md",
        v3 / "BASELINE_CONTEXT.md",
        v3 / "REPRODUCIBILITY.md",
        v3 / "research_journal.json",
        v3 / "environment/core_vulnerability_audit.json",
        v3 / "visuals/reliability_diagrams.png",
        v3 / "visuals/risk_coverage_curves.png",
        v3 / "analysis/raw_registration/raw_registration_summary.json",
        v3 / "ontology/cleanlab_candidate_review.csv",
        v3 / "enrollment_analysis/paired_comparisons.csv",
        v3 / "extended_morphology_analysis/paired_statistics.csv",
        v3 / "secondary_window_predictions/january_to_april_may__cnn_seed20260805.npz",
        v3 / "secondary_window_predictions/april_may_to_january__tcn_seed20260805.npz",
        v3 / "exploratory_inherited/umap_session_embedding.png",
        v3 / "exploratory_inherited/development_summary.json",
        v3 / "exploratory_inherited/feature_drift_tests.csv",
        v3 / "exploratory_inherited/tooling_security_audit.json",
    ]
    checks["required_deliverables"] = {
        "paths": [path.relative_to(phi_root).as_posix() for path in required],
        "missing": [path.relative_to(phi_root).as_posix() for path in required if not path.is_file()],
    }
    checks["required_deliverables"]["status"] = (
        "pass" if not checks["required_deliverables"]["missing"] else "fail"
    )

    command_checks = {}
    if run_commands:
        command_checks["pytest"] = _run(
            [sys.executable, "-m", "pytest", "-q", "--tb=short"], cwd=phi_root
        )
        command_checks["compileall"] = _run(
            [sys.executable, "-m", "compileall", "-q", "src/phi_research"], cwd=phi_root
        )
    else:
        command_checks["status"] = "skipped_for_unit_test"
    checks["verification_commands"] = command_checks

    failed_checks = [name for name, row in checks.items() if row.get("status") == "fail"]
    if run_commands:
        failed_checks.extend(
            f"verification_commands.{name}"
            for name, row in command_checks.items()
            if isinstance(row, dict) and row.get("status") == "fail"
        )
    result = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 requirement-level completion audit",
        "evidence_status": "retrospective development; external confirmation required",
        "checks": checks,
        "failed_checks": sorted(failed_checks),
        "known_nonblocking_limitations": [
            "The historical target-query outcomes were already exposed before v3; no v3 result is confirmatory.",
            "Only two acquisition-era cohorts are available, with class/date and subtype/date confounding.",
            "Reliable subject/operator identity and an independent deployment domain are unavailable.",
            "The shared environment has a transformers/huggingface-hub pip-check conflict that does not affect imported v3 paths.",
            "pip-audit reports installer-package advisories and cannot audit the local CUDA-tagged torch build against PyPI.",
        ],
        "overall_status": "pass_with_recorded_limitations" if not failed_checks else "fail",
    }
    result["payload_sha256"] = canonical_json_hash(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.phi_root.resolve(), run_commands=True)
    args.output.resolve().write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(
        json.dumps(
            {
                "overall_status": result["overall_status"],
                "failed_checks": result["failed_checks"],
                "payload_sha256": result["payload_sha256"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    if result["failed_checks"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
