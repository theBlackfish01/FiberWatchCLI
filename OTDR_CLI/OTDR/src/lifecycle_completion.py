from __future__ import annotations

"""Machine-readable completion gate for the OTDR lifecycle study."""

import json
from pathlib import Path
from typing import Any

from .lifecycle_analysis import (
    REQUIRED_ENRICHMENT_VERSION,
    discover_runs,
)
from .study_state import atomic_json, validate_run


REGIME_EXPECTATIONS = {
    "full": 105,
    "trace_only": 105,
    "summary_only": 21,
}
REQUIRED_AUXILIARY = {
    "stress": ("stress", "lifecycle-stress-validation-v1"),
    "external": ("external", "lifecycle-external-validation-v1"),
    "kpsc_ablation": ("ablations", "lifecycle-kpsc-ablation-v1"),
    "tabpfn_v2": (
        "baselines/tabpfn_v2",
        "tabpfn-v2-representative-pilot",
    ),
}
def _cuda_manifest(run_dir: Path) -> bool:
    payload = json.loads(
        (run_dir / "manifest.json").read_text(encoding="utf-8")
    )
    device = payload.get("device")
    if device is None:
        device = payload.get("environment", {}).get("device")
    return str(device).startswith("cuda:")


def completion_audit(study_root: Path) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    all_source_runs: list[Path] = []
    for regime, expected in REGIME_EXPECTATIONS.items():
        runs = discover_runs(study_root, regime=regime)
        all_source_runs.extend(runs)
        checks[f"{regime}_matrix"] = {
            "passed": len(runs) == expected,
            "validated_runs": len(runs),
            "expected_runs": expected,
            "all_cuda": bool(runs) and all(
                _cuda_manifest(run) for run in runs
            ),
        }
        checks[f"{regime}_matrix"]["passed"] &= checks[
            f"{regime}_matrix"
        ]["all_cuda"]

        enrichment_root = study_root / "posthoc_calibration" / regime
        enriched = []
        for path in enrichment_root.glob("*/manifest.json"):
            valid, _ = validate_run(
                path.parent,
                expected={
                    "enrichment_version": REQUIRED_ENRICHMENT_VERSION
                },
            )
            if valid:
                enriched.append(path.parent)
        checks[f"{regime}_enrichment"] = {
            "passed": len(enriched) == expected,
            "validated_runs": len(enriched),
            "expected_runs": expected,
            "all_cuda": bool(enriched) and all(
                _cuda_manifest(run) for run in enriched
            ),
        }
        checks[f"{regime}_enrichment"]["passed"] &= checks[
            f"{regime}_enrichment"
        ]["all_cuda"]

        summary_path = (
            study_root / "tables" / regime / "headline_summary.json"
        )
        summary_complete = False
        reconstruction_passed = False
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary_complete = bool(summary.get("complete"))
        reconstruction_path = (
            study_root / "tables" / regime / "metric_reconstruction.json"
        )
        if reconstruction_path.exists():
            reconstruction_passed = bool(json.loads(
                reconstruction_path.read_text(encoding="utf-8")
            ).get("all_passed"))
        checks[f"{regime}_analysis"] = {
            "passed": summary_complete and reconstruction_passed,
            "summary_complete": summary_complete,
            "reconstruction_passed": reconstruction_passed,
        }

    for name, (relative, run_id) in REQUIRED_AUXILIARY.items():
        root = study_root / relative
        valid, reason = validate_run(root, expected={"run_id": run_id})
        checks[name] = {
            "passed": valid and _cuda_manifest(root) if valid else False,
            "manifest_valid": valid,
            "reason": reason,
            "cuda": _cuda_manifest(root) if valid else False,
        }

    for name in ("TEST_VALIDATION.json", "PLOT_INSPECTION.json"):
        path = study_root / name
        payload = (
            json.loads(path.read_text(encoding="utf-8"))
            if path.exists()
            else {}
        )
        checks[name] = {
            "passed": bool(payload.get("passed")),
            "path": str(path),
        }

    source_snapshot = study_root / "SOURCE_SNAPSHOT.json"
    checks["source_snapshot"] = {
        "passed": source_snapshot.exists()
        and bool(json.loads(
            source_snapshot.read_text(encoding="utf-8")
        ).get("files")),
        "path": str(source_snapshot),
    }
    failures = [
        line
        for line in (study_root / "failures.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    checks["failure_log"] = {
        "passed": True,
        "recorded_failures": len(failures),
        "interpretation": (
            "Recorded failures are allowed only when their disposition is "
            "explained in the final report."
        ),
    }
    complete = all(value["passed"] for value in checks.values())
    result = {
        "schema_version": 1,
        "complete": complete,
        "checks": checks,
        "validated_source_runs": len(all_source_runs),
    }
    atomic_json(study_root / "COMPLETION_AUDIT.json", result)
    return result
