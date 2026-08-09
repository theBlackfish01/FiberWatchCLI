"""Finalize reproducibility metadata and the machine-readable PHI-OTDR v3 artifact index."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import sklearn
import torch

from .data_contract import canonical_json_hash
from .session_neural_v3 import SessionNet
from .siamese_session_v3 import SiameseSessionEncoder


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(command: list[str], *, cwd: Path) -> dict[str, object]:
    completed = subprocess.run(command, cwd=cwd, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _package_metadata(distributions: list[str]) -> list[dict[str, object]]:
    output = []
    for distribution in distributions:
        metadata = importlib.metadata.metadata(distribution)
        output.append(
            {
                "distribution": distribution,
                "version": importlib.metadata.version(distribution),
                "license_expression": metadata.get("License-Expression"),
                "license_field": metadata.get("License"),
                "homepage": metadata.get("Home-page") or metadata.get("Project-URL"),
            }
        )
    return output


def _analysis_package_metadata(python_path: Path, packages: list[str], cwd: Path) -> list[dict[str, object]]:
    code = (
        "import importlib.metadata,json; packages="
        + repr(packages)
        + "; print(json.dumps([{'distribution':p,'version':importlib.metadata.version(p),"
        "'license_expression':importlib.metadata.metadata(p).get('License-Expression'),"
        "'license_field':importlib.metadata.metadata(p).get('License'),"
        "'homepage':importlib.metadata.metadata(p).get('Home-page') or importlib.metadata.metadata(p).get('Project-URL')}"
        " for p in packages]))"
    )
    result = _run([str(python_path), "-c", code], cwd=cwd)
    if result["returncode"] != 0:
        raise RuntimeError(f"Analysis package metadata failed: {result}")
    return json.loads(str(result["stdout"]))


def _vulnerability_audit_summary(path: Path) -> dict[str, object]:
    """Summarize pip-audit output without treating skipped packages as clean."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    affected = []
    skipped = []
    unique_advisories: set[str] = set()
    for dependency in payload.get("dependencies", []):
        vulnerabilities = dependency.get("vulns", [])
        if vulnerabilities:
            ids = sorted({str(row["id"]) for row in vulnerabilities})
            unique_advisories.update(ids)
            affected.append(
                {
                    "name": dependency["name"],
                    "version": dependency.get("version"),
                    "unique_advisory_ids": ids,
                    "fixed_versions": sorted(
                        {
                            str(version)
                            for row in vulnerabilities
                            for version in row.get("fix_versions", [])
                        }
                    ),
                }
            )
        if dependency.get("skip_reason"):
            skipped.append(
                {
                    "name": dependency["name"],
                    "version": dependency.get("version"),
                    "reason": dependency["skip_reason"],
                }
            )
    return {
        "path": path.as_posix(),
        "sha256": _sha256(path),
        "dependency_records": len(payload.get("dependencies", [])),
        "affected_package_count": len(affected),
        "unique_advisory_count": len(unique_advisories),
        "affected_packages": affected,
        "skipped_packages": skipped,
        "interpretation": (
            "A skipped package was not audited and must not be interpreted as vulnerability-free. "
            "Duplicate advisory rows in the raw scanner output are deduplicated here by advisory ID."
        ),
    }


def _parameter_counts(phi_root: Path, summary_paths: list[Path]) -> list[dict[str, object]]:
    neural_config = json.loads((phi_root / "config/v3/session_neural_v3.json").read_text(encoding="utf-8"))
    enrollment_config = json.loads((phi_root / "config/v3/enrollment_v3.json").read_text(encoding="utf-8"))
    neural_dimensions = {}
    for path in (phi_root / "experiments/phi_research_v3/neural_window_cache").glob("*.npz"):
        with np.load(path, allow_pickle=False) as bundle:
            neural_dimensions[path.stem] = int(bundle["features"].shape[1]) + 1
    with np.load(
        phi_root
        / "experiments/phi_research_v3/spatial_logistic/session_views/registered_position__temporal_difference_energy__dynamics.npz",
        allow_pickle=False,
    ) as bundle:
        siamese_input_dim = int(bundle["features"].shape[1])
    output = []
    for summary_path in summary_paths:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for row in summary["runs"]:
            checkpoint_path = phi_root / row["checkpoint_path"]
            if "architecture" in row:
                view_name = str(row["view"]["name"])
                model = SessionNet(
                    input_dim=neural_dimensions[view_name],
                    hidden_dim=int(neural_config["model"]["hidden_dim"]),
                    dropout=float(neural_config["model"]["dropout"]),
                    architecture=str(row["architecture"]),
                )
            else:
                model = SiameseSessionEncoder(
                    input_dim=siamese_input_dim,
                    hidden_dim=int(enrollment_config["siamese"]["hidden_dim"]),
                    embedding_dim=int(enrollment_config["siamese"]["embedding_dim"]),
                )
            parameter_count = int(sum(value.numel() for value in model.parameters()))
            output.append(
                {
                    "family": "session_neural" if "architecture" in row else "siamese_session",
                    "run_name": row["run_name"],
                    "direction": row["direction"],
                    "architecture": row.get("architecture", "siamese"),
                    "view": row.get("view", {}).get("name") if isinstance(row.get("view"), dict) else row.get("view"),
                    "parameter_count": parameter_count,
                    "checkpoint_path": row["checkpoint_path"],
                    "checkpoint_sha256": row["checkpoint_sha256"],
                    "peak_cuda_memory_bytes": row["peak_cuda_memory_bytes"],
                    "training_seconds": row.get("training_seconds_selected_candidate", row.get("training_seconds")),
                    "inference_seconds": row.get("target_inference_seconds", row.get("embedding_inference_seconds")),
                }
            )
    return output


def _copy_snapshots(phi_root: Path, v3_root: Path) -> dict[str, list[str]]:
    inventory_dir = v3_root / "inventory"
    config_dir = v3_root / "config_snapshots"
    secondary_dir = v3_root / "secondary_window_predictions"
    exploratory_dir = v3_root / "exploratory_inherited"
    inventory_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    secondary_dir.mkdir(parents=True, exist_ok=True)
    exploratory_dir.mkdir(parents=True, exist_ok=True)
    inventory_sources = [
        phi_root / "experiments/phi_research_v2/recovered_dataset/audit_complete/dataset_audit.json",
        phi_root / "experiments/phi_research_v2/recovered_dataset/audit_complete/dataset_inventory.csv",
        phi_root / "experiments/phi_research_v2/recovered_dataset/audit_complete/session_inventory.csv",
        phi_root / "experiments/phi_research_v2/recovered_dataset/audit_complete/duplicate_arrays.json",
        phi_root / "experiments/phi_research_v2/recovered_dataset/complete_dataset_contract_v1.json",
        phi_root / "experiments/phi_research_v2/recovered_dataset/complete_dataset_contract_v1.sha256",
    ]
    copied_inventory = []
    for source in inventory_sources:
        destination = inventory_dir / source.name
        shutil.copy2(source, destination)
        copied_inventory.append(destination.relative_to(phi_root).as_posix())
    copied_configs = []
    for source in sorted((phi_root / "config/v3").glob("*")):
        if source.is_file():
            destination = config_dir / source.name
            shutil.copy2(source, destination)
            copied_configs.append(destination.relative_to(phi_root).as_posix())
    copied_window_predictions = []
    frozen_root = phi_root / "experiments/phi_research_v2/recovered_dataset/frozen_neural_target"
    for source in sorted(frozen_root.glob("*/*/frozen_target_predictions.npz")):
        direction = source.parents[1].name
        model_run = source.parent.name
        destination = secondary_dir / f"{direction}__{model_run}.npz"
        shutil.copy2(source, destination)
        copied_window_predictions.append(destination.relative_to(phi_root).as_posix())
    exploratory_sources = [
        phi_root / "experiments/phi_research_v2/development_summary.json",
        phi_root
        / "experiments/phi_research_v2/recovered_dataset/tooling_evaluation/umap_session_embedding.png",
        phi_root
        / "experiments/phi_research_v2/recovered_dataset/tooling_evaluation/feature_drift_tests.csv",
        phi_root
        / "experiments/phi_research_v2/recovered_dataset/tooling_evaluation/tooling_security_audit.json",
    ]
    copied_exploratory = []
    for source in exploratory_sources:
        destination = exploratory_dir / source.name
        shutil.copy2(source, destination)
        copied_exploratory.append(destination.relative_to(phi_root).as_posix())
    return {
        "inventory": copied_inventory,
        "configs": copied_configs,
        "secondary_window_predictions": copied_window_predictions,
        "exploratory_inherited": copied_exploratory,
    }


def _write_commands(path: Path) -> None:
    path.write_text(
        """# PHI-OTDR v3 reproducibility commands

Run from `OTDR_CLI/PHI-OTDR` with the repository `.venv-zsl` environment unless noted.

The target-query evidence is retrospective. Re-running commands does not convert it into confirmation.

```powershell
$env:PYTHONPATH='src'

# Verify complete data and frozen acquisition manifests
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.era_cli verify-dataset --help

# Spatial morphology experiments (deterministic cache is reused)
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.spatial_experiment --bundle experiments\\phi_research_v3\\cache\\complete_morphology_v3.npz --manifests config\\v3\\acquisition_january_to_april_may_v3.json config\\v3\\acquisition_april_may_to_january_v3.json --protocol config\\v3\\acquisition_protocol_v3.json --protocol-hash config\\v3\\acquisition_protocol_v3.sha256 --output-dir experiments\\phi_research_v3\\spatial_logistic

# Session neural models: CUDA is mandatory and CPU fallback raises an error
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.session_neural_v3 --help

# Distributional and Siamese enrollment
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.distributional_enrollment_v3 --help
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.siamese_session_v3 --help
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.siamese_enrollment_v3 --help

# Cluster-aware enrollment statistics
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.enrollment_analysis_v3 --classical experiments\\phi_research_v3\\distributional_enrollment\\distributional_enrollment_results.json --siamese experiments\\phi_research_v3\\siamese_enrollment\\siamese_enrollment_results.json --manifest config\\v3\\acquisition_january_to_april_may_v3.json --manifest config\\v3\\acquisition_april_may_to_january_v3.json --output-dir experiments\\phi_research_v3\\enrollment_analysis

# Complete-cohort Pandera/ontology audit uses the isolated analysis environment
$env:PYTHONPATH='OTDR_CLI\\PHI-OTDR\\src'
.\\tmp\\phi-analysis-tools\\Scripts\\python.exe -m phi_research.ontology_audit_v3 --help

# Morphology-before-name experiment and statistics
$env:PYTHONPATH='src'
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.morphology_attributes_v3 --bundle experiments\\phi_research_v3\\cache\\complete_morphology_v3.npz --bundle-metadata experiments\\phi_research_v3\\cache\\complete_morphology_v3.json --config config\\v3\\morphology_attributes_v3.json --config-hash config\\v3\\morphology_attributes_v3.sha256 --manifest config\\v3\\acquisition_january_to_april_may_v3.json --manifest config\\v3\\acquisition_april_may_to_january_v3.json --spatial-results experiments\\phi_research_v3\\spatial_logistic\\spatial_results.json --output-dir experiments\\phi_research_v3\\morphology_attributes
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.morphology_attribute_analysis_v3 --morphology experiments\\phi_research_v3\\morphology_attributes\\morphology_attributes_results.json --classical experiments\\phi_research_v3\\distributional_enrollment\\distributional_enrollment_results.json --siamese experiments\\phi_research_v3\\siamese_enrollment\\siamese_enrollment_results.json --spatial-predictions experiments\\phi_research_v3\\spatial_logistic\\spatial_target_predictions.csv --output-dir experiments\\phi_research_v3\\morphology_attribute_analysis

# Bounded wavelet-energy and spatial-effective-rank missing control
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.extended_morphology_v3 --phi-root . --data-root src\\data\\das_data --config config\\v3\\extended_morphology_v3.json --config-hash config\\v3\\extended_morphology_v3.sha256 --manifest config\\v3\\acquisition_january_to_april_may_v3.json --manifest config\\v3\\acquisition_april_may_to_january_v3.json --output-dir experiments\\phi_research_v3\\extended_morphology
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.extended_morphology_analysis_v3 --extended experiments\\phi_research_v3\\extended_morphology\\extended_morphology_results.json --morphology experiments\\phi_research_v3\\morphology_attributes\\morphology_attributes_results.json --output-dir experiments\\phi_research_v3\\extended_morphology_analysis

# Uncertainty plots
..\\..\\.venv-zsl\\Scripts\\python.exe -m phi_research.report_visuals_v3 --spatial experiments\\phi_research_v3\\spatial_logistic\\spatial_target_predictions.csv --attribute experiments\\phi_research_v3\\morphology_attributes\\classification_predictions.csv --neural experiments\\phi_research_v3\\neural_full\\neural_target_predictions.csv --output-dir experiments\\phi_research_v3\\visuals

# Full verification
..\\..\\.venv-zsl\\Scripts\\python.exe -m pytest -q --tb=short
```

The original complete morphology extraction takes several minutes and requires the ignored local `das_data` directory. It is unnecessary when the content-hashed cache is present.
""",
        encoding="utf-8",
    )


def _journal() -> list[dict[str, object]]:
    return [
        {"stage": "repository_and_complete_data_audit", "category": "development audit", "decision": "continue", "result": "15,419 listed, 15,418 readable, 441 sessions; one unreadable file and one duplicate-array pair"},
        {"stage": "safe_analysis_tooling", "category": "tooling evaluation", "decision": "accept selectively", "result": "Pandera, Cleanlab, UMAP, SciPy/statsmodels and pip-audit retained in an isolated environment; Evidently, Frouros and YData Profiling rejected"},
        {"stage": "spatial_registration_smoke", "category": "smoke", "decision": "continue", "result": "padding-based registration and boundary tests passed; no circular wrap"},
        {"stage": "complete_morphology_cache", "category": "development preprocessing", "decision": "freeze", "result": "15,418 windows, 441 sessions, 102 window descriptors"},
        {"stage": "spatial_logistic", "category": "retrospective development", "decision": "continue", "result": "registered dynamics improved reverse transfer; invariant fused remained strongest balanced reverse representation"},
        {"stage": "targeted_tree", "category": "exploratory retrospective", "decision": "control", "result": "higher reverse mean macro-F1 but poorer worst-class recall"},
        {"stage": "padding_shift_augmentation", "category": "exploratory retrospective", "decision": "auxiliary only", "result": "competitive mean results with class collapse; not primary"},
        {"stage": "session_neural_smoke", "category": "CUDA smoke", "decision": "continue to bounded full run", "result": "CUDA/AMP verified on RTX 4060 Laptop GPU"},
        {"stage": "session_neural_full", "category": "retrospective development negative", "decision": "stop scaling", "result": "24 DeepSets/attention runs underperformed registered morphology and had poor calibration/worst-class recall"},
        {"stage": "physics_guided_ssl", "category": "stopped before expensive run", "decision": "stop", "result": "neural development gate failed; no scientific justification for scaling SSL"},
        {"stage": "distributional_enrollment_integration", "category": "failed integration then fixed", "decision": "rerun from scratch", "result": "PCA rejected a 64-bit seed; normalized to uint32 with regression test; no partial results retained"},
        {"stage": "distributional_enrollment", "category": "retrospective development", "decision": "retain", "result": "3,564 classical episodes; five-shot sliced-Wasserstein/hybrid strongest descriptively"},
        {"stage": "siamese_session_smoke", "category": "CUDA smoke", "decision": "continue to bounded full run", "result": "session-level contrastive/classification training verified on CUDA"},
        {"stage": "siamese_session_and_enrollment", "category": "retrospective development negative", "decision": "stop ordinary Siamese scaling", "result": "36 CUDA encoders and 3,564 enrollment episodes underperformed classical distributions"},
        {"stage": "enrollment_statistics", "category": "retrospective statistical synthesis", "decision": "interpret descriptively", "result": "support-identical pairing and exact six-class tests; no Enrollment-H comparison survived BH correction"},
        {"stage": "ontology_audit", "category": "complete-cohort audit", "decision": "retain weak composite background", "result": "nine nearby-activity background sessions all occur on one April/May date; seventh class is acquisition-confounded"},
        {"stage": "ontology_sensitivity", "category": "post-hoc sensitivity", "decision": "not a primary explanation", "result": "excluding nine sessions changed macro-F1 by -0.0132 through query composition and -0.0033 through reverse source refitting"},
        {"stage": "morphology_before_name", "category": "retrospective development", "decision": "retain as interpretable control", "result": "morphology-only matched invariant reverse transfer; position hurt; retrieval gain in the hard direction was uncertain; enrollment remained below distributional controls"},
        {"stage": "wavelet_and_effective_rank_control", "category": "retrospective development negative", "decision": "stop", "result": "Haar scale energy and spatial covariance effective rank were weak alone; fusion changed forward macro-F1 by +0.0235 (q=0.401) and reverse by -0.0398 (q=0.158)"},
        {"stage": "confirmatory_status", "category": "evidence classification", "decision": "external data required", "result": "no v3 result is independently confirmatory because historical target outcomes were already exposed"},
    ]


def _write_journal(v3_root: Path) -> None:
    rows = _journal()
    payload = {"schema_version": 1, "entries": rows}
    payload["payload_sha256"] = canonical_json_hash(payload)
    (v3_root / "research_journal.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    lines = ["# PHI-OTDR v3 research journal", ""]
    for row in rows:
        lines.extend(
            [
                f"## {row['stage']}",
                "",
                f"- Category: {row['category']}",
                f"- Decision: {row['decision']}",
                f"- Result: {row['result']}",
                "",
            ]
        )
    (v3_root / "RESEARCH_JOURNAL.md").write_text("\n".join(lines), encoding="utf-8")


def _artifact_index(phi_root: Path, v3_root: Path) -> dict[str, object]:
    index_path = v3_root / "artifact_index.json"
    entries = []
    for path in sorted(v3_root.rglob("*")):
        if not path.is_file() or path == index_path:
            continue
        relative = path.relative_to(phi_root).as_posix()
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
                "suffix": path.suffix.lower(),
                "top_level_group": path.relative_to(v3_root).parts[0],
            }
        )
    required = {
        "frozen_protocol_and_hashes": ["config_snapshots/acquisition_protocol_v3.json", "config_snapshots/acquisition_protocol_v3.sha256"],
        "dataset_and_session_inventories": ["inventory/dataset_inventory.csv", "inventory/session_inventory.csv"],
        "environment_and_cuda": [
            "environment/environment_cuda.json",
            "environment/core_packages.txt",
            "environment/core_vulnerability_audit.json",
        ],
        "training_logs_and_checkpoints": ["neural_full/neural_summary.json", "siamese_full/siamese_summary.json"],
        "predictions": [
            "spatial_logistic/spatial_target_predictions.csv",
            "morphology_attributes/classification_predictions.csv",
            "secondary_window_predictions/january_to_april_may__cnn_seed20260805.npz",
            "secondary_window_predictions/april_may_to_january__cnn_seed20260805.npz",
        ],
        "support_draws": ["distributional_enrollment/support_draws.csv", "siamese_enrollment/support_draws.csv"],
        "calibration_and_confusions": ["visuals/calibration_summary.json", "visuals/reliability_diagrams.png", "visuals/confusion_january_to_april_may.png"],
        "registration_diagnostics": ["analysis/raw_registration/raw_registration_summary.json"],
        "cleanlab_review": ["ontology/cleanlab_candidate_review.csv"],
        "exploratory_embedding": ["exploratory_inherited/umap_session_embedding.png"],
        "ordered_aggregation_control": ["exploratory_inherited/development_summary.json"],
        "drift_tests": ["exploratory_inherited/feature_drift_tests.csv"],
        "tooling_and_sbom": ["exploratory_inherited/tooling_security_audit.json"],
        "statistical_tables": [
            "enrollment_analysis/paired_comparisons.csv",
            "morphology_attribute_analysis/classification_paired_statistics.csv",
            "extended_morphology_analysis/paired_statistics.csv",
        ],
        "journal": ["research_journal.json", "RESEARCH_JOURNAL.md"],
        "commands": ["REPRODUCIBILITY.md"],
        "completion_audit": ["completion_audit.json"],
        "final_reports": ["FINAL_TECHNICAL_REPORT.md", "BASELINE_CONTEXT.md"],
    }
    checklist = {}
    for name, paths in required.items():
        existence = [(v3_root / path).is_file() for path in paths]
        checklist[name] = {"status": "pass" if all(existence) else "missing", "paths": paths}
    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 machine-readable artifact index",
        "root": v3_root.relative_to(phi_root).as_posix(),
        "file_count_excluding_index": len(entries),
        "total_bytes_excluding_index": sum(row["bytes"] for row in entries),
        "entries": entries,
        "required_artifact_checklist": checklist,
        "all_required_groups_present": all(row["status"] == "pass" for row in checklist.values()),
        "generated_artifacts_ignored": True,
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    index_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return payload


def finalize(
    *,
    phi_root: Path,
    analysis_python: Path,
) -> dict[str, object]:
    v3_root = phi_root / "experiments/phi_research_v3"
    environment_dir = v3_root / "environment"
    environment_dir.mkdir(parents=True, exist_ok=True)
    snapshots = _copy_snapshots(phi_root, v3_root)
    _write_commands(v3_root / "REPRODUCIBILITY.md")
    _write_journal(v3_root)

    pip_freeze = _run([sys.executable, "-m", "pip", "freeze"], cwd=phi_root)
    if pip_freeze["returncode"] != 0:
        raise RuntimeError(f"pip freeze failed: {pip_freeze}")
    (environment_dir / "core_packages.txt").write_text(
        str(pip_freeze["stdout"]) + "\n", encoding="utf-8"
    )
    pip_check = _run([sys.executable, "-m", "pip", "check"], cwd=phi_root)
    nvidia_smi = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        cwd=phi_root,
    )
    parameter_counts = _parameter_counts(
        phi_root,
        [
            v3_root / "neural_full/neural_summary.json",
            v3_root / "siamese_full/siamese_summary.json",
        ],
    )
    core_metadata = _package_metadata(
        ["numpy", "scipy", "scikit-learn", "joblib", "matplotlib", "torch", "pytest"]
    )
    analysis_metadata = _analysis_package_metadata(
        analysis_python,
        ["pandera", "cleanlab", "umap-learn", "pip-audit", "statsmodels", "seaborn"],
        phi_root,
    )
    vulnerability_path = environment_dir / "core_vulnerability_audit.json"
    if not vulnerability_path.is_file():
        raise FileNotFoundError(
            "Missing core vulnerability audit; run the isolated pip-audit command before finalization"
        )
    vulnerability_audit = _vulnerability_audit_summary(vulnerability_path)
    cuda = {
        "available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "compute_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None,
        "total_vram_bytes": int(torch.cuda.get_device_properties(0).total_memory) if torch.cuda.is_available() else None,
        "float16_autocast_used": True,
        "cpu_fallback_allowed": False,
        "nvidia_smi": nvidia_smi,
    }
    if not cuda["available"]:
        raise RuntimeError("CUDA unavailable during final environment audit")
    environment = {
        "schema_version": 1,
        "python": {"version": sys.version, "executable": sys.executable, "platform": platform.platform()},
        "numpy_version": np.__version__,
        "scikit_learn_version": sklearn.__version__,
        "cuda": cuda,
        "pip_check": pip_check,
        "vulnerability_audit": vulnerability_audit,
        "core_package_metadata": core_metadata,
        "isolated_analysis_package_metadata": analysis_metadata,
        "model_runs": parameter_counts,
        "parameter_count_summary": {
            "minimum": min(row["parameter_count"] for row in parameter_counts),
            "maximum": max(row["parameter_count"] for row in parameter_counts),
            "unique": sorted({row["parameter_count"] for row in parameter_counts}),
            "run_count": len(parameter_counts),
        },
        "license_interpretation": [
            "Package metadata is recorded for attribution review and is not legal advice.",
            "Cleanlab remains an isolated analysis tool; it is not a runtime dependency or copied library code.",
            "Model and plotting libraries require preservation of their upstream notices when redistributed.",
        ],
    }
    environment["payload_sha256"] = canonical_json_hash(environment)
    (environment_dir / "environment_cuda.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True), encoding="utf-8"
    )

    inherited = {
        "schema_version": 1,
        "references": [
            {
                "role": "exploratory UMAP",
                "path": "experiments/phi_research_v2/recovered_dataset/tooling_evaluation/umap_session_embedding.png",
            },
            {
                "role": "feature drift tests with multiple-testing correction",
                "path": "experiments/phi_research_v2/recovered_dataset/tooling_evaluation/feature_drift_tests.csv",
            },
            {
                "role": "safe tooling audit and SBOM",
                "path": "experiments/phi_research_v2/recovered_dataset/tooling_evaluation/tooling_security_audit.json",
            },
            {
                "role": "secondary frozen window-level CNN/TCN results",
                "path": "experiments/phi_research_v2/recovered_dataset/frozen_neural_summary.json",
            },
        ],
    }
    for row in inherited["references"]:
        path = phi_root / row["path"]
        row["sha256"] = _sha256(path)
        row["bytes"] = path.stat().st_size
    inherited["payload_sha256"] = canonical_json_hash(inherited)
    (v3_root / "inherited_references.json").write_text(
        json.dumps(inherited, indent=2, sort_keys=True), encoding="utf-8"
    )
    index = _artifact_index(phi_root, v3_root)
    return {
        "snapshots": snapshots,
        "environment_payload_sha256": environment["payload_sha256"],
        "artifact_index_payload_sha256": index["payload_sha256"],
        "artifact_files": index["file_count_excluding_index"],
        "all_required_groups_present": index["all_required_groups_present"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi-root", type=Path, required=True)
    parser.add_argument("--analysis-python", type=Path, required=True)
    args = parser.parse_args()
    result = finalize(phi_root=args.phi_root.resolve(), analysis_python=args.analysis_python.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
