"""Executable complete-cohort data-contract and filename-ontology audit for PHI-OTDR v3."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data_contract import CLASS_NAMES, canonical_json_hash


NEARBY_BACKGROUND_PATTERNS = ("_walk", "heavy_steps")
SUBTYPE_PATTERNS = {
    "speed": ("fast", "slow", "median"),
    "locomotion": ("walk", "run"),
    "spatial_contact": ("center", "edge", "fiber"),
    "shake_direction": ("up", "down"),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def filename_attributes(session_id: str, class_name: str) -> dict[str, object]:
    lower = session_id.lower()
    result: dict[str, object] = {}
    for field, values in SUBTYPE_PATTERNS.items():
        found = [value for value in values if re.search(rf"(?:^|_){re.escape(value)}(?:\d|_|$)", lower)]
        result[field] = "+".join(found) if found else "unspecified"
    duration = re.search(r"(?:^|_)(?:[a-z]+)?(\d+)s(?:_|$)", lower)
    distance = re.search(r"(?:^|_)(\d+)cm(?:_|$)", lower)
    result["duration_seconds"] = int(duration.group(1)) if duration else None
    result["distance_cm"] = int(distance.group(1)) if distance else None
    result["nearby_non_target_activity"] = bool(
        class_name == "background" and any(pattern in lower for pattern in NEARBY_BACKGROUND_PATTERNS)
    )
    result["background_subtype"] = (
        "nearby_non_target_activity"
        if result["nearby_non_target_activity"]
        else "nominal_background"
        if class_name == "background"
        else "not_background"
    )
    return result


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _pandera_validate(
    raw_inventory: pd.DataFrame,
    morphology_metadata: pd.DataFrame,
) -> str:
    try:
        import pandera
        import pandera.pandas as pa
    except ImportError as exc:  # pragma: no cover - tested in isolated analysis environment
        raise RuntimeError("Pandera is required in the isolated PHI analysis environment") from exc

    raw_schema = pa.DataFrameSchema(
        {
            "rel_path": pa.Column(str, pa.Check.str_length(min_value=1), unique=True),
            "class_id": pa.Column(int, pa.Check.isin(range(len(CLASS_NAMES)))),
            "class_name": pa.Column(str, pa.Check.isin(CLASS_NAMES)),
            "session_id": pa.Column(str, pa.Check.str_length(min_value=1)),
            "window_id": pa.Column(int, pa.Check.ge(0)),
            "exists": pa.Column(bool),
            "readable": pa.Column(bool),
        },
        strict=False,
        coerce=True,
    )
    morphology_schema = pa.DataFrameSchema(
        {
            "rel_path": pa.Column(str, pa.Check.str_length(min_value=1), unique=True),
            "label": pa.Column(int, pa.Check.isin(range(len(CLASS_NAMES)))),
            "session": pa.Column(str, pa.Check.str_length(min_value=1)),
            "window_id": pa.Column(int, pa.Check.ge(0)),
            "partition": pa.Column(str, pa.Check.isin(
                ["source_train", "source_validation", "source_calibration", "target_support", "target_calibration", "target_query"]
            )),
            "era": pa.Column(str, pa.Check.isin(["january", "april_may"])),
            "date_token": pa.Column(str, pa.Check.str_matches(r"^\d{6}$")),
            "source_token": pa.Column(str, pa.Check.str_length(min_value=1)),
        },
        strict=False,
        coerce=True,
    )
    raw_schema.validate(raw_inventory, lazy=True)
    morphology_schema.validate(morphology_metadata, lazy=True)
    return str(pandera.__version__)


def audit(
    *,
    session_inventory_path: Path,
    raw_inventory_path: Path,
    raw_audit_path: Path,
    morphology_path: Path,
    morphology_metadata_path: Path,
    cleanlab_path: Path,
    prior_pandera_path: Path,
    output_dir: Path,
) -> dict[str, object]:
    session_rows = _read_csv(session_inventory_path)
    if len(session_rows) != 441 or len({row["session_id"] for row in session_rows}) != 441:
        raise ValueError("Session inventory must contain 441 unique sessions")
    raw_frame = pd.read_csv(raw_inventory_path)
    raw_audit = json.loads(raw_audit_path.read_text(encoding="utf-8"))
    if len(raw_frame) != int(raw_audit["listed_file_count"]):
        raise ValueError("Raw inventory row count disagrees with audit")
    readable = raw_frame["readable"].astype(str).str.lower() == "true"
    if int(readable.sum()) != 15418 or int((~readable).sum()) != 1:
        raise ValueError("Raw readability counts changed")
    if not raw_frame.loc[readable, "shape"].eq("10000x12").all():
        raise ValueError("Readable raw shape contract failed")
    if not raw_frame.loc[readable, "dtype"].eq("uint16").all():
        raise ValueError("Readable raw dtype contract failed")
    if not raw_frame.loc[readable, "finite"].astype(str).str.lower().eq("true").all():
        raise ValueError("Readable raw finiteness contract failed")

    with np.load(morphology_path, allow_pickle=False) as source:
        bundle = {key: source[key] for key in source.files}
    if bundle["features"].shape != (15418, 102):
        raise ValueError(f"Unexpected complete morphology shape: {bundle['features'].shape}")
    if not np.isfinite(bundle["features"]).all():
        raise ValueError("Non-finite complete morphology feature")
    if len(np.unique(bundle["sessions"].astype(str))) != 441:
        raise ValueError("Complete morphology session count changed")
    if len(np.unique(bundle["rel_paths"].astype(str))) != 15418:
        raise ValueError("Complete morphology paths are not unique")
    morphology_metadata = pd.DataFrame(
        {
            "rel_path": bundle["rel_paths"].astype(str),
            "label": bundle["labels"].astype(int),
            "session": bundle["sessions"].astype(str),
            "window_id": bundle["window_ids"].astype(int),
            "partition": bundle["partitions"].astype(str),
            "era": bundle["eras"].astype(str),
            "date_token": bundle["date_tokens"].astype(str),
            "source_token": bundle["source_tokens"].astype(str),
        }
    )
    pandera_version = _pandera_validate(raw_frame, morphology_metadata)

    grouped = morphology_metadata.groupby("session").agg(
        label_count=("label", "nunique"),
        era_count=("era", "nunique"),
        date_count=("date_token", "nunique"),
        source_count=("source_token", "nunique"),
        readable_windows=("rel_path", "size"),
    )
    if not (grouped[["label_count", "era_count", "date_count", "source_count"]] == 1).all().all():
        raise ValueError("Morphology session metadata inconsistency")

    taxonomy_rows = []
    subtype_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in session_rows:
        attributes = filename_attributes(row["session_id"], row["class_name"])
        output = {
            "session_id": row["session_id"],
            "class_name": row["class_name"],
            "class_id": int(row["class_id"]),
            "date_token": row["date_token"],
            "source_token": row["source_token"],
            "listed_window_count": int(row["window_count"]),
            **attributes,
            "audit_only_not_inference_feature": True,
        }
        taxonomy_rows.append(output)
        for field in SUBTYPE_PATTERNS:
            if attributes[field] != "unspecified":
                subtype_counts[f"{row['class_name']}:{field}"][str(attributes[field])] += 1

    nearby = [row for row in taxonomy_rows if row["nearby_non_target_activity"]]
    if len(nearby) != 9 or {row["date_token"] for row in nearby} != {"220509"}:
        raise ValueError("Expected nine one-date nearby-activity background sessions")

    cleanlab_rows = _read_csv(cleanlab_path)
    if len(cleanlab_rows) != 441:
        raise ValueError("Cleanlab audit must cover 441 sessions")
    taxonomy_by_session = {str(row["session_id"]): row for row in taxonomy_rows}
    candidate_rows = []
    for row in sorted(cleanlab_rows, key=lambda item: float(item["label_quality_score"])):
        taxonomy = taxonomy_by_session[row["session"]]
        score = float(row["label_quality_score"])
        flagged = row["cleanlab_flagged"].lower() == "true"
        ambiguity = bool(taxonomy["nearby_non_target_activity"])
        priority = "high" if flagged or ambiguity or score < 0.10 else "medium" if score < 0.50 else "routine"
        candidate_rows.append(
            {
                "session_id": row["session"],
                "class_name": row["class_name"],
                "era": row["era"],
                "date_token": row["date_token"],
                "source_token": row["source_token"],
                "window_count": int(row["window_count"]),
                "oof_predicted_class": row["oof_predicted_class"],
                "label_quality_score": score,
                "cleanlab_flagged": flagged,
                "filename_background_subtype": taxonomy["background_subtype"],
                "review_priority": priority,
                "audit_interpretation": (
                    "possible nearby non-target disturbance intentionally labelled background"
                    if ambiguity
                    else "low model agreement is not proof of label error"
                ),
                "label_action": "retain original label; no authoritative relabel evidence",
            }
        )

    prior_pandera = json.loads(prior_pandera_path.read_text(encoding="utf-8"))
    morphology_metadata_payload = json.loads(morphology_metadata_path.read_text(encoding="utf-8"))
    stored_morphology_hash = str(morphology_metadata_payload.pop("payload_sha256"))
    if stored_morphology_hash != canonical_json_hash(morphology_metadata_payload):
        raise ValueError("Morphology metadata payload hash mismatch")

    _write_csv(output_dir / "filename_taxonomy.csv", taxonomy_rows)
    _write_csv(output_dir / "cleanlab_candidate_review.csv", candidate_rows)
    _write_csv(output_dir / "background_nearby_activity_sessions.csv", nearby)
    output_hashes = {
        name: _sha256(output_dir / name)
        for name in (
            "filename_taxonomy.csv",
            "cleanlab_candidate_review.csv",
            "background_nearby_activity_sessions.csv",
        )
    }
    payload = {
        "schema_version": 1,
        "protocol": "PHI-OTDR v3 complete-cohort ontology and executable contract audit",
        "evidence_status": "retrospective audit; filename semantics remain partially unresolved",
        "dataset_fingerprint_sha256": raw_audit["dataset_fingerprint_sha256"],
        "input_hashes": {
            "session_inventory_sha256": _sha256(session_inventory_path),
            "raw_inventory_sha256": _sha256(raw_inventory_path),
            "raw_audit_sha256": _sha256(raw_audit_path),
            "morphology_bundle_sha256": _sha256(morphology_path),
            "morphology_metadata_payload_sha256": stored_morphology_hash,
            "cleanlab_csv_sha256": _sha256(cleanlab_path),
            "prior_pandera_result_sha256": _sha256(prior_pandera_path),
        },
        "output_hashes": output_hashes,
        "executable_contract": {
            "status": "pass",
            "pandera_version": pandera_version,
            "raw_listed_rows": len(raw_frame),
            "raw_readable_rows": int(readable.sum()),
            "raw_unreadable_rows": int((~readable).sum()),
            "raw_shape": "10000x12",
            "raw_dtype": "uint16",
            "raw_nonfinite_readable_rows": 0,
            "morphology_rows": int(bundle["features"].shape[0]),
            "morphology_features": int(bundle["features"].shape[1]),
            "morphology_all_finite": True,
            "unique_sessions": int(morphology_metadata["session"].nunique()),
            "unique_readable_paths": int(morphology_metadata["rel_path"].nunique()),
            "session_metadata_consistent": True,
            "class_mapping": {str(index): name for index, name in enumerate(CLASS_NAMES)},
        },
        "prior_pandera_scope": {
            "status": prior_pandera["status"],
            "window_rows": prior_pandera["window_rows"],
            "session_rows": prior_pandera["session_rows"],
            "scope_caveat": "The earlier 339-feature Pandera run covered 15,253 feature rows, not the complete 15,418-row v3 morphology bundle. This v3 audit closes the complete-row metadata and finiteness gap.",
        },
        "cleanlab": {
            "sessions_scored": len(cleanlab_rows),
            "flagged_sessions": sum(row["cleanlab_flagged"].lower() == "true" for row in cleanlab_rows),
            "lowest_score": min(float(row["label_quality_score"]) for row in cleanlab_rows),
            "interpretation": "Model-disagreement candidates for human review; not automatic label corrections.",
        },
        "filename_subtype_counts": {
            key: dict(sorted(counter.items())) for key, counter in sorted(subtype_counts.items())
        },
        "background_ontology": {
            "official_background_sessions": sum(row["class_name"] == "background" for row in taxonomy_rows),
            "nearby_activity_sessions": len(nearby),
            "nearby_activity_date_tokens": sorted({row["date_token"] for row in nearby}),
            "nearby_activity_eras": ["april_may"],
            "primary_decision": "retain one official background class but designate it weakly/compositely labelled",
            "exploratory_decision": "evaluate quiet-background sensitivity by excluding the nine filename-identified nearby-activity sessions",
            "decomposition_decision": "do not train a seventh class",
            "decomposition_reason": "all nine candidate nearby-activity sessions come from one date and one era, so a learned seventh class would be perfectly acquisition-confounded and lacks authoritative ground truth",
        },
        "semantics_limits": [
            "Filename tokens are audit metadata only and must not be inference features.",
            "The source token resembles participant initials in early batches but event words in later batches; it is not a verified subject identifier.",
            "Fast/slow, duration, direction, center/edge/fiber, and run/walk tokens are inconsistently available rather than a complete factorial annotation.",
            "No labels are changed by this audit.",
        ],
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "ontology_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-inventory", type=Path, required=True)
    parser.add_argument("--raw-inventory", type=Path, required=True)
    parser.add_argument("--raw-audit", type=Path, required=True)
    parser.add_argument("--morphology", type=Path, required=True)
    parser.add_argument("--morphology-metadata", type=Path, required=True)
    parser.add_argument("--cleanlab", type=Path, required=True)
    parser.add_argument("--prior-pandera", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = audit(
        session_inventory_path=args.session_inventory,
        raw_inventory_path=args.raw_inventory,
        raw_audit_path=args.raw_audit,
        morphology_path=args.morphology,
        morphology_metadata_path=args.morphology_metadata,
        cleanlab_path=args.cleanlab,
        prior_pandera_path=args.prior_pandera,
        output_dir=args.output_dir,
    )
    print(json.dumps({"status": result["executable_contract"]["status"], "payload_sha256": result["payload_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
