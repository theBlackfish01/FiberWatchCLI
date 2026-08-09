from __future__ import annotations

import sys
from pathlib import Path


SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import json

from phi_research.artifact_finalize_v3 import _journal, _vulnerability_audit_summary


def test_journal_distinguishes_evidence_categories_and_stopped_work() -> None:
    rows = _journal()
    categories = {row["category"] for row in rows}
    decisions = {row["decision"] for row in rows}
    assert "smoke" in categories
    assert "retrospective development negative" in categories
    assert "stopped before expensive run" in categories
    assert "stop scaling" in decisions
    assert any(row["stage"] == "confirmatory_status" for row in rows)


def test_vulnerability_summary_deduplicates_and_preserves_skips(tmp_path: Path) -> None:
    path = tmp_path / "audit.json"
    path.write_text(
        json.dumps(
            {
                "dependencies": [
                    {
                        "name": "installer",
                        "version": "1.0",
                        "vulns": [
                            {"id": "ADV-1", "fix_versions": ["1.1"]},
                            {"id": "ADV-1", "fix_versions": ["1.1"]},
                        ],
                    },
                    {"name": "cuda-wheel", "skip_reason": "not on index"},
                ],
                "fixes": [],
            }
        ),
        encoding="utf-8",
    )
    summary = _vulnerability_audit_summary(path)
    assert summary["affected_package_count"] == 1
    assert summary["unique_advisory_count"] == 1
    assert summary["affected_packages"][0]["unique_advisory_ids"] == ["ADV-1"]
    assert summary["skipped_packages"][0]["name"] == "cuda-wheel"
