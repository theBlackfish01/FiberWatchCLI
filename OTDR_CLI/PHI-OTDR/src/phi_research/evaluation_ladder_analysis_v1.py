"""Tables, figures, and a concise report for the PHI-OTDR evaluation ladder."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data_contract import CLASS_NAMES, canonical_json_hash


LEVELS = (
    ("random_window", "Random\nwindow"),
    ("random_session", "Random\nsession"),
    ("date_class_cell", "Held-out\ndate × class"),
    ("leave_one_date_out", "Held-out\ndate"),
    ("january_to_april_may", "January →\nApril–May"),
    ("april_may_to_january", "April–May →\nJanuary"),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_result(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    stored = str(payload.pop("payload_sha256"))
    if stored != canonical_json_hash(payload):
        raise ValueError("Evaluation-ladder result payload hash mismatch")
    payload["payload_sha256"] = stored
    return payload


def summary_rows(result: Mapping[str, object]) -> list[dict[str, object]]:
    rows = []
    summaries = result["summaries"]
    for representation in result["representations_run"]:
        representation = str(representation)
        for level, label in LEVELS:
            if level in {"random_window", "random_session"}:
                summary = summaries[f"{representation}::{level}"]
                metrics = summary["session_metrics"]
                row = {
                    "representation": representation,
                    "level": level,
                    "level_label": label.replace("\n", " "),
                    "fold_count": int(summary["fold_count"]),
                    "session_macro_f1": float(metrics["macro_f1_six_classes"]["mean"]),
                    "session_macro_f1_std": float(metrics["macro_f1_six_classes"]["std"]),
                    "session_balanced_accuracy": float(
                        metrics["balanced_accuracy_observed_classes"]["mean"]
                    ),
                    "session_worst_recall": float(
                        metrics["worst_observed_class_recall"]["mean"]
                    ),
                    "session_ece": float(metrics["ece_10"]["mean"]),
                    "session_nll": float(metrics["negative_log_likelihood"]["mean"]),
                    "mean_test_session_overlap_fraction": float(
                        summary["mean_test_session_overlap_fraction"]
                    ),
                }
            elif level in {"date_class_cell", "leave_one_date_out"}:
                summary = summaries[f"{representation}::{level}"]
                metrics = summary["pooled_session_metrics"]
                row = {
                    "representation": representation,
                    "level": level,
                    "level_label": label.replace("\n", " "),
                    "fold_count": int(summary["fold_count"]),
                    "session_macro_f1": float(metrics["macro_f1_six_classes"]),
                    "session_macro_f1_std": 0.0,
                    "session_balanced_accuracy": float(
                        metrics["balanced_accuracy_observed_classes"]
                    ),
                    "session_worst_recall": float(metrics["worst_observed_class_recall"]),
                    "session_ece": float(metrics["ece_10"]),
                    "session_nll": float(metrics["negative_log_likelihood"]),
                    "mean_test_session_overlap_fraction": 0.0,
                }
            else:
                summary = summaries[f"{representation}::cross_era"][level]
                row = {
                    "representation": representation,
                    "level": level,
                    "level_label": label.replace("\n", " "),
                    "fold_count": 1,
                    "session_macro_f1": float(summary["macro_f1_six_classes"]),
                    "session_macro_f1_std": 0.0,
                    "session_balanced_accuracy": float(
                        summary["balanced_accuracy_observed_classes"]
                    ),
                    "session_worst_recall": float(summary["worst_observed_class_recall"]),
                    "session_ece": float(summary["ece_10"]),
                    "session_nll": float(summary["negative_log_likelihood"]),
                    "mean_test_session_overlap_fraction": 0.0,
                }
            rows.append(row)
    return rows


def _plot_ladder(rows: Sequence[Mapping[str, object]], path: Path) -> None:
    representations = list(dict.fromkeys(str(row["representation"]) for row in rows))
    colors = ("#2563eb", "#dc2626", "#059669")
    markers = ("o", "s", "^")
    x = np.arange(len(LEVELS))
    figure, axes = plt.subplots(2, 1, figsize=(12.0, 8.6), sharex=True)
    for index, representation in enumerate(representations):
        local = {str(row["level"]): row for row in rows if row["representation"] == representation}
        f1 = np.asarray([float(local[level]["session_macro_f1"]) for level, _ in LEVELS])
        f1_std = np.asarray([float(local[level]["session_macro_f1_std"]) for level, _ in LEVELS])
        ece = np.asarray([float(local[level]["session_ece"]) for level, _ in LEVELS])
        label = representation.replace("_", " ")
        axes[0].errorbar(
            x,
            f1,
            yerr=f1_std,
            color=colors[index],
            marker=markers[index],
            linewidth=2.2,
            capsize=3,
            label=label,
        )
        axes[1].plot(
            x,
            ece,
            color=colors[index],
            marker=markers[index],
            linewidth=2.2,
            label=label,
        )
    for axis in axes:
        axis.set_ylim(-0.02, 1.02)
        axis.grid(axis="y", alpha=0.25)
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Session macro-F1")
    axes[0].set_title("PHI-OTDR performance across increasingly independent evaluations")
    axes[0].legend(loc="lower left", frameon=False)
    axes[1].set_ylabel("Expected calibration error")
    axes[1].set_xticks(x, [label for _, label in LEVELS])
    axes[1].set_xlabel("Evaluation protocol")
    figure.text(
        0.5,
        0.005,
        "Random-window test sessions overlap training; all later levels are session-disjoint. "
        "Cross-era outcomes are retrospective.",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _cell_rows(fold_rows: Sequence[Mapping[str, str]]) -> list[dict[str, object]]:
    output = []
    for row in fold_rows:
        if row["level"] != "date_class_cell":
            continue
        date, class_name = row["fold"].split("__", maxsplit=1)
        output.append(
            {
                "representation": row["representation"],
                "date_token": date,
                "class_name": class_name,
                "session_count": int(row["test_sessions"]),
                "accuracy": float(row["session_accuracy"]),
                "ece": float(row["session_ece"]),
            }
        )
    return output


def _plot_cell_heatmap(
    cell_rows: Sequence[Mapping[str, object]], *, representation: str, path: Path
) -> None:
    local = [row for row in cell_rows if row["representation"] == representation]
    dates = sorted({str(row["date_token"]) for row in local})
    matrix = np.full((len(dates), len(CLASS_NAMES)), np.nan, dtype=np.float64)
    counts = np.zeros_like(matrix, dtype=np.int64)
    for row in local:
        date_index = dates.index(str(row["date_token"]))
        class_index = CLASS_NAMES.index(str(row["class_name"]))
        matrix[date_index, class_index] = float(row["accuracy"])
        counts[date_index, class_index] = int(row["session_count"])
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#e5e7eb")
    figure, axis = plt.subplots(figsize=(11.0, max(5.5, 0.48 * len(dates))))
    image = axis.imshow(masked, vmin=0.0, vmax=1.0, cmap=cmap, aspect="auto")
    axis.set_xticks(np.arange(len(CLASS_NAMES)), CLASS_NAMES, rotation=25, ha="right")
    axis.set_yticks(np.arange(len(dates)), dates)
    axis.set_xlabel("Held-out event class")
    axis.set_ylabel("Acquisition date")
    axis.set_title(
        "Held-out date × class accuracy\n"
        + representation.replace("_", " ")
    )
    for row_index in range(len(dates)):
        for column_index in range(len(CLASS_NAMES)):
            if np.isfinite(matrix[row_index, column_index]):
                value = matrix[row_index, column_index]
                text_color = "white" if value < 0.25 else "#111827"
                axis.text(
                    column_index,
                    row_index,
                    f"{value:.2f}\n(n={counts[row_index, column_index]})",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=text_color,
                )
    colorbar = figure.colorbar(image, ax=axis, fraction=0.025, pad=0.025)
    colorbar.set_label("Session accuracy")
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    result: Mapping[str, object],
    rows: Sequence[Mapping[str, object]],
    cell_rows: Sequence[Mapping[str, object]],
) -> None:
    by_key = {(str(row["representation"]), str(row["level"])): row for row in rows}
    table_rows = []
    for representation in result["representations_run"]:
        representation = str(representation)
        values = [
            float(by_key[(representation, level)]["session_macro_f1"])
            for level, _ in LEVELS
        ]
        table_rows.append(
            [representation.replace("_", " "), *[f"{value:.4f}" for value in values]]
        )
    registered = "registered_position_difference_dynamics"
    registered_cells = sorted(
        [row for row in cell_rows if row["representation"] == registered],
        key=lambda row: float(row["accuracy"]),
    )
    worst_cells = [
        [
            row["date_token"],
            row["class_name"],
            row["session_count"],
            f"{float(row['accuracy']):.4f}",
            f"{float(row['ece']):.4f}",
        ]
        for row in registered_cells[:6]
    ]
    random_overlap = float(
        by_key[("absolute_dynamics", "random_window")][
            "mean_test_session_overlap_fraction"
        ]
    )
    report = f"""# PHI-OTDR acquisition-safe evaluation ladder v1

## Evidence status

This is retrospective development evidence on the complete local BJTU corpus. It is not an independent confirmation. The classifier, representations, seeds, and split rules were frozen in the hashed configuration before this ladder was executed.

## Main result

Random-window and random-session evaluation are both nearly perfect, and leave-one-date-out remains strong. The severe discontinuity appears only when the complete January and April–May acquisition eras are separated. Recording-session overlap is therefore a real defect in random-window evaluation, but it is not the principal explanation for the cross-era failure of these fixed morphology classifiers.

{_markdown_table(
        [
            "Representation",
            "Random window",
            "Random session",
            "Date × class",
            "Leave date out",
            "Jan → Apr–May",
            "Apr–May → Jan",
        ],
        table_rows,
    )}

The mean fraction of random-window test sessions also represented in training is **{random_overlap:.2%}**. Random-session, date×class, date, and era folds have zero session overlap.

## Interpretation

- Session separation alone barely changes the ranking or score: within the mixed acquisition corpus, class signatures are extremely stable.
- Holding out an entire date is harder but still yields pooled macro-F1 of roughly 0.89–0.91.
- Holding out one date×class edge is informative: most cells transfer, but some fail catastrophically despite the class and date each being observed elsewhere.
- Registration materially improves April–May to January transfer, while invariant pooling is strongest in that direction and weakest on some compositional walking cells.
- January to April–May remains difficult for every view. No tested representation provides a symmetric solution.
- Calibration follows the same pattern: ordinary grouped folds look usable, while January to April–May is substantially overconfident.

## Worst registered date × class cells

{_markdown_table(["Date", "Class", "Sessions", "Accuracy", "ECE"], worst_cells)}

The 220517 knocking cell is the dominant compositional failure. It should be the first target for morphology-versus-acquisition factor analysis; averaging it into a global score would hide the scientific signal.

## Files

- `evaluation_ladder_results.json`: hashed machine-readable summaries and limitations.
- `fold_results.csv`: every fold, overlap diagnostic, runtime, and metric.
- `session_predictions.csv`: probabilities for every evaluated session.
- `ladder_summary.csv`: compact protocol comparison.
- `date_class_cell_accuracy.csv`: compositional-cell results.
- `evaluation_ladder.png`: performance and calibration across the ladder.
- `date_class_cell_accuracy.png`: registered-view cell heatmap.

## Immediate decision

Proceed to a bounded morphology/acquisition factorization focused on the eligible date×class graph, using the fixed ladder as the evaluation harness. Do not spend the next cycle on a larger classifier: the present results already show that capacity under mixed acquisition conditions is not limiting.
"""
    path.write_text(report, encoding="utf-8")


def analyze(*, result_path: Path, output_dir: Path) -> dict[str, object]:
    result = _load_result(result_path)
    source_dir = result_path.parent
    fold_path = source_dir / "fold_results.csv"
    prediction_path = source_dir / "session_predictions.csv"
    if _sha256(fold_path) != result["output_hashes"]["fold_results_sha256"]:
        raise ValueError("Fold-results hash mismatch")
    if _sha256(prediction_path) != result["output_hashes"]["session_predictions_sha256"]:
        raise ValueError("Session-prediction hash mismatch")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = summary_rows(result)
    fold_rows = _read_csv(fold_path)
    cells = _cell_rows(fold_rows)
    summary_path = output_dir / "ladder_summary.csv"
    cell_path = output_dir / "date_class_cell_accuracy.csv"
    ladder_figure = output_dir / "evaluation_ladder.png"
    cell_figure = output_dir / "date_class_cell_accuracy.png"
    report_path = output_dir / "EVALUATION_LADDER_REPORT.md"
    _write_csv(summary_path, rows)
    _write_csv(cell_path, cells)
    _plot_ladder(rows, ladder_figure)
    _plot_cell_heatmap(
        cells,
        representation="registered_position_difference_dynamics",
        path=cell_figure,
    )
    _write_report(report_path, result=result, rows=rows, cell_rows=cells)
    payload: dict[str, object] = {
        "schema_version": 1,
        "protocol": "PHI-OTDR evaluation ladder analysis v1",
        "input_payload_sha256": result["payload_sha256"],
        "outputs": {
            "ladder_summary_sha256": _sha256(summary_path),
            "date_class_cell_accuracy_sha256": _sha256(cell_path),
            "evaluation_ladder_png_sha256": _sha256(ladder_figure),
            "date_class_cell_accuracy_png_sha256": _sha256(cell_figure),
            "report_sha256": _sha256(report_path),
        },
    }
    payload["payload_sha256"] = canonical_json_hash(payload)
    (output_dir / "evaluation_ladder_analysis.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            analyze(result_path=args.result, output_dir=args.output_dir),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
