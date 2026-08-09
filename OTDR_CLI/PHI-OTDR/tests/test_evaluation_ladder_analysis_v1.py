from __future__ import annotations

from phi_research.evaluation_ladder_analysis_v1 import LEVELS, summary_rows


def _metrics(value: float) -> dict[str, float]:
    return {
        "macro_f1_six_classes": value,
        "balanced_accuracy_observed_classes": value,
        "worst_observed_class_recall": value,
        "ece_10": 1.0 - value,
        "negative_log_likelihood": 1.0,
    }


def test_summary_rows_normalize_all_ladder_levels() -> None:
    representation = "test_view"
    repeated = {
        metric: {"mean": value, "std": 0.01, "minimum": value, "maximum": value}
        for metric, value in {
            "accuracy": 0.9,
            "macro_f1_six_classes": 0.8,
            "balanced_accuracy_observed_classes": 0.7,
            "worst_observed_class_recall": 0.6,
            "negative_log_likelihood": 0.5,
            "ece_10": 0.1,
        }.items()
    }
    result = {
        "representations_run": [representation],
        "summaries": {
            f"{representation}::random_window": {
                "fold_count": 3,
                "session_metrics": repeated,
                "mean_test_session_overlap_fraction": 1.0,
            },
            f"{representation}::random_session": {
                "fold_count": 3,
                "session_metrics": repeated,
                "mean_test_session_overlap_fraction": 0.0,
            },
            f"{representation}::date_class_cell": {
                "fold_count": 2,
                "pooled_session_metrics": _metrics(0.75),
            },
            f"{representation}::leave_one_date_out": {
                "fold_count": 2,
                "pooled_session_metrics": _metrics(0.65),
            },
            f"{representation}::cross_era": {
                "january_to_april_may": _metrics(0.45),
                "april_may_to_january": _metrics(0.55),
            },
        },
    }
    rows = summary_rows(result)
    assert [row["level"] for row in rows] == [level for level, _ in LEVELS]
    assert rows[0]["session_macro_f1"] == 0.8
    assert rows[0]["mean_test_session_overlap_fraction"] == 1.0
    assert rows[-1]["session_macro_f1"] == 0.55
