"""Utility for exploratory analysis of OTDR datasets.

This module provides a reusable entry point for inspecting tabular OTDR
measurements. It focuses on helping engineers understand feature
relationships, feature distributions, and the overall structure of a dataset
that mirrors the schema used by the machine-learning models in this project.

The main workflow is encapsulated in :func:`analyze_dataset`.  It loads a CSV
file into a pandas ``DataFrame`` and produces:

* High-level metadata (row/column counts, numerical feature summaries).
* Missing-value audits and class-distribution summaries.
* Correlation matrices and top correlated feature pairs.
* Visualization artifacts that are saved to disk for offline review.

All heavy lifting is intentionally tucked away in separate helper functions so
that the analysis can easily be reused from notebooks or other scripts.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Default plotting style tweaks for consistency.
sns.set_theme(style="whitegrid")


@dataclass
class DatasetSummary:
    """Container for high-level dataset information."""

    n_rows: int
    n_columns: int
    numerical_columns: List[str]
    categorical_columns: List[str]
    missing_counts: pd.Series
    class_distribution: Optional[pd.Series]
    describe_table: pd.DataFrame
    correlation_matrix_path: Optional[Path]
    saved_figures: List[Path]
    top_correlated_pairs: List[Tuple[str, str, float]]

    def to_dict(self) -> dict:
        """Serialise the summary to a plain Python dictionary."""

        return {
            "n_rows": self.n_rows,
            "n_columns": self.n_columns,
            "numerical_columns": self.numerical_columns,
            "categorical_columns": self.categorical_columns,
            "missing_counts": self.missing_counts.fillna(0).astype(int).to_dict(),
            "class_distribution": None
            if self.class_distribution is None
            else self.class_distribution.to_dict(),
            "describe_table": self.describe_table.to_dict(),
            "correlation_matrix_path": str(self.correlation_matrix_path)
            if self.correlation_matrix_path
            else None,
            "saved_figures": [str(path) for path in self.saved_figures],
            "top_correlated_pairs": [
                {"feature_a": a, "feature_b": b, "correlation": corr}
                for a, b, corr in self.top_correlated_pairs
            ],
        }

    def pretty_print(self) -> str:
        """Return a human-readable multi-line representation."""

        header = textwrap.dedent(
            f"""
            Dataset summary
            ----------------
            Rows: {self.n_rows}
            Columns: {self.n_columns}
            Numerical columns ({len(self.numerical_columns)}): {', '.join(self.numerical_columns)}
            Categorical columns ({len(self.categorical_columns)}): {', '.join(self.categorical_columns)}
            """
        ).strip()

        missing = "Missing values (per column):\n" + self.missing_counts.to_string()
        class_dist = (
            "Class distribution:\n" + self.class_distribution.to_string()
            if self.class_distribution is not None
            else "Class distribution: <not computed>"
        )
        describe = "\nDescriptive statistics:\n" + self.describe_table.to_string()

        figures = "Saved figures:\n" + "\n".join(f" - {path}" for path in self.saved_figures)
        if not self.saved_figures:
            figures = "Saved figures: <none>"

        corr_lines = [
            "Top correlated feature pairs:" if self.top_correlated_pairs else "No correlation pairs computed"
        ]
        for feature_a, feature_b, corr_value in self.top_correlated_pairs:
            corr_lines.append(f" - {feature_a} vs {feature_b}: {corr_value:.3f}")
        corr_lines.append(
            f"Correlation heatmap saved to: {self.correlation_matrix_path}"
            if self.correlation_matrix_path
            else "Correlation heatmap not generated"
        )
        corr = "\n".join(corr_lines)

        return "\n\n".join([header, missing, class_dist, describe, corr, figures])


def _identify_column_types(df: pd.DataFrame) -> tuple[List[str], List[str]]:
    """Separate the columns into numerical and categorical groups."""

    numerical_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [col for col in df.columns if col not in numerical_columns]
    return numerical_columns, categorical_columns


def _save_histograms(df: pd.DataFrame, output_dir: Path, columns: Iterable[str]) -> List[Path]:
    """Generate histograms for the provided columns."""

    saved_paths: List[Path] = []
    for column in columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[column].dropna(), kde=True, bins=30)
        plt.title(f"Distribution of {column}")
        plt.tight_layout()
        path = output_dir / f"hist_{column}.png"
        plt.savefig(path)
        plt.close()
        saved_paths.append(path)
    return saved_paths


def _save_boxplots_by_class(
    df: pd.DataFrame, output_dir: Path, numerical_columns: Iterable[str], class_column: str
) -> List[Path]:
    """Create boxplots grouped by the specified class column."""

    saved_paths: List[Path] = []
    if class_column not in df.columns:
        return saved_paths

    selected_columns = [col for col in numerical_columns if col != class_column]
    for column in selected_columns[:10]:  # prevent figure explosion on wide datasets
        plt.figure(figsize=(8, 4))
        sns.boxplot(data=df, x=class_column, y=column)
        plt.title(f"Distribution of {column} by {class_column}")
        plt.tight_layout()
        path = output_dir / f"box_{column}_by_{class_column}.png"
        plt.savefig(path)
        plt.close()
        saved_paths.append(path)
    return saved_paths


def _save_scatter(
    df: pd.DataFrame,
    output_dir: Path,
    x_col: str,
    y_col: str,
    hue_column: Optional[str] = None,
) -> Optional[Path]:
    """Save a scatter plot for two selected columns if present."""

    if x_col not in df.columns or y_col not in df.columns:
        return None

    hue = hue_column if hue_column in df.columns else None

    plt.figure(figsize=(6, 6))
    scatter_kwargs = {"data": df, "x": x_col, "y": y_col}
    if hue is not None:
        scatter_kwargs.update({"hue": hue, "palette": "viridis"})
    sns.scatterplot(**scatter_kwargs)
    plt.title(f"Scatter plot of {y_col} vs {x_col}")
    plt.tight_layout()
    path = output_dir / f"scatter_{y_col}_vs_{x_col}.png"
    plt.savefig(path)
    plt.close()
    return path


def _save_correlation_heatmap(df: pd.DataFrame, output_dir: Path, columns: Iterable[str]) -> Optional[Path]:
    """Generate and save a correlation heatmap for the dataset."""

    numerical_df = df[list(columns)].dropna(axis=1, how="all")
    if numerical_df.empty:
        return None

    corr = numerical_df.corr()
    plt.figure(figsize=(min(16, 0.5 * corr.shape[0] + 6), min(12, 0.5 * corr.shape[1] + 6)))
    sns.heatmap(corr, cmap="coolwarm", annot=False, fmt=".2f")
    plt.title("Feature correlation heatmap")
    plt.tight_layout()
    path = output_dir / "correlation_heatmap.png"
    plt.savefig(path)
    plt.close()
    return path


def analyze_dataset(
    csv_path: Path | str,
    output_dir: Path | str = "analysis_outputs",
    class_column: str = "Class",
    summary_json: bool = True,
) -> DatasetSummary:
    """Run exploratory data analysis and export visualisations.

    Parameters
    ----------
    csv_path:
        Path to the dataset CSV file.
    output_dir:
        Directory where figures and reports will be written. Created if missing.
    class_column:
        Column that denotes the categorical class label; used for grouped stats
        and plots. If absent, those analyses are skipped gracefully.
    summary_json:
        Whether to export a machine-readable summary JSON file alongside the
        figures for quick inspection.
    """

    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    numerical_columns, categorical_columns = _identify_column_types(df)

    missing_counts = df.isna().sum().sort_values(ascending=False)
    describe_table = df[numerical_columns].describe().transpose()

    class_distribution = None
    if class_column in df.columns:
        class_distribution = df[class_column].value_counts().sort_index()

    saved_figures: List[Path] = []
    saved_figures.extend(_save_histograms(df, output_dir, numerical_columns[:8]))
    saved_figures.extend(
        _save_boxplots_by_class(df, output_dir, numerical_columns, class_column)
    )

    scatter_path = _save_scatter(
        df,
        output_dir,
        x_col="Position",
        y_col="loss",
        hue_column=class_column,
    )
    if scatter_path:
        saved_figures.append(scatter_path)

    correlation_matrix_path = _save_correlation_heatmap(df, output_dir, numerical_columns)
    top_correlated_pairs: List[Tuple[str, str, float]] = []
    if numerical_columns:
        corr_matrix = df[numerical_columns].corr().abs()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        upper_triangle = corr_matrix.where(mask)
        stacked = upper_triangle.stack().sort_values(ascending=False)
        for (feature_a, feature_b), corr_value in stacked.head(10).items():
            top_correlated_pairs.append((feature_a, feature_b, float(corr_value)))
    if correlation_matrix_path:
        saved_figures.append(correlation_matrix_path)

    summary = DatasetSummary(
        n_rows=df.shape[0],
        n_columns=df.shape[1],
        numerical_columns=numerical_columns,
        categorical_columns=categorical_columns,
        missing_counts=missing_counts,
        class_distribution=class_distribution,
        describe_table=describe_table,
        correlation_matrix_path=correlation_matrix_path,
        saved_figures=saved_figures,
        top_correlated_pairs=top_correlated_pairs,
    )

    if summary_json:
        json_path = output_dir / "dataset_summary.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(summary.to_dict(), f, indent=2)

    return summary


if __name__ == "__main__":  # pragma: no cover - manual execution entry point
    import argparse

    parser = argparse.ArgumentParser(description="Analyze an OTDR dataset")
    parser.add_argument("csv_path", type=Path, help="Path to the OTDR dataset CSV file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/analysis_outputs"),
        help="Directory where analysis artifacts will be saved",
    )
    parser.add_argument(
        "--class-column",
        type=str,
        default="Class",
        help="Name of the class/label column for grouped analysis",
    )
    parser.add_argument(
        "--no-summary-json",
        action="store_true",
        help="Do not export the dataset summary as JSON",
    )

    args = parser.parse_args()

    summary = analyze_dataset(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        class_column=args.class_column,
        summary_json=not args.no_summary_json,
    )

    print(summary.pretty_print())
