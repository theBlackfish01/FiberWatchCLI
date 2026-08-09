"""Generate publication-oriented figures from frozen machine-readable results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    directions = ("january_to_april_may", "april_may_to_january")
    labels = ("Jan → Apr/May", "Apr/May → Jan")
    closed = [json.loads((args.root / d / "confirmatory_closed_set" /
                          "confirmatory_closed_set_results.json").read_text(encoding="utf-8")) for d in directions]
    open_results = [json.loads((args.root / d / "confirmatory_open" /
                                "confirmatory_open_results.json").read_text(encoding="utf-8")) for d in directions]
    views = ("amplitude__logistic", "dynamics__logistic", "full__logistic")
    colors = ("#b7bcc5", "#147d92", "#d18b2c")
    x = np.arange(2)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, (view, color) in enumerate(zip(views, colors)):
        values = [row["results"][view]["session_metrics"]["macro_f1"] for row in closed]
        ax.bar(x + (index - 1) * .23, values, .21, label=view.split("__")[0].title(), color=color)
    ax.set_xticks(x, labels); ax.set_ylim(0, 1); ax.set_ylabel("Target-era session macro-F1")
    ax.legend(frameon=False, ncol=3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(args.output_dir / "closed_set_transfer.png", dpi=220); plt.close(fig)

    methods = ("pca24", "aligned_weight_0", "aligned_weight_100")
    method_labels = ("PCA-24", "Encoder, λ=0", "Encoder, λ=100")
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, method in enumerate(methods):
        values = [row["summary"][method]["detection_h"]["mean"] for row in open_results]
        ax.bar(x + (index - 1) * .23, values, .21, label=method_labels[index], color=colors[index])
    ax.set_xticks(x, labels); ax.set_ylim(0, .65); ax.set_ylabel("Unknown-detection H (mean over folds/seeds)")
    ax.legend(frameon=False, ncol=3); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(args.output_dir / "unknown_detection.png", dpi=220); plt.close(fig)

    stats = json.loads((args.root / "confirmatory_statistics.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    shots = np.asarray([1, 3, 5])
    for method, label, color in zip(methods, method_labels, colors):
        medoid = [stats["estimates"][method][f"{shot}shot_medoid"]["mean"] for shot in shots]
        random = [stats["estimates"][method][f"{shot}shot_random"]["mean"] for shot in shots]
        ax.plot(shots, medoid, marker="o", color=color, label=f"{label}, medoid")
        ax.plot(shots, random, marker="x", linestyle="--", color=color, alpha=.7, label=f"{label}, random")
    ax.set_xticks(shots); ax.set_ylim(.3, .6); ax.set_xlabel("Enrollment shots")
    ax.set_ylabel("Enrollment H (12-cluster mean)"); ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(args.output_dir / "active_enrollment.png", dpi=220); plt.close(fig)


if __name__ == "__main__":
    main()
