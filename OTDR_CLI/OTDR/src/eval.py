"""
Evaluation script

Two modes:

1. Pipeline – GRU‑AE anomaly detection ➜ selected samples → classifier (TCN)
2. Direct – classifier directly on the full test set

LLM explanation of random samples using vision‑capable GPT‑5 with RAG and XAI (SHAP or LIME)
"""

from __future__ import annotations
import base64
import os
import click
import json
from time import perf_counter
from typing import Any, List, Sequence, Tuple
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import shap
from lime.lime_tabular import LimeTabularExplainer
import torch
import wandb
from model_functions.siamese import SiameseNetwork
from model_functions.tcn import OTDR_TCN
from sklearn.metrics import (
    accuracy_score,
    root_mean_squared_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
    mean_absolute_error
)
from data_helper import (
    load_raw_dataframe,
    make_splits,
    tensorise_splits,
    measurement_columns,
    summarise_feature_layout,
    build_feature_config,
    summarise_feature_layout
)
from model_functions.gruae import VectorGRUAE, reconstruction_error
from model_functions.tcn import predict as predict_tcn
from model_functions.tcn_binary import predict as predict_tcn_binary
from model_functions.tst import predict as predict_tst
from pipeline import (
    load_classifier,
    load_classifier_meta,
    remap_anomaly_only_targets,
    run_cascade,
)
import config.config as cfg
from pathlib import Path
from rag import build_reference_block, retrieve
from openai import OpenAI
import warnings
import matplotlib.patheffects as path_effects


warnings.filterwarnings("ignore", category=FutureWarning)  # noqa: T201

client = OpenAI(api_key=cfg.OPENAI_API_KEY)

LLM_SAMPLE_TARGET = 5


def _init_wandb_run(config: dict[str, Any]) -> wandb.sdk.wandb_run.Run | None:
    """Initialise a Weights & Biases run when the API key is available."""

    api_key = cfg.WANDB_API_KEY
    if not api_key:
        print("[WARN] WANDB_API_KEY not set – skipping WANDB logging.")
        return None

    os.environ.setdefault("WANDB_API_KEY", api_key)
    try:
        wandb.login(key=api_key, relogin=True, force=True)
        return wandb.init(project="OTDR_Eval", config=config, reinit=True)
    except Exception as exc:  # pragma: no cover - WANDB optional
        print(f"[WARN] Unable to initialise WANDB logging: {exc}")
        return None


def _dedupe_preserve(seq):
    return tuple(dict.fromkeys(seq))


def _validate_feature_config(
    meta_cfg: dict[str, Any],
    expected_cfg: dict[str, Any],
    context: str,
) -> None:
    """Ensure metadata feature configuration aligns with the requested one."""

    expected_columns = list(expected_cfg.get("columns", []))
    expected_use_lr = bool(expected_cfg.get("use_loss_reflectance", False))
    expected_requested = list(expected_cfg.get("requested_extras", []))

    meta_sig = meta_cfg.get("signature")
    expected_sig = expected_cfg.get("signature")
    if meta_sig and expected_sig and meta_sig == expected_sig:
        return

    meta_columns = [str(c) for c in meta_cfg.get("columns") or []]
    if meta_columns and meta_columns != expected_columns:
        raise ValueError(
            f"{context} expects measurement columns {meta_columns}, "
            f"but the requested configuration is {expected_columns}."
        )

    meta_use_lr = meta_cfg.get("use_loss_reflectance")
    if meta_use_lr is not None and bool(meta_use_lr) != expected_use_lr:
        raise ValueError(
            f"{context} was trained with use_loss_reflectance={bool(meta_use_lr)}, "
            f"but the flag is set to {expected_use_lr}."
        )

    meta_requested = meta_cfg.get("requested_extras")
    if meta_requested is not None:
        meta_requested_list = [str(c) for c in meta_requested]
        if meta_requested_list != expected_requested:
            raise ValueError(
                f"{context} expects additional features {meta_requested_list}, "
                f"but {expected_requested} were requested."
            )


def _validate_metadata_features(
    meta: dict[str, Any] | None,
    expected_cfg: dict[str, Any],
    context: str,
) -> None:
    """Compare metadata payloads against the requested features."""

    if not meta:
        return

    cfg = meta.get("feature_config")
    if cfg:
        _validate_feature_config(cfg, expected_cfg, context)
        return

    expected_columns = list(expected_cfg.get("columns", []))
    expected_use_lr = bool(expected_cfg.get("use_loss_reflectance", False))

    meta_columns = meta.get("active_features") or meta.get("feature_names")
    if meta_columns and list(meta_columns) != expected_columns:
        raise ValueError(
            f"{context} expects measurement columns {list(meta_columns)}, "
            f"but the requested configuration is {expected_columns}."
        )

    meta_use_lr = meta.get("use_loss_reflectance")
    if meta_use_lr is not None and bool(meta_use_lr) != expected_use_lr:
        raise ValueError(
            f"{context} was trained with use_loss_reflectance={bool(meta_use_lr)}, "
            f"but the flag is set to {expected_use_lr}."
        )


def _extract_response_text(resp: Any) -> str:
    """Best-effort extraction of text content from the Responses API result."""

    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_obj = getattr(content, "text", None)
            if isinstance(text_obj, str):
                texts.append(text_obj)
            else:
                value = getattr(text_obj, "value", None)
                if isinstance(value, str):
                    texts.append(value)

    if texts:
        return "\n".join(t.strip() for t in texts if t.strip())

    raise RuntimeError("No textual content returned by the Responses API")


def _call_responses_api(
        llm_client: OpenAI,
        model: str,
        system_text: str,
        user_content: list[dict[str, Any]],
) -> str:
    """Invoke the Responses API for multimodal prompts."""

    resp = llm_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ],
    )
    return _extract_response_text(resp)

# --------------------------------------------------
# Utility helpers
# --------------------------------------------------
def _load_gru_ae(det_path: Path, device: torch.device) -> Tuple[VectorGRUAE, float, np.ndarray, np.ndarray]:
    """Return (model, threshold, scaler_mean, scaler_scale)."""
    meta_path = det_path.with_suffix(".json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Expected {meta_path} alongside weights")
    meta = json.loads(Path(meta_path).read_text())

    # reconstruct scaler params (used only for plots in this script)
    if "scaler_mean" in meta and "scaler_scale" in meta:
        scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32)
        scaler_scale = np.array(meta["scaler_scale"], dtype=np.float32)
    elif "mean" in meta and "scale" in meta:
        # backwards compatibility with early metadata files
        scaler_mean = np.array(meta["mean"], dtype=np.float32)
        scaler_scale = np.array(meta["scale"], dtype=np.float32)
    else:
        raise KeyError("GRU-AE metadata must contain scaler_mean/scale entries")
    threshold = float(meta["threshold"])

    # feature dim = scaler_mean length
    feat_dim = scaler_mean.size
    ae = VectorGRUAE(feat_dim=feat_dim)
    ae.load_state_dict(torch.load(det_path, map_location=device))
    ae.eval().to(device)
    return ae, threshold, scaler_mean, scaler_scale


def _visualise_sample(
        classifier: str,
        amps: np.ndarray,
        snr: float,
        *,
        loss_db: float | None = None,
        reflectance_db: float | None = None,
        true_cls: int,
        pred_cls: int | None,
        true_pos: float,
        pred_pos: float | None,
        idx: int,
        out_dir: Path,
        classifier_label: str | None = None,
        localisation_label: str | None = None,
):
    classifier = classifier_label.upper() if classifier_label else classifier.upper()
    loc_model = localisation_label.upper() if localisation_label else None
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(amps.size), amps, label="Amplitude")
    pred_label = "N/A" if pred_cls is None else str(pred_cls)
    pred_pos_str = "N/A" if pred_pos is None else f"{pred_pos:.3f}"
    loss_str = "N/A" if loss_db is None else f"{loss_db:.2f} dB"
    refl_str = "N/A" if reflectance_db is None else f"{reflectance_db:.2f} dB"
    model_title = f"Model - {classifier}"
    if loc_model:
        model_title += f" / {loc_model}"
    plt.title(
        f"Sample #{idx} | TrueC={true_cls} PredC={pred_label} | "
        f"TruePos={true_pos:.3f}  PredPos={pred_pos_str} | SNR={snr:.2f} dB | "
        f"{model_title}"
    )
    plt.gca().text(
        0.02,
        0.98,
        f"SNR: {snr:.2f} dB\nLoss: {loss_str}\nReflectance: {refl_str}",
        transform=plt.gca().transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#f0f0f0", edgecolor="gray", alpha=0.9),
    )
    plt.xlabel("P-index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    fname = out_dir / f"sample_{idx}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def _plot_radial(
        classifier: str,
        y_true: np.ndarray | None,
        y_pred: np.ndarray | None,
        class_ids: List[int],
        out_dir: Path,
        *,
        y_pos_true: np.ndarray | None = None,
        y_pos_pred: np.ndarray | None = None,
        include_loss_reflectance: bool = False,
) -> dict[str, Path]:
    """
    Create multiple per-class diagnostic plots:
    - Radial polar bar chart of per-class accuracy with enhanced styling.
    - Bar chart of per-class accuracy (Cartesian) with color-coded performance.
    - Per-class localisation error (RMSE) when localisation is available.
    - Radial plot of localisation errors per class.
    - Combined accuracy vs support scatter plot.

    Returns a dict mapping plot name -> saved Path.
    """
    classifier = classifier.upper()
    out_dir.mkdir(exist_ok=True)
    artifacts: dict[str, Path] = {}

    classification_available = (
        y_true is not None
        and y_pred is not None
        and y_true.size > 0
        and y_pred.size > 0
    )

    has_classes = bool(class_ids)

    if not has_classes and not (y_pos_true is not None and y_pos_pred is not None):
        return artifacts

    y_true = np.asarray(y_true) if y_true is not None else None
    y_pred = np.asarray(y_pred) if y_pred is not None else None

    has_loc = y_pos_true is not None and y_pos_pred is not None
    if has_loc:
        y_pos_true = np.asarray(y_pos_true, dtype=np.float32)
        y_pos_pred = np.asarray(y_pos_pred, dtype=np.float32)
        # If shapes do not agree with y_true, disable localisation plots
        if (
                y_pos_true.shape[0] != y_true.shape[0]
                or y_pos_pred.shape[0] != y_true.shape[0]
        ):
            has_loc = False

    angles = np.linspace(0, 2 * np.pi, len(class_ids), endpoint=False) if has_classes else np.array([])
    width = (2 * np.pi) / max(len(class_ids), 1) if has_classes else 0.0

    accuracies: list[float] = []
    supports: list[int] = []
    rmse_errors: list[float] = []

    suffix = f"_{classifier.lower()}"
    if include_loss_reflectance:
        suffix += "_lr"

    for cls in class_ids:
        mask = (y_true == cls) if y_true is not None else np.array([], dtype=bool)
        supports.append(int(mask.sum()))
        if mask.size and mask.any():
            if classification_available:
                acc = float((y_pred[mask] == cls).mean())
                accuracies.append(acc)
            else:
                accuracies.append(np.nan)
            if has_loc:
                err = y_pos_pred[mask] - y_pos_true[mask]
                rmse = float(np.sqrt(np.mean(np.square(err))))
                rmse_errors.append(rmse)
            else:
                rmse_errors.append(np.nan)
        else:
            accuracies.append(np.nan if not classification_available else 0.0)
            rmse_errors.append(np.nan)

    # ---------- Enhanced Radial Accuracy Plot ----------
    if classification_available and has_classes:
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(12, 12))

        # Use a perceptually uniform colormap
        colors = plt.cm.RdYlGn(np.clip(np.nan_to_num(accuracies, nan=0.0), 0, 1))

        bars = ax.bar(
            angles,
            accuracies,
            width=width * 0.85,
            bottom=0.0,
            color=colors,
            alpha=0.9,
            edgecolor='white',
            linewidth=1.6,
        )

        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_ylim(0, 1.0)

        # Remove degree markers / theta tick labels
        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=10, fontweight='bold')
        ax.tick_params(axis='y', pad=6)
        ax.grid(True, alpha=0.3, linestyle='--')

        # Add labels with class ID and support
        for angle, bar, cls, support, acc in zip(angles, bars, class_ids, supports, accuracies):
            # Position label outside the bar
            label_radius = 0.9
            ax.text(
                angle,
                label_radius,
                f"Class {cls}\n({support})",
                ha="center",
                va="center",
                fontsize=11,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=0.85),
            )
            # Add accuracy percentage inside/near bar
            if isinstance(acc, (int, float)) and acc > 0.15:
                ax.text(
                    angle,
                    acc / 2,
                    f"{acc * 100:.0f}%",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color='white' if acc > 0.5 else 'black',
                    fontweight='bold',
                )

        ax.set_title(
            f"Per-Class Accuracy (Radial View) Model - {classifier}",
            fontsize=16,
            fontweight='bold',
            pad=36,
        )
        plt.tight_layout(pad=1.6)
        radial_path = out_dir / f"radial_class_accuracy{suffix}.png"
        print(f"Saving radial accuracy plot to {radial_path}")  # noqa: T201
        plt.savefig(radial_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        artifacts["radial_accuracy"] = radial_path

        # ---------- Enhanced Cartesian Accuracy Bar Chart ----------
        fig, ax = plt.subplots(figsize=(12.5, 6.5))

        # Color bars based on performance thresholds
        bar_colors = [
            '#d32f2f' if (acc < 0.6) else '#ffa726' if (acc < 0.8) else '#66bb6a'
            for acc in np.nan_to_num(accuracies, nan=0.0)
        ]

        bars = ax.bar(
            class_ids,
            np.nan_to_num(accuracies, nan=0.0),
            alpha=0.85,
            color=bar_colors,
            edgecolor='black',
            linewidth=0.9,
        )

        ax.set_xlabel("Class ID", fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold', labelpad=10)
        ax.set_title(
            f"Per-Class Accuracy Model - {classifier}",
            fontsize=15,
            fontweight='bold',
            pad=18
        )
        ax.set_ylim(0, 1.08)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5, label='80% threshold')
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5, label='60% threshold')
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=6)
        ax.legend(loc='lower right', fontsize=10, frameon=True, fancybox=True, framealpha=0.9)

        # Add value labels on top of bars
        for cls, acc, support, bar in zip(class_ids, accuracies, supports, bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.025,
                f"{acc if np.isfinite(acc) else 0.0:.2f}\nn={support}",
                ha="center",
                va="bottom",
                fontsize=9.5,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8),
            )

        plt.tight_layout(pad=1.2)
        acc_bar_path = out_dir / f"accuracy_per_class_bar{suffix}.png"
        print(f"Saving accuracy bar plot to {acc_bar_path}")  # noqa: T201
        plt.savefig(acc_bar_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        artifacts["accuracy_per_class_bar"] = acc_bar_path

        # ---------- Accuracy vs Support Scatter Plot ----------
        fig, ax = plt.subplots(figsize=(10.5, 6.5))

        scatter_colors = [
            '#d32f2f' if (acc < 0.6) else '#ffa726' if (acc < 0.8) else '#66bb6a'
            for acc in np.nan_to_num(accuracies, nan=0.0)
        ]

        ax.scatter(
            supports,
            np.nan_to_num(accuracies, nan=0.0),
            s=210,
            c=scatter_colors,
            alpha=0.75,
            edgecolors='black',
            linewidth=1.5,
        )

        # Add class labels
        for cls, sup, acc in zip(class_ids, supports, accuracies):
            ax.annotate(
                f"C{cls}",
                (sup, acc if np.isfinite(acc) else 0.0),
                fontsize=9.5,
                fontweight='bold',
                ha='center',
                va='center',
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.75),
            )

        ax.set_xlabel("Number of Samples (Support)", fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel("Accuracy", fontsize=13, fontweight='bold', labelpad=10)
        ax.set_title(
            f"Classification Performance vs Sample Support Model - {classifier}",
            fontsize=15,
            fontweight='bold',
            pad=18,
        )
        ax.set_ylim(-0.05, 1.07)
        ax.axhline(y=0.8, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=6)

        plt.tight_layout(pad=1.2)
        scatter_path = out_dir / f"accuracy_vs_support{suffix}.png"
        print(f"Saving accuracy vs support plot to {scatter_path}")  # noqa: T201
        plt.savefig(scatter_path, dpi=220, bbox_inches='tight')
        plt.close(fig)
        artifacts["accuracy_vs_support"] = scatter_path

    # ---------- Enhanced Localisation Error Bar Chart ----------
    if has_loc and np.any(np.isfinite(rmse_errors)):
        rmse_plot_vals = [0.0 if not np.isfinite(v) else v for v in rmse_errors]
        max_rmse = max(rmse_plot_vals) if rmse_plot_vals else 1.0

        fig, ax = plt.subplots(figsize=(12.5, 9.0))

        # Color bars based on error magnitude
        bar_colors = ['#66bb6a' if rmse < max_rmse * 0.3 else '#ffa726' if rmse < max_rmse * 0.6
        else '#d32f2f' for rmse in rmse_plot_vals]

        bars = ax.bar(class_ids, rmse_plot_vals, alpha=0.88, color=bar_colors,
                      edgecolor='black', linewidth=0.9)

        ax.set_xlabel("Class ID", fontsize=13, fontweight='bold', labelpad=10)
        ax.set_ylabel("Root Mean Squared Error", fontsize=13, fontweight='bold', labelpad=10)
        ax.set_title(
            f"Per-Class Localisation Error Model - {classifier}",
            fontsize=15,
            fontweight='bold',
            pad=18
        )
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.tick_params(axis="both", labelsize=11, width=1.0, length=6)

        # Add value labels
        for cls, rmse, support, bar in zip(class_ids, rmse_plot_vals, supports, bars):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + max_rmse * 0.02,
                f"{rmse:.3f}",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="none", alpha=0.8),
            )

        plt.tight_layout(pad=1.3)
        loc_err_path = out_dir / f"localisation_error_per_class{suffix}.png"
        print(f"Saving localisation error plot to {loc_err_path}")  # noqa: T201
        plt.savefig(loc_err_path, dpi=400, bbox_inches='tight')
        plt.close(fig)
        artifacts["localisation_error_per_class"] = loc_err_path

        # ---------- Radial Localisation Error Plot ----------
        fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(12, 12))

        # Normalize errors for color mapping (inverted - lower error = better = greener)
        norm_errors = np.array(rmse_plot_vals)
        if max_rmse > 0:
            norm_errors = 1.0 - (norm_errors / max_rmse)
        else:
            norm_errors = np.ones_like(norm_errors)

        colors = plt.cm.RdYlGn(np.clip(norm_errors, 0, 1))

        bars = ax.bar(
            angles,
            rmse_plot_vals,
            width=width * 0.85,
            bottom=0.0,
            color=colors,
            alpha=0.9,
            edgecolor='white',
            linewidth=1.6,
        )

        ax.set_theta_direction(-1)
        ax.set_theta_offset(np.pi / 2.0)
        ax.set_ylim(0, max_rmse * 1.1)

        # Remove degree markers / theta tick labels
        ax.set_xticks([])
        ax.set_xticklabels([])

        ax.grid(True, alpha=0.3, linestyle='--')
        ax.tick_params(axis='y', labelsize=7, width=1.0, length=5, pad=6)

        # Add labels
        for angle, bar, cls, support, rmse in zip(angles, bars, class_ids, supports, rmse_plot_vals):
            label_radius = max_rmse * 0.85
            ax.text(
                angle,
                label_radius,
                f"Class {cls}\n({support})",
                ha="center",
                va="center",
                fontsize=20,
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="black", alpha=0.85),
            )
            # Add error value
            if rmse > max_rmse * 0.05:
                ax.text(
                    angle,
                    rmse / 2,
                    f"{rmse:.3f}",
                    ha="center",
                    va="center",
                    fontsize=14,
                    color='white',
                    fontweight='bold',
                    path_effects=[path_effects.Stroke(linewidth=2, foreground='black'),
                                    path_effects.Normal()],
                )

        ax.set_title(
            f"Per-Class Localisation Error - {classifier}",
            fontsize=16,
            fontweight='bold',
            pad=36
        )
        plt.tight_layout(pad=1.6)
        radial_loc_path = out_dir / f"radial_localisation_error{suffix}.png"
        print(f"Saving radial localisation error plot to {radial_loc_path}")  # noqa: T201
        plt.savefig(radial_loc_path, dpi=320, bbox_inches='tight')
        plt.close(fig)
        artifacts["radial_localisation_error"] = radial_loc_path

    return artifacts



def _plot_localisation_vs_snr(
        X: torch.Tensor,
        idx_to_eval: torch.Tensor,
        scaler: StandardScaler,
        y_pos: torch.Tensor,
        pos_hat: torch.Tensor | None,
        out_dir: Path,
        *,
        classifier: str,
        localisation_model: str | None = None,
        include_loss_reflectance: bool = False,
        feature_names: Sequence[str] | None = None,
) -> dict[str, Path]:
    """Scatter plot of predicted localisation vs key scalar features.

    Always plots localisation vs SNR, and when loss/reflectance features are
    available (``include_loss_reflectance``), also plots localisation against
    those scalars using the same error colouring.
    """

    if pos_hat is None or idx_to_eval.numel() == 0:
        return {}

    classifier = classifier.upper()
    if localisation_model:
        classifier = f"{classifier}+{localisation_model.upper()}"
    suffix = f"_{classifier.lower()}"
    if include_loss_reflectance:
        suffix += "_lr"

    out_dir.mkdir(parents=True, exist_ok=True)

    pred_pos = pos_hat.detach().cpu().numpy()
    true_pos = y_pos[idx_to_eval].detach().cpu().numpy()
    error = pred_pos - true_pos

    def _denormalise(column_idx: int) -> np.ndarray:
        values = X[idx_to_eval, column_idx].detach().cpu().numpy()
        return values * scaler.scale_[column_idx] + scaler.mean_[column_idx]

    def _make_plot(feature: np.ndarray, label: str, key: str) -> Path:
        path = out_dir / f"localisation_vs_{key}{suffix}.png"
        fig, ax = plt.subplots(figsize=(12, 8))
        rng = np.random.default_rng(11)
        jitter_scale = max(float(np.ptp(feature)), 1e-5) * 0.004
        feature_jitter = feature + rng.normal(0.0, jitter_scale, size=feature.shape)
        true_offset = rng.normal(0.0, jitter_scale, size=true_pos.shape)

        sc = ax.scatter(
            feature_jitter,
            pred_pos,
            c=error,
            cmap="coolwarm",
            s=75,
            alpha=0.82,
            edgecolors="k",
            linewidths=0.35,
            label="Predicted position",
        )
        ax.scatter(
            feature_jitter,
            true_pos + true_offset,
            c="black",
            s=20,
            alpha=0.35,
            label="True position",
        )
        ax.set_xlabel(label, fontsize=13.5, fontweight="bold", labelpad=12)
        ax.set_ylabel("Position", fontsize=13.5, fontweight="bold", labelpad=12)
        ax.set_title(
            f"Localisation vs {label} (error-coloured) – {classifier}",
            fontsize=16,
            fontweight="bold",
            pad=26,
        )
        ax.legend(loc="upper right", fontsize=11.5, frameon=True, fancybox=True, framealpha=0.9)
        ax.tick_params(axis="both", labelsize=12, width=1.05, length=6)
        ax.grid(True, linestyle=":", alpha=0.30)
        ax.set_facecolor("#fafafa")
        ax.set_axisbelow(True)

        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label("Prediction error", fontsize=12.5, fontweight="bold", labelpad=10)
        cbar.ax.tick_params(labelsize=11, width=1.0)

        plt.tight_layout(pad=1.6)
        print(f"Saving localisation vs {key} plot to {path}")  # noqa: T201
        plt.savefig(path, dpi=175, bbox_inches="tight")
        plt.close(fig)
        return path

    plots: dict[str, Path] = {}
    plots["snr"] = _make_plot(_denormalise(0), "SNR (dB)", "snr")

    if include_loss_reflectance:
        lower_name_map = {name.lower(): idx for idx, name in enumerate(feature_names or [])}
        loss_idx = lower_name_map.get("loss")
        reflectance_idx = lower_name_map.get("reflectance")

        if loss_idx is not None:
            plots["loss"] = _make_plot(_denormalise(loss_idx), "Loss (dB)", "loss")
        if reflectance_idx is not None:
            plots["reflectance"] = _make_plot(
                _denormalise(reflectance_idx), "Reflectance (dB)", "reflectance"
            )

    return plots



def _b64(path: Path) -> str:
    """Return a data‑URL (PNG/JPEG) ready for image_url."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with path.open("rb") as f:
        enc = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{enc}"


def _make_predict_fn(
    model,
    classifier: str,
    device: torch.device,
    *,
    pos_count: int,
):
    """Wrap the classifier into a numpy → probability function for explainability."""

    def _predict(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        data = torch.from_numpy(arr)
        if classifier == "tcn":
            logits, _ = predict_tcn(model, data, device=device, pos_count=pos_count)
        elif classifier == "tcn_binary":
            logits = predict_tcn_binary(model, data, device=device, pos_count=pos_count)
        elif classifier == "tst":
            raise RuntimeError("TST model does not provide classification logits.")
        else:
            raise ValueError(f"Unsupported classifier '{classifier}' for explainability analysis.")
        probs = torch.softmax(logits, dim=1)
        return probs.numpy()

    return _predict


def _describe_feature_space(feature_names: List[str], shape: tuple[int, int] | None = None) -> str:
    feature_list = ", ".join(feature_names)
    shape_text = f"({shape[0]}, {shape[1]})" if shape else "(N, F)"
    return (
        "Model input is a numpy.ndarray[float32] shaped "
        f"{shape_text} with ordered features: "
        f"{feature_list}. Leakage-prone columns (loss / Reflectance) are included only when explicitly enabled."
    )


def _summarise_contributions(
    method_tag: str,
    sample_idx: int,
    pred_cls: int,
    base_val: float,
    predicted_prob: float,
    contrib_vec: np.ndarray,
    feature_names: List[str],
) -> str:
    top_idx = np.argsort(np.abs(contrib_vec))[::-1]
    top_k = top_idx[:5]

    print(
        f"[{method_tag}] Sample {sample_idx} → class {pred_cls}: base prob {base_val:.3f}, "
        f"pred prob {predicted_prob:.3f}."
    )
    print("        Top feature contributions (Δprobability):")
    for rank, j in enumerate(top_k, start=1):
        direction = "raises" if contrib_vec[j] >= 0 else "lowers"
        print(
            f"          #{rank}: {feature_names[j]} {contrib_vec[j]:+.4f} ({direction} class {pred_cls} probability)"
        )

    shap_contribs = ", ".join(
        f"#{idx + 1} {feature_names[j]} ({contrib_vec[j]:+.3f})" for idx, j in enumerate(top_k)
    )
    pos_total = float(np.sum(contrib_vec[contrib_vec > 0]))
    neg_total = float(np.sum(contrib_vec[contrib_vec < 0]))
    summary = (
        f"Sample {sample_idx} → class {pred_cls} | base prob {base_val:.3f} → predicted {predicted_prob:.3f}. "
        f"Top drivers: {shap_contribs}. Σpositive={pos_total:+.3f}, Σnegative={neg_total:+.3f}."
    )
    return summary


def _extract_shap_vector(shap_exp: shap.Explanation, sample_idx: int, class_idx: int) -> np.ndarray:
    """Return the SHAP vector for the given sample/class regardless of layout."""

    values = shap_exp.values
    if values.ndim == 1:
        return np.asarray(values)
    if values.ndim == 2:
        return np.asarray(values[sample_idx])
    if values.ndim == 3:
        return np.asarray(values[sample_idx, :, class_idx])
    raise ValueError("Unexpected SHAP values shape")


def _extract_base_value(shap_exp: shap.Explanation, sample_idx: int, class_idx: int) -> float:
    base_vals = np.asarray(shap_exp.base_values)
    if base_vals.ndim == 0:
        return float(base_vals)
    if base_vals.ndim == 1:
        return float(base_vals[sample_idx])
    if base_vals.ndim == 2:
        return float(base_vals[sample_idx, class_idx])
    raise ValueError("Unexpected SHAP base_values shape")


def _compute_shap_summaries(
        model,
        classifier: str,
        device: torch.device,
        background: np.ndarray,
        samples: np.ndarray,
        sample_indices: List[int],
        pred_lookup: dict[int, int],
        feature_names: List[str],
        *,
        pos_count: int,
) -> List[str]:
    """Compute SHAP attributions and return formatted summaries per sample."""

    if samples.size == 0:
        return []

    bg = np.asarray(background, dtype=np.float32)
    if bg.ndim == 1:
        bg = bg.reshape(1, -1)
    sample_arr = np.asarray(samples, dtype=np.float32)

    masker = shap.maskers.Independent(bg)
    predict_fn = _make_predict_fn(model, classifier, device, pos_count=pos_count)
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")
    max_evals = 2 * sample_arr.shape[1] + 2048
    shap_exp = explainer(sample_arr, max_evals=max_evals)

    dataset_context = _describe_feature_space(
        feature_names, (sample_arr.shape[0], sample_arr.shape[1])
    )
    print(f"[SHAP] {dataset_context}")

    summaries: List[str] = [dataset_context]
    for local_idx, global_idx in enumerate(sample_indices):
        pred_cls = int(pred_lookup[int(global_idx)])
        shap_vec = _extract_shap_vector(shap_exp, local_idx, pred_cls)
        base_val = _extract_base_value(shap_exp, local_idx, pred_cls)

        predicted_prob = float(np.clip(base_val + shap_vec.sum(), 0.0, 1.0))
        summary = _summarise_contributions(
            "SHAP",
            global_idx,
            pred_cls,
            base_val,
            predicted_prob,
            np.asarray(shap_vec, dtype=np.float32),
            feature_names,
        )
        summaries.append(summary)

    return summaries


def _compute_lime_summaries(
        model,
        classifier: str,
        device: torch.device,
        background: np.ndarray,
        samples: np.ndarray,
        sample_indices: List[int],
        pred_lookup: dict[int, int],
        feature_names: List[str],
        *,
        pos_count: int,
) -> List[str]:
    """Compute LIME attributions and return formatted summaries per sample."""

    if LimeTabularExplainer is None:
        raise RuntimeError("LIME explainability requested but lime package is not installed.")

    if samples.size == 0:
        return []

    bg = np.asarray(background, dtype=np.float32)
    if bg.ndim == 1:
        bg = bg.reshape(1, -1)
    sample_arr = np.asarray(samples, dtype=np.float32)

    predict_fn = _make_predict_fn(model, classifier, device, pos_count=pos_count)
    explainer = LimeTabularExplainer(
        bg,
        feature_names=feature_names,
        discretize_continuous=False,
        mode="classification",
    )

    dataset_context = _describe_feature_space(
        feature_names, (sample_arr.shape[0], sample_arr.shape[1])
    )
    print(f"[LIME] {dataset_context}")

    summaries: List[str] = [dataset_context]
    prob_matrix = predict_fn(sample_arr)
    for local_idx, global_idx in enumerate(sample_indices):
        pred_cls = int(pred_lookup[int(global_idx)])
        explanation = explainer.explain_instance(
            sample_arr[local_idx],
            predict_fn,
            top_labels=1,
            num_features=sample_arr.shape[1],
        )
        local_exp = explanation.local_exp.get(pred_cls)
        if not local_exp:
            print(f"[LIME] No explanation produced for sample {global_idx} class {pred_cls}")
            continue

        contrib_vec = np.zeros(sample_arr.shape[1], dtype=np.float32)
        for feat_idx, weight in local_exp:
            contrib_vec[int(feat_idx)] = float(weight)

        intercepts = getattr(explanation, "intercept", {})
        base_val = float(intercepts[pred_cls]) if pred_cls in intercepts else 0.0
        predicted_prob = float(prob_matrix[local_idx, pred_cls])

        summary = _summarise_contributions(
            "LIME",
            global_idx,
            pred_cls,
            base_val,
            predicted_prob,
            contrib_vec,
            feature_names,
        )
        summaries.append(summary)

    return summaries

def _llm_explain_with_self_reflection(
        img_paths: List[Path],
        classifier_type: str = "tcn",
        openai_model: str = "gpt-4o-mini",
        attribution_summaries: List[str] | None = None,
        *,
        attribution_method: str = "shap",
) -> Tuple[str, str, str, bool] | None:
    """
    DIRECT pass -> SELF-REFLECTION pass -> OPS DIGEST pass with explicit TrueC/PredC handling.
    Returns (direct_text, refined_text, digest_text, rag_used_flag) or None if no API key.
    """
    from datetime import datetime
    api_key = cfg.OPENAI_API_KEY
    if not api_key:
        print("OPENAI_API_KEY not set – skipping LLM explanation")
        return None

    client = OpenAI(api_key=api_key)

    # Per-call run id so all three passes can be grouped in the logs
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    # ---------- Fault class block (shared) ---------------------------------
    fault_classes_block = (
        "Position context: The given position value × 10 ≈ physical distance in meters (e.g., 0.10 = 1.0m).\n"
        "Fault Classes are labelled as follows:\n"
        "id\tfault type\t\t\ttypical signs\n"
        "0\tnormal / no fault\t\tbaseline trace, loss ≈ 0, position = 0\n"
        "1\tfiber tapping\t\tlocalized disturbance, moderate loss due to coupler, reflectance can be low/absent\n"
        "2\tbad splice\t\t\tlocalized event with excess loss, small/possible reflection\n"
        "3\tbending event\t\tgradual/medium loss, usually no clear reflectance peak\n"
        "4\tdirty connector\t\tconnector-like event with extra loss and messy/variable reflectance\n"
        "5\tfiber cut\t\t\tabrupt large loss/end-of-trace, may appear near end position\n"
        "6\tPC connector\t\tclean connector-type reflective event, expected position\n"
        "7\treflector\t\t\tstrongly reflective event, high reflectance value\n"
    )

    # ---------- RAG retrieval (shared) -------------------------------------
    query = "OTDR fault plots – " + ", ".join(p.stem for p in img_paths)
    try:
        retrieved = retrieve(query, k=5)
    except Exception as exc:
        print(f"RAG retrieval failed. {exc}")
        retrieved = []
    ref_block = build_reference_block(retrieved, max_context_tokens=1600)
    rag_flag = bool(retrieved)
    if rag_flag:
        print("RAG retrieval successful, using retrieved snippets in LLM prompt.")

    # ---------- Attribution text (shared) ----------------------------------
    method_label_raw = attribution_method or "none"
    method_label = method_label_raw.upper()
    attr_text = "\n".join(attribution_summaries) if attribution_summaries else ""
    has_attribution = bool(attr_text)

    # ---------- Shared logging helper --------------------------------------
    def _log_llm_input(
        section: str,
        system_prompt: str,
        user_content_text_parts: List[dict[str, Any]],
        image_paths_for_log: List[Path] | None = None,
    ) -> None:
        """
        Log input to the LLM in a human-readable way, without dumping base64 images.

        - `user_content_text_parts`: same structure as the API call but *only* text entries.
        - `image_paths_for_log`: list of Path objects for the images that were sent to the LLM.
        """
        try:
            with open("outputs/llm_inputs.log", "a", encoding="utf-8") as f:
                f.write(f"=== {section} (run_id={run_id}) ===\n")
                f.write("SYSTEM_PROMPT:\n")
                f.write(repr(system_prompt) + "\n")
                f.write("USER_CONTENT_TEXT_PARTS:\n")
                f.write(repr(user_content_text_parts) + "\n")
                if image_paths_for_log:
                    f.write("IMAGE_PATHS:\n")
                    for p in image_paths_for_log:
                        f.write(f"  - {str(p)}\n")
                f.write("\n")
        except Exception:
            # Logging must never affect main functionality
            pass

    # ---------- DIRECT pass -------------------------------------------------
    # IMPORTANT: disambiguate TrueC vs PredC and force wording when they differ
    true_pred_rules = (
        "READ TITLES CAREFULLY: each figure title contains 'TrueC=<int> PredC=<int>'.\n"
        "- 'TrueC' is the ground-truth class.\n"
        "- 'PredC' is the model's predicted class (may be 'N/A' for regression-only models).\n"
        "- If TrueC != PredC, explicitly write: \"misclassified as <PredC> (true: <TrueC>)\".\n"
        "- NEVER swap or rename these; do not call TrueC the prediction or PredC the truth.\n"
    )

    method_clause = (
        f"Use the reference snippets and the {method_label} feature attributions when available. "
        f"Cite snippets like [1], [2] when used. If {method_label} is present, explicitly state which features raised/lowered "
        "the predicted class probability.\n\n"
    )
    if not has_attribution:
        method_clause = (
            "Use the reference snippets. No attribution is provided for these samples; "
            "focus on visual evidence and retrieved context.\n\n"
        )

    system_direct = (
        "You are an optical-fibre fault-analysis expert. "
        "Given the following figures (OTDR amplitude over P-points; titles include TrueC/PredC and positions), "
        f"write a concise explanation for each figure predicted by a {classifier_type} model. "
        "Explain fault type, position (in meters), likely causes, and concrete next actions. "
        + method_clause
        + true_pred_rules
        + fault_classes_block
    )
    print(f"SYSTEM_DIRECT:\n{system_direct}")
    # --- Build the actual API payload --------------------------------------
    user_direct_parts: List[dict[str, Any]] = [
        {"type": "input_text", "text": "Reference snippets:\n" + (ref_block or "*<no snippets retrieved>*")},
    ]
    if attr_text:
        user_direct_parts.append(
            {"type": "input_text", "text": f"{method_label} attributions per sample:\n" + attr_text}
        )
    user_direct_parts.append(
        {"type": "input_text", "text": "Selected samples for inspection (images):"}
    )

    # Separate image payload (for API) and log-friendly representation
    direct_image_payloads: List[dict[str, Any]] = []
    for p in img_paths:
        direct_image_payloads.append(
            {"type": "input_image", "image_url": _b64(p)}
        )
    user_direct_parts += direct_image_payloads

    # ---- LOG DIRECT INPUT (without base64) --------------------------------
    direct_text_parts_for_log = [
        part for part in user_direct_parts if part.get("type") == "input_text"
    ]
    _log_llm_input(
        section="DIRECT PASS",
        system_prompt=system_direct,
        user_content_text_parts=direct_text_parts_for_log,
        image_paths_for_log=img_paths,
    )

    direct_text = _call_responses_api(client, openai_model, system_direct, user_direct_parts)

    # ---------- SELF-REFLECTION pass ---------------------------------------
    # Provide SAME images so the reviewer can verify visually.
    # Repeat the TrueC/PredC rule to avoid drift.
    reflect_intro = (
        "You are a meticulous QA reviewer for optical-fibre explanations. "
        "You will receive: (a) the same context (reference snippets"
    )
    if has_attribution:
        reflect_intro += " and feature-attribution summaries"
    reflect_intro += ", and (b) a DRAFT explanation. "

    reflect_rules = (
        "OUTPUT ONLY an improved explanation that:\n"
        f"1) Matches {method_label} signs (positive values → increase predicted class probability; negative → decrease).\n"
        f"2) Mentions the top-k absolute {method_label} contributors (k≈5) in plain English.\n"
        "3) Grounds standards/definitions with citations [i] that exist in the provided snippet list.\n"
        "4) Avoids hallucinated numbers; if a number isn’t present, use cautious wording or a justified range.\n"
        "5) Keeps the operator section actionable (2–3 steps).\n\n"
    )
    if not has_attribution:
        reflect_rules = (
            "OUTPUT ONLY an improved explanation that:\n"
            "1) Aligns with the provided figures and snippet references.\n"
            "2) Clearly calls out any disagreements between TrueC and PredC.\n"
            "3) Grounds standards/definitions with citations [i] that exist in the provided snippet list.\n"
            "4) Avoids hallucinated numbers; if a number isn’t present, use cautious wording or a justified range.\n"
            "5) Keeps the operator section actionable (2–3 steps).\n\n"
        )

    system_reflect = (
        reflect_intro
        + reflect_rules
        + true_pred_rules
        + fault_classes_block
    )
    print(f"[LLM] {system_reflect}")
    reflect_user_content: List[dict[str, Any]] = [
        {"type": "input_text", "text": "Reference snippets:\n" + (ref_block or "*<no snippets retrieved>*")},
    ]
    if attr_text:
        reflect_user_content.append(
            {"type": "input_text", "text": f"{method_label} attributions per sample:\n" + attr_text}
        )
    reflect_user_content.append(
        {"type": "input_text", "text": "Images (verify titles with TrueC/PredC and positions):"}
    )

    reflect_image_payloads: List[dict[str, Any]] = []
    for p in img_paths:
        reflect_image_payloads.append(
            {"type": "input_image", "image_url": _b64(p)}
        )
    reflect_user_content += reflect_image_payloads
    reflect_user_content.append(
        {"type": "input_text", "text": "DRAFT explanation to review:\n" + direct_text}
    )

    # ---- LOG SELF-REFLECTION INPUT (without base64) -----------------------
    reflect_text_parts_for_log = [
        part for part in reflect_user_content if part.get("type") == "input_text"
    ]
    _log_llm_input(
        section="SELF-REFLECTION PASS",
        system_prompt=system_reflect,
        user_content_text_parts=reflect_text_parts_for_log,
        image_paths_for_log=img_paths,
    )

    refined_text = _call_responses_api(client, openai_model, system_reflect, reflect_user_content)

    # ---------- FIELD OPS DIGEST pass --------------------------------------
    system_digest = (
        "You lead field operations for a fibre network."
        " Convert the improved explanation into a 3-section incident digest:"
        "\n1) \"Key Findings\" (bullet list)."
        "\n2) \"Impact vs SNR\" (describe confidence level relative to SNR trends in the figures)."
        "\n3) \"Next Actions\" (2–3 concrete steps)."
        " Tie any localisation statements to positions, and flag uncertain ones."
    )
    digest_content: List[dict[str, Any]] = [
        {"type": "input_text", "text": "Reference snippets:\n" + (ref_block or "*<no snippets retrieved>*")},
    ]
    if attr_text:
        digest_content.append(
            {"type": "input_text", "text": f"{method_label} attributions:\n" + attr_text}
        )
    digest_content.append(
        {"type": "input_text", "text": "Improved explanation to convert:\n" + refined_text}
    )
    print("System prompt: ", system_digest)
    # ---- LOG DIGEST INPUT (only text, no images in this pass) -------------
    digest_text_parts_for_log = [
        part for part in digest_content if part.get("type") == "input_text"
    ]
    _log_llm_input(
        section="DIGEST PASS",
        system_prompt=system_digest,
        user_content_text_parts=digest_text_parts_for_log,
        image_paths_for_log=None,
    )

    digest_text = _call_responses_api(client, openai_model, system_digest, digest_content)

    return direct_text, refined_text, digest_text, rag_flag



# --------------------------------------------------
# Main eval flow
# --------------------------------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--mode",
    type=click.Choice(["pipeline", "direct", "siamese"], case_sensitive=False),
    required=True,
    help="Evaluation mode: pipeline, direct, or siamese.",
)
@click.option(
    "--classifier",
    type=click.Choice(["tcn", "tcn_binary", "tst"], case_sensitive=False),
    required=True,
    help="Classifier to use.",
)
@click.option(
    "--data", "data_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/OTDR_DATA.csv"),
    show_default=True,
    help="Path to the dataset CSV. Must have 'Class', 'SNR', 'Position', and P{N} columns.",
)
@click.option(
    "--detector",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to GRU-AE weights (defaults based on feature configuration).",
)
@click.option(
    "--tcn-anomaly-only/--tcn-all-data",
    "tcn_anomaly_only",
    default=False,
    help="When evaluating TCN models, select the anomaly-only classifier variant.",
)
@click.option(
    "--orchestrate-tst",
    is_flag=True,
    help="Chain binary TCN ➜ anomaly-only TCN ➜ TST for localisation evaluation.",
)
@click.option(
    "--tcn-full-to-tst",
    is_flag=True,
    help=(
        "Run full multi-class TCN classification before TST localisation (uses TCN"
        " predictions as class tokens)."
    ),
)
@click.option(
    "--tcn-full-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Override path to the full multi-class TCN checkpoint used with --tcn-full-to-tst.",
)
@click.option(
    "--cls-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to classifier weights; defaults by --classifier.",
)
@click.option(
    "--binary-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional override for the binary TCN checkpoint (pipeline mode).",
)
@click.option(
    "--anomaly-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional override for the anomaly-only TCN checkpoint (pipeline mode).",
)
@click.option(
    "--tst-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional override for the TST localisation checkpoint (pipeline mode).",
)
@click.option(
    "--siamese-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Path to the Siamese encoder checkpoint (for siamese mode).",
)
@click.option(
    "--num-samples",
    type=click.IntRange(0, None),
    default=18,
    show_default=True,
    help=(
        "Random samples to visualise & explain. The script automatically ensures that up to "
        f"{LLM_SAMPLE_TARGET} unique samples are analysed for the LLM explanations."
    ),
)
@click.option(
    "--explain-method",
    type=click.Choice(["shap", "lime", "both", "none"], case_sensitive=False),
    default="both",
    show_default=True,
    help=(
        "Feature attribution method used for sample explainability (both runs SHAP + LIME). "
        "Use 'none' to disable attribution generation."
    ),
)
@click.option(
    "--extra-feature",
    "extra_features",
    multiple=True,
    help=(
        "Optional additional feature columns to append to the default measurement "
        "set (repeat flag for multiple columns)."
    ),
)
@click.option(
    "--use-loss-reflectance",
    is_flag=True,
    help=(
        "Append 'loss' and 'Reflectance' to the measurement vector and load models "
        "trained with those leakage-prone features."
    ),
)
@click.option(
    "--out-dir",
    type=str,
    default="eval_outputs",
    show_default=True,
    help="Folder name under outputs/ for artifacts.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="cuda | cpu | leave empty for auto-detect.",
)
@click.option(
    "--test-noise-level",
    type=click.FloatRange(min=0.0),
    default=0.0,
    show_default=True,
    help=(
        "Standard deviation of Gaussian noise added to the scaled test set features. "
        "Set to 0 to disable noise injection."
    ),
)
@click.option(
    "--profile-flops",
    is_flag=True,
    help=(
        "Profile classifier/localiser inference to report FLOP counts alongside latency. "
        "Adds overhead; disable for routine evaluation runs."
    ),
)
@click.option(
    "--use-llm",
    is_flag=True,
    help="Enable LLM-based explanation generation (requires OpenAI API key).",
)
def main(
    mode,
    classifier,
    data_path,
    detector,
    cls_path,
    binary_path,
    anomaly_path,
    tst_path,
    tcn_full_to_tst,
    tcn_full_path,
    num_samples,
    explain_method,
    out_dir,
    device,
    test_noise_level,
    tcn_anomaly_only,
    orchestrate_tst,
    extra_features,
    use_loss_reflectance,
    profile_flops,
    use_llm,
    siamese_path,
):  # noqa: C901
    out_dir = Path("outputs") / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # Siamese Evaluation Mode
    # -------------------------------------------------------------------------
    if mode == "siamese":
        print(f"\n=== Evaluating Siamese Network on {device} ===")
        device_obj = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Load Data
        print("Loading data...")
        df = load_raw_dataframe(data_path)
        train_df, val_df, test_df = make_splits(df)
        
        measurements = measurement_columns(train_df, include_loss_reflectance=use_loss_reflectance)
        layout = summarise_feature_layout(measurements)
        pos_count = int(layout["pos_count"])

        scaler = StandardScaler()
        scaler.fit(train_df[measurements].values.astype(np.float32))
        
        splits = tensorise_splits(train_df, val_df, test_df, scaler, measurement_override=measurements)
        test_X = splits["test"].X

        if test_noise_level > 0:
            test_X = test_X + torch.randn_like(test_X) * float(test_noise_level)
            print(f"[INFO] Added Gaussian noise to test set with σ={test_noise_level:.4f}.")
            
        test_X = test_X.to(device_obj)
        test_y = splits["test"].y_class.cpu().numpy()
        
        # Load Model
        n_classes = int(splits["train"].y_class.max().item() + 1)
        
        # Determine actual input channels after reshaping
        # It's 1 (Pos) + 1 (SNR) + num_extras
        in_ch = 2 + (len(measurements) - 1 - pos_count)
        
        base_model = OTDR_TCN(n_classes=n_classes, in_ch=in_ch).to(device_obj)
        siamese_net = SiameseNetwork(base_model).to(device_obj)
        
        if siamese_path and Path(siamese_path).exists():
            siamese_net.base_model.load_state_dict(torch.load(siamese_path, map_location=device_obj))
            print(f"Loaded Siamese weights from {siamese_path}")
        else:
            print("[WARN] No checkpoint provided (or not found). Using random weights.")

        siamese_net.eval()
        
        # Helper to reshape inputs for OTDR_TCN
        def _reshape_input(xb: torch.Tensor) -> torch.Tensor:
            if xb.size(1) < pos_count + 1:
                pass # Already shaped?
            pos = xb[:, 1 : 1 + pos_count]
            extras = xb[:, 1 + pos_count :]
            seq_len = pos.size(1)
            channels = [pos]
            snr = xb[:, 0]
            channels.append(snr.unsqueeze(1).repeat(1, seq_len))
            if extras.numel() > 0:
                for idx in range(extras.size(1)):
                    feat = extras[:, idx]
                    channels.append(feat.unsqueeze(1).repeat(1, seq_len))
            return torch.stack(channels, dim=1)

        print("Generating embeddings...")
        embeddings = []
        BATCH_SIZE = 128
        with torch.no_grad():
            for i in range(0, len(test_X), BATCH_SIZE):
                batch = test_X[i:i+BATCH_SIZE]
                # Fix shape
                batch = _reshape_input(batch)
                
                emb = siamese_net.forward_once(batch)
                embeddings.append(emb.cpu().numpy())
        
        embeddings = np.concatenate(embeddings, axis=0)
        
        # PCA Visualisation
        print("Projecting with PCA...")
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        emb_pca = pca.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(emb_pca[:, 0], emb_pca[:, 1], c=test_y, cmap="tab10", alpha=0.6, s=15)
        plt.colorbar(scatter, label="Class")
        ax.set_title("Siamese Embeddings (PCA) - Test Set")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(True, linestyle=":", alpha=0.3)
        out_file = out_dir / "siamese_pca.png"
        fig.savefig(out_file, dpi=150)
        plt.close(fig)
        print(f"Saved PCA plot to {out_file}")
        
        return

    if classifier != "tcn" and tcn_anomaly_only:
        raise click.BadOptionUsage(
            "--tcn-anomaly-only",
            "The anomaly-only flag is only applicable when --classifier=tcn.",
        )

    explain_method = explain_method.lower()

    extras = _dedupe_preserve(extra_features)
    if explain_method == "both":
        explain_methods = ("shap", "lime")
    elif explain_method == "none":
        explain_methods = ("none",)
    else:
        explain_methods = (explain_method,)

    requested_methods = ",".join(explain_methods) if explain_methods else "none"

    wandb_run = _init_wandb_run(
        {
            "mode": mode,
            "classifier": classifier,
            "requested_methods": requested_methods,
            "num_samples": num_samples,
            "tcn_anomaly_only": tcn_anomaly_only,
            "orchestrate_tst": orchestrate_tst,
            "tcn_full_to_tst": tcn_full_to_tst,
            "use_loss_reflectance": use_loss_reflectance,
            "extra_features": list(extras),
            "test_noise_level": test_noise_level,
            "profile_flops": profile_flops,
        }
    )
    feature_suffix = "_lr" if use_loss_reflectance else ""
    detector_path = (
        Path(detector)
        if detector is not None
        else Path("models") / (f"gru_ae{feature_suffix}.pt" if feature_suffix else "gru_ae.pt")
    )
    binary_default = Path("models") / (
        f"tcn_binary{feature_suffix}.pt" if feature_suffix else "tcn_binary.pt"
    )
    anomaly_default = Path("models") / (
        f"tcn_anomaly{feature_suffix}.pt" if feature_suffix else "tcn_anomaly.pt"
    )
    tcn_full_default = Path("models") / (
        f"tcn_full{feature_suffix}.pt" if feature_suffix else "tcn_full.pt"
    )
    tst_default = Path("models") / (
        f"tst{feature_suffix}.pt" if feature_suffix else "tst.pt"
    )

    binary_ckpt = Path(binary_path) if binary_path else binary_default
    anomaly_ckpt = Path(anomaly_path) if anomaly_path else anomaly_default
    tst_ckpt = Path(tst_path) if tst_path else tst_default
    tcn_full_ckpt = Path(tcn_full_path) if tcn_full_path else tcn_full_default

    # ---------- data ---------- #
    df = load_raw_dataframe(data_path)
    _, _, test_df = make_splits(df)

    scaler = StandardScaler()
    feature_names_meta: list[str] | None = None
    scaler_meta: dict[str, Any] | None = None
    detector_meta: dict[str, Any] | None = None
    scaler_candidates: list[Path] = []
    candidate_dirs = [detector_path.parent, binary_ckpt.parent, anomaly_ckpt.parent, tst_ckpt.parent]
    seen_dirs: set[Path] = set()
    for base in candidate_dirs:
        base = base.resolve()
        if base in seen_dirs:
            continue
        seen_dirs.add(base)
        if feature_suffix:
            scaler_candidates.append(base / f"scaler{feature_suffix}.json")
        scaler_candidates.append(base / "scaler.json")
    for candidate in scaler_candidates:
        if not candidate.exists():
            continue
        scaler_meta = json.loads(candidate.read_text())
        feature_cfg = scaler_meta.get("feature_config")
        if feature_cfg and feature_cfg.get("columns"):
            feature_names_meta = list(feature_cfg["columns"])
        else:
            feature_names_meta = (
                scaler_meta.get("feature_names")
                or scaler_meta.get("active_features")
                or feature_names_meta
            )
        scaler.mean_ = np.asarray(scaler_meta["mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(scaler_meta["scale"], dtype=np.float32)
        break

    if scaler_meta is None:
        detector_meta_path = detector_path.with_suffix(".json")
        detector_meta = json.loads(detector_meta_path.read_text())
        feature_cfg = detector_meta.get("feature_config")
        if feature_cfg and feature_cfg.get("columns"):
            feature_names_meta = list(feature_cfg["columns"])
        else:
            feature_names_meta = (
                detector_meta.get("feature_names")
                or detector_meta.get("active_features")
            )
        scaler.mean_ = np.asarray(detector_meta["scaler_mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(detector_meta["scaler_scale"], dtype=np.float32)

    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]

    try:
        requested_cols = measurement_columns(
            test_df,
            extras,
            include_loss_reflectance=use_loss_reflectance,
        )
    except KeyError as exc:
        raise click.BadOptionUsage("--extra-feature", str(exc)) from exc

    expected_feature_config = build_feature_config(
        requested_cols,
        use_loss_reflectance=use_loss_reflectance,
        requested_extra_features=extras,
    )

    _validate_metadata_features(scaler_meta, expected_feature_config, "Scaler metadata")
    _validate_metadata_features(detector_meta, expected_feature_config, "Detector metadata")

    if feature_names_meta:
        meas_cols = list(feature_names_meta)
        missing_cols = [c for c in meas_cols if c not in test_df.columns]
        if missing_cols:
            raise ValueError(
                "Dataset is missing feature columns required by the scaler metadata: "
                + ", ".join(missing_cols)
            )
        if meas_cols != requested_cols:
            raise ValueError(
                "Requested feature configuration does not match the saved scaler metadata "
                "(did you train with --use-loss-reflectance?)."
            )
    else:
        meas_cols = list(requested_cols)

    if len(meas_cols) != scaler.n_features_in_:
        raise ValueError(
            "Scaler metadata dimensionality does not match selected measurement columns."
        )

    leakage_cols = {"Reflectance", "loss", "Loss"}
    leaked = [c for c in meas_cols if c in leakage_cols]
    if leaked and not (extras or use_loss_reflectance):
        raise ValueError(
            "Measurement column selection must not include leakage features, found: "
            + ", ".join(leaked)
        )
    if leaked and extras and not use_loss_reflectance:
        print(
            "[WARN] Additional features include potential leakage columns: "
            + ", ".join(leaked)
        )

    layout = summarise_feature_layout(meas_cols)
    pos_count = int(layout["pos_count"])
    extra_scalar_count = len(layout["extra_features"])
    input_channels = 1 + 1 + extra_scalar_count

    if pos_count <= 0:
        raise ValueError("No positional measurement columns (P*) were detected in the dataset.")

    print("[INFO] Using measurement columns (ordered): " + ", ".join(meas_cols))
    if extras:
        print("[INFO] Extra features appended: " + ", ".join(extras))
    if use_loss_reflectance:
        print("[INFO] Loss/Reflectance features enabled – using dedicated checkpoints.")

    splits = tensorise_splits(
        test_df,
        test_df,
        test_df,
        scaler,
        measurement_override=meas_cols,
    )  # only need "test" key
    X_test = splits["test"].X
    y_cls_test = splits["test"].y_class
    y_pos_test = splits["test"].y_pos

    if test_noise_level > 0:
        noise = torch.randn_like(X_test) * float(test_noise_level)
        X_test = X_test + noise
        print(
            f"[INFO] Added Gaussian noise to test set with σ={test_noise_level:.4f}."
        )

    tcn_full_bridge = orchestrate_tst and tcn_full_to_tst

    if tcn_full_to_tst and not orchestrate_tst:
        raise click.BadOptionUsage(
            "--tcn-full-to-tst",
            "The full-TCN➜TST bridge requires --orchestrate-tst to be enabled.",
        )

    if classifier == "tst" and not orchestrate_tst and not tcn_full_bridge:
        fault_mask = y_cls_test != 0
        if fault_mask.sum().item() == 0:
            raise ValueError("No faulty samples available in the test set for TST evaluation.")
        X_test = X_test[fault_mask]
        y_cls_test = y_cls_test[fault_mask]
        y_pos_test = y_pos_test[fault_mask]
        class_feature = y_cls_test.to(dtype=X_test.dtype).unsqueeze(1)
        tst_features = torch.cat([class_feature, X_test], dim=1)
    else:
        tst_features = None

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None else torch.device(device)
    )
    print("[INFO] Using device:", device)

    classifier_for_eval = "tcn" if tcn_full_bridge else classifier
    classifier_display_label = classifier.upper()
    localisation_display_label: str | None = None
    wandb_prefix = "tcn_full_tst_" if tcn_full_bridge else ""

    if tcn_full_bridge:
        classifier_display_label = "TCN"
        localisation_display_label = "TST"

    # ---------- load models ---------- #
    if classifier_for_eval == "tcn":
        cls_base = "tcn_anomaly" if tcn_anomaly_only else "tcn_full"
    elif classifier_for_eval == "tcn_binary":
        cls_base = "tcn_binary"
    else:
        cls_base = "tst"
    default_cls_name = f"{cls_base}{feature_suffix}.pt" if feature_suffix else f"{cls_base}.pt"
    if tcn_full_bridge:
        cls_path = tcn_full_ckpt
    else:
        cls_path = Path(cls_path) if cls_path else Path("models") / default_cls_name

    cls_meta = load_classifier_meta(cls_path)
    _validate_metadata_features(cls_meta, expected_feature_config, "Classifier metadata")

    if cls_meta:
        meta_features = cls_meta.get("active_features") or cls_meta.get("feature_names")
        if meta_features and list(meta_features) != meas_cols:
            raise ValueError(
                "Classifier checkpoint features do not match the requested measurement configuration."
            )
        meta_use_lr = cls_meta.get("use_loss_reflectance")
        if meta_use_lr is not None and bool(meta_use_lr) != use_loss_reflectance:
            raise ValueError(
                "Classifier checkpoint was trained with a different loss/reflectance setting."
            )
        meta_pos = cls_meta.get("pos_count")
        if meta_pos is not None and int(meta_pos) != pos_count:
            raise ValueError("Classifier checkpoint expects a different number of position columns.")
        meta_channels = cls_meta.get("input_channels")
        if meta_channels is not None and int(meta_channels) != input_channels:
            raise ValueError("Classifier checkpoint expects a different input channel arrangement.")

    default_n_classes = int(df["Class"].max() + 1)

    if tcn_full_bridge:
        for chk in (tcn_full_ckpt, tst_ckpt):
            if not chk.exists():
                raise FileNotFoundError(f"Checkpoint not found: {chk}")

    if orchestrate_tst and mode != "pipeline" and not tcn_full_bridge:
        if classifier != "tst":
            raise click.BadOptionUsage(
                "--orchestrate-tst",
                "The chained orchestrator is only available when --classifier=tst.",
            )

        for chk in (binary_ckpt, anomaly_ckpt, tst_ckpt):
            if not chk.exists():
                raise FileNotFoundError(f"Checkpoint not found: {chk}")

        for label, chk in (
            ("Binary TCN", binary_ckpt),
            ("Anomaly-only TCN", anomaly_ckpt),
            ("TST", tst_ckpt),
        ):
            meta = load_classifier_meta(chk)
            _validate_metadata_features(meta, expected_feature_config, f"{label} metadata")

        cascade_start = perf_counter()
        pipeline_result = run_cascade(
            X_test=X_test,
            y_cls_test=y_cls_test,
            y_pos_test=y_pos_test,
            device=device,
            out_dir=out_dir,
            binary_path=binary_ckpt,
            anomaly_path=anomaly_ckpt,
            tst_path=tst_ckpt,
            default_n_classes=default_n_classes,
            pos_count=pos_count,
            input_channels=input_channels,
        )
        print(
            f"[PIPELINE] Cascade evaluation completed in "
            f"{perf_counter() - cascade_start:.2f}s"
        )
        print("\n".join(pipeline_result.summary_lines))  # noqa: T201
        print(f"Confusion matrix saved to {pipeline_result.confusion_matrix_path}")  # noqa: T201
        if wandb_run:
            wandb_run.finish()
        return

    if classifier_for_eval == "tcn_binary":
        n_classes = 2
    elif classifier_for_eval == "tcn":
        variant = "anomaly_only" if tcn_anomaly_only else "full"
        if cls_meta and "variant" in cls_meta:
            meta_variant = cls_meta["variant"]
            if meta_variant not in {"full", "anomaly_only"}:
                raise ValueError(f"Unknown TCN variant '{meta_variant}' in metadata.")
            if meta_variant != variant:
                if tcn_anomaly_only:
                    raise ValueError(
                        "Requested anomaly-only TCN but provided checkpoint metadata "
                        "describes the full-data variant."
                    )
                else:
                    print(
                        f"[INFO] Detected '{meta_variant}' TCN variant from metadata; adjusting evaluation."
                    )
                    variant = meta_variant
        tcn_anomaly_only = variant == "anomaly_only"
        if tcn_anomaly_only:
            if cls_meta is None:
                raise ValueError(
                    "Anomaly-only TCN checkpoints must include metadata for class remapping."
                )
            class_labels_meta = cls_meta.get("class_labels") or cls_meta.get("original_classes")
            if not class_labels_meta:
                raise ValueError(
                    "Anomaly-only TCN metadata must include 'class_labels' or 'original_classes'."
                )
            n_classes = len(class_labels_meta)
        else:
            if cls_meta and "n_classes" in cls_meta:
                n_classes = int(cls_meta["n_classes"])
            else:
                n_classes = default_n_classes
    else:
        n_classes = default_n_classes

    if mode == "pipeline":
        for chk in (binary_ckpt, anomaly_ckpt, tst_ckpt):
            if not chk.exists():
                raise FileNotFoundError(f"Checkpoint not found: {chk}")

        cascade_start = perf_counter()
        pipeline_result = run_cascade(
            X_test=X_test,
            y_cls_test=y_cls_test,
            y_pos_test=y_pos_test,
            device=device,
            out_dir=out_dir,
            binary_path=binary_ckpt,
            anomaly_path=anomaly_ckpt,
            tst_path=tst_ckpt,
            default_n_classes=default_n_classes,
            pos_count=pos_count,
            input_channels=input_channels,
        )
        print(
            f"[PIPELINE] Cascade evaluation completed in "
            f"{perf_counter() - cascade_start:.2f}s"
        )
        print("\n".join(pipeline_result.summary_lines))  # noqa: T201
        print(f"Confusion matrix saved to {pipeline_result.confusion_matrix_path}")  # noqa: T201
        return

    anomaly_mapping: dict[int, int] | None = None
    if classifier_for_eval == "tcn" and tcn_anomaly_only:
        X_test, y_cls_test, y_pos_test, anomaly_mapping, _ = remap_anomaly_only_targets(
            X_test,
            y_cls_test,
            y_pos_test,
            cls_meta or {},
        )
        mapping_desc = ", ".join(
            f"{orig}→{idx}" for orig, idx in sorted(anomaly_mapping.items())
        )
        print(f"[INFO] Evaluating anomaly-only TCN with mapping: {mapping_desc}")

    seq_len = tst_features.shape[1] if tst_features is not None else X_test.shape[1]
    classifier_model = load_classifier(
        classifier_for_eval,
        cls_path,
        seq_len=seq_len,
        n_classes=n_classes,
        device=device,
        input_channels=input_channels,
    )

    if tcn_full_bridge:
        tst_meta = load_classifier_meta(tst_ckpt)
        _validate_metadata_features(tst_meta, expected_feature_config, "TST metadata")
        if tst_meta:
            tst_pos = tst_meta.get("pos_count")
            if tst_pos is not None and int(tst_pos) != pos_count:
                raise ValueError(
                    "TST checkpoint expects a different number of position columns."
                )
            tst_channels = tst_meta.get("input_channels")
            if tst_channels is not None and int(tst_channels) != input_channels:
                raise ValueError(
                    "TST checkpoint expects a different input channel arrangement."
                )
    # ---------- model summary ---------- #
    try:
        from torchinfo import summary as torchinfo_summary

        # Build a dummy input that matches what the model.forward expects.
        # Try the raw test tensor first.
        if classifier_for_eval == "tst":
            if tst_features is None or tst_features.size(0) == 0:
                raise ValueError("tst_features is empty; cannot build dummy input.")
            dummy = tst_features[:1].to(device)
        else:
            if X_test.size(0) == 0:
                raise ValueError("X_test is empty; cannot build dummy input.")
            dummy = X_test[:1].to(device)

            # TCN variants expect channel-first inputs; mirror prediction preprocessing
            # by converting vector rows to (B, C, L).
            if classifier_for_eval in {"tcn", "tcn_binary"}:
                from model_functions.tcn import _to_two_channel

                dummy = _to_two_channel(dummy.cpu(), pos_count=pos_count).to(device)

        def _render_summary(tensor: torch.Tensor) -> str:
            """Return a formatted torchinfo summary for the given dummy tensor."""

            with torch.no_grad():
                info = torchinfo_summary(
                    classifier_model,
                    input_data=tensor,
                    col_names=("input_size", "output_size", "num_params", "trainable"),
                    depth=6,
                    verbose=0,
                )
            return str(info)

        print("\n[MODEL SUMMARY] ------------------------------")
        try:
            # First attempt: run summary with raw dummy shape
            summary_text = _render_summary(dummy)
        except Exception:
            # Common alt layout for TCNs: (B, C, T). Try permuting if 3D or reshaping if 2D.
            if dummy.dim() == 3:
                dummy_alt = dummy.permute(0, 2, 1).contiguous()
            elif dummy.dim() == 2:
                # If your model expects channel-first, treat features as channels.
                dummy_alt = dummy.unsqueeze(1)  # (B, 1, T)
            else:
                dummy_alt = dummy

            summary_text = _render_summary(dummy_alt)

        print(summary_text)

        total_params = sum(p.numel() for p in classifier_model.parameters())
        trainable_params = sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)
        print(f"[INFO] Total params: {total_params:,} | Trainable params: {trainable_params:,}")
        print("[MODEL SUMMARY END] --------------------------\n")

    except Exception as e:
        # Hard fallback: never crash evaluation because of summary
        print(f"[WARN] Could not create torchinfo summary: {e}")
        print(classifier_model)
        total_params = sum(p.numel() for p in classifier_model.parameters())
        trainable_params = sum(p.numel() for p in classifier_model.parameters() if p.requires_grad)
        print(f"[INFO] Total params: {total_params:,} | Trainable params: {trainable_params:,}\n")

    idx_to_eval = torch.arange(X_test.size(0))
    loc_indices = idx_to_eval

    # ------------- inference ------------- #
    pos_hat = None
    logits = None
    preds_cls = None
    eval_start = perf_counter()

    def _aggregate_profiler_flops(prof: torch.profiler.profile) -> float | None:
        try:
            flops = [evt.flops for evt in prof.key_averages() if evt.flops is not None]
            return float(sum(flops)) if flops else None
        except Exception as exc:  # pragma: no cover - best-effort aggregation
            print(f"[WARN] Unable to aggregate FLOPs from profiler: {exc}")
            return None

    def _format_flops(value: float | None) -> str:
        if value is None:
            return "N/A"
        units = (("TFLOPs", 1e12), ("GFLOPs", 1e9), ("MFLOPs", 1e6))
        for label, factor in units:
            if value >= factor:
                return f"{value / factor:.2f} {label}"
        return f"{value:.0f} FLOPs"

    def _run_inference_with_profile(
        tag: str,
        call_fn,
        *,
        flop_model: torch.nn.Module | None = None,
        flop_inputs: tuple[torch.Tensor, ...] | None = None,
    ) -> tuple[Any, float, float | None]:
        start_time = perf_counter()
        flop_total: float | None = None
        if profile_flops:
            activities = [torch.profiler.ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            try:
                with torch.profiler.profile(
                    activities=activities, with_flops=True
                ) as prof:
                    result = call_fn()
                duration = perf_counter() - start_time
                flop_total = _aggregate_profiler_flops(prof)
                return result, duration, flop_total
            except Exception as prof_exc:  # pragma: no cover - profiling optional
                print(f"[WARN] Profiling for {tag} failed with torch.profiler: {prof_exc}")
                if flop_model is not None and flop_inputs is not None:
                    try:
                        from fvcore.nn import FlopCountAnalysis

                        flop_total = float(FlopCountAnalysis(flop_model, flop_inputs).total())
                    except Exception as fv_exc:  # pragma: no cover - optional dependency
                        print(f"[WARN] FLOP analysis fallback failed for {tag}: {fv_exc}")
                start_time = perf_counter()
                result = call_fn()
                duration = perf_counter() - start_time
                return result, duration, flop_total

        result = call_fn()
        duration = perf_counter() - start_time
        return result, duration, flop_total

    def _log_inference_stats(label: str, duration: float, flop_total: float | None) -> None:
        suffix = f" | FLOPs: {_format_flops(flop_total)}" if profile_flops else ""
        print(f"[EVAL] {label} inference completed in {duration:.2f}s{suffix}")

    if tcn_full_bridge:
        (logits, _), tcn_duration, tcn_flops = _run_inference_with_profile(
            "tcn_full_bridge",
            lambda: predict_tcn(
                classifier_model,
                X_test[idx_to_eval],
                pos_count=pos_count,
            ),
            flop_model=classifier_model,
            flop_inputs=(X_test[:1].to(device),),
        )
        _log_inference_stats("TCN_FULL", tcn_duration, tcn_flops)
        preds_cls = logits.argmax(1)
        # Use PREDICTED faults to drive the localisation stage (realistic pipeline behavior)
        fault_mask = preds_cls != 0
        if fault_mask.sum().item() == 0:
            print("[WARN] No faults predicted by TCN. Skipping TST localisation.")
            pos_hat = None
        else:
            loc_indices = torch.nonzero(fault_mask, as_tuple=False).view(-1)
            class_feature = preds_cls[loc_indices].to(dtype=X_test.dtype).unsqueeze(1)
            bridge_features = torch.cat([class_feature, X_test[loc_indices]], dim=1)
            tst_model = load_classifier(
                "tst",
                tst_ckpt,
                seq_len=bridge_features.shape[1],
                n_classes=n_classes,
                device=device,
                input_channels=input_channels,
            )
            pos_hat, tst_duration, tst_flops = _run_inference_with_profile(
                "tst_bridge",
                lambda: predict_tst(tst_model, bridge_features, device=device),
                flop_model=tst_model,
                flop_inputs=(bridge_features.to(device),),
            )
            _log_inference_stats("TST", tst_duration, tst_flops)
    elif classifier_for_eval == "tcn":
        (logits, pos_hat), infer_duration, infer_flops = _run_inference_with_profile(
            "tcn",
            lambda: predict_tcn(
                classifier_model,
                X_test[idx_to_eval],
                pos_count=pos_count,
            ),
            flop_model=classifier_model,
            flop_inputs=(X_test[:1].to(device),),
        )
        _log_inference_stats("TCN", infer_duration, infer_flops)
        preds_cls = logits.argmax(1)
    elif classifier_for_eval == "tcn_binary":
        logits, infer_duration, infer_flops = _run_inference_with_profile(
            "tcn_binary",
            lambda: predict_tcn_binary(
                classifier_model,
                X_test[idx_to_eval],
                pos_count=pos_count,
            ),
            flop_model=classifier_model,
            flop_inputs=(X_test[:1].to(device),),
        )
        _log_inference_stats("TCN_BINARY", infer_duration, infer_flops)
        preds_cls = logits.argmax(1)
    elif classifier_for_eval == "tst":
        pos_hat, infer_duration, infer_flops = _run_inference_with_profile(
            "tst",
            lambda: predict_tst(classifier_model, tst_features[idx_to_eval]),
            flop_model=classifier_model,
            flop_inputs=(tst_features[:1].to(device),),
        )
        _log_inference_stats("TST", infer_duration, infer_flops)
        preds_cls = None

    if pos_hat is not None:
        y_pos_true_global = y_pos_test[loc_indices].numpy()
        y_pos_pred_global = pos_hat.numpy()

        rmse = root_mean_squared_error(y_pos_true_global, y_pos_pred_global)
        mae = mean_absolute_error(y_pos_true_global, y_pos_pred_global)
        
        # apply TN filter
        mask = (y_pos_true_global != 0) | (y_pos_pred_global != 0)
        if mask.any():
            rmse = root_mean_squared_error(y_pos_true_global[mask], y_pos_pred_global[mask])
            mae = mean_absolute_error(y_pos_true_global[mask], y_pos_pred_global[mask])
    else:
        rmse = None
        mae = None

    classifier_plot_label = classifier_display_label
    if localisation_display_label:
        classifier_plot_label = f"{classifier_display_label}+{localisation_display_label}"

    acc = None
    auc_val = None
    cm_path: Path | None = None
    y_true: np.ndarray | None = None
    y_pred: np.ndarray | None = None

    if preds_cls is not None:
        acc = accuracy_score(y_cls_test[idx_to_eval].numpy(), preds_cls.numpy())
        if rmse is not None:
            print(
                f"Eval subset size = {idx_to_eval.size(0)} | Acc = {acc:.3f} | RMSE = {rmse:.3f} | MAE = {mae:.3f}"
            )  # noqa: T201
        else:
            print(
                f"Eval subset size = {idx_to_eval.size(0)} | Acc = {acc:.3f}"
            )  # noqa: T201
        y_true = y_cls_test[idx_to_eval].numpy()
        y_pred = preds_cls.numpy()
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=3))
        if logits is not None and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            auc_val = roc_auc_score(y_true, probs)
            print(f"AUC = {auc_val:.3f}")

        # Confusion matrix plot
        cm = confusion_matrix(y_cls_test[idx_to_eval].numpy(), preds_cls.numpy())
        fig, ax = plt.subplots(figsize=(9, 7))
        disp = ConfusionMatrixDisplay(cm)
        disp.plot(include_values=True, cmap="Blues", colorbar=False, ax=ax, values_format="d")
        ax.set_facecolor("#fafafa")
        ax.grid(False)
        ax.set_title(
            f"Confusion Matrix – Eval subset ({classifier_plot_label})",
            fontsize=15,
            fontweight="bold",
            pad=18,
        )
        ax.set_xlabel("Predicted label", fontsize=12, fontweight="bold", labelpad=12)
        ax.set_ylabel("True label", fontsize=12, fontweight="bold", labelpad=12)
        ax.tick_params(axis="both", which="major", labelsize=11, width=1.2, length=6)
        plt.setp(ax.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")
        for text in ax.texts:
            text.set_fontsize(11)
            text.set_fontweight("bold")
        fig.tight_layout(pad=1.1)
        cm_suffix = classifier_plot_label.lower().replace("+", "_").replace(" ", "_")
        if use_loss_reflectance:
            cm_suffix += "_lr"
        cm_path = out_dir / f"confusion_matrix_{cm_suffix}.png"
        fig.savefig(cm_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
    else:
        if rmse is not None:
            print(
                f"Eval subset size = {idx_to_eval.size(0)} | RMSE = {rmse:.3f}"
            )  # noqa: T201

    print(
        f"[EVAL] {classifier_plot_label} evaluation completed in "
        f"{perf_counter() - eval_start:.2f}s"
    )

    scatter_path: Path | None = None
    scatter_key: str | None = None
    radial_artifacts: dict[str, Path] = {}
    y_true_np = y_cls_test[idx_to_eval].numpy() if y_cls_test is not None else None
    y_pred_np = preds_cls.numpy() if preds_cls is not None else None
    y_pos_true_np = y_pos_test[loc_indices].numpy() if pos_hat is not None else None
    y_pos_pred_np = pos_hat.detach().cpu().numpy() if pos_hat is not None else None

    radial_y_true = y_true_np
    radial_y_pred = y_pred_np
    if pos_hat is not None:
        # Create full-sized position arrays aligned with y_true_np/y_pred_np
        # idx_to_eval usually covers the whole test set in pipeline mode
        
        # Ground truth positions for ALL samples
        y_pos_true_np = y_pos_test[idx_to_eval].numpy()
        
        # Predicted positions: init with 0.0 (default for Class 0/Normal)
        y_pos_pred_np = np.zeros_like(y_pos_true_np)
        
        # Map the subset predictions (pos_hat) back to their full-array indices
        abs_to_rel = {idx.item(): i for i, idx in enumerate(idx_to_eval)}
        
        pos_hat_np = pos_hat.detach().cpu().numpy()
        loc_indices_np = loc_indices.detach().cpu().numpy()
        
        for i, abs_idx in enumerate(loc_indices_np):
            if abs_idx in abs_to_rel:
                rel_idx = abs_to_rel[abs_idx]
                y_pos_pred_np[rel_idx] = pos_hat_np[i]
                
        # Now radial_y_true/pred can just be the full arrays
        radial_y_true = y_true_np
        radial_y_pred = y_pred_np

    if (tcn_full_bridge or classifier_for_eval in {"tst", "tcn"}) and y_pos_true_np is not None:
        scatter_stem = "tcn_full_bridge" if tcn_full_bridge else classifier_for_eval
        scatter_path = out_dir / f"{scatter_stem}_localisation_scatter.png"
        fig, ax = plt.subplots(figsize=(9, 7))
        rng = np.random.default_rng(7)
        jitter_scale = max(float(np.ptp(y_pos_true_np)), float(np.ptp(y_pos_pred_np)), 1e-3) * 0.005
        jitter_true = y_pos_true_np + rng.normal(0.0, jitter_scale, size=y_pos_true_np.shape)
        jitter_pred = y_pos_pred_np + rng.normal(0.0, jitter_scale, size=y_pos_pred_np.shape)

        error = y_pos_pred_np - y_pos_true_np
        
        # Filter out True Negatives (where both true and pred are 0)
        # We only care about error when a fault exists OR was predicted
        mask = (y_pos_true_np != 0) | (y_pos_pred_np != 0)
        if mask.any():
            filtered_error = error[mask]
        else:
            filtered_error = error  # Fallback to avoid empty array if no faults exist
        
        mean_error = np.mean(filtered_error)
        var_error = np.var(filtered_error)
        mae_error = np.mean(np.abs(filtered_error))
        rmse_error = np.sqrt(np.mean(filtered_error**2))
        print(
            f"[{classifier_plot_label}] Localisation Error (excluding TNs): "
            f"Mean={mean_error:.4f}, Variance={var_error:.4f}, "
            f"MAE={mae_error:.4f}, RMSE={rmse_error:.4f}"
        )

        # Compute mean and variance of actual and predicted values
        pred_mean = np.mean(y_pos_pred_np)
        pred_std  = np.std(y_pos_pred_np)
        true_mean = np.mean(y_pos_true_np)
        true_std  = np.std(y_pos_true_np)

        ax.scatter(
            jitter_true,
            jitter_pred,
            alpha=0.7,
            edgecolor="black",
            linewidth=0.6,
            s=55,
            color="tab:blue",
            label=f"{'Predictions':<11} \u03bc={pred_mean:.4f}  \u03c3={pred_std:.4f}",
        )
        diag_min = float(min(y_pos_true_np.min(), y_pos_pred_np.min()))
        diag_max = float(max(y_pos_true_np.max(), y_pos_pred_np.max()))
        ax.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            linestyle="--",
            color="tab:red",
            label=f"{'Actual':<13} \u03bc={true_mean:.4f}  \u03c3={true_std:.4f}",
        )
        ax.set_xlabel("True fault position", fontsize=13, fontweight="bold", labelpad=10)
        ax.set_ylabel("Predicted fault position", fontsize=13, fontweight="bold", labelpad=10)
        ax.set_title(
            f"{classifier_plot_label} localisation",
            fontsize=15,
            fontweight="bold",
            pad=18,
        )
        

        
        ax.legend(frameon=True, fontsize=11, loc="upper left", fancybox=True, framealpha=0.92)
        ax.tick_params(axis="both", labelsize=11, width=1.1, length=6)
        ax.grid(True, linestyle=":", alpha=0.22)
        ax.set_facecolor("#fafafa")
        fig.tight_layout(pad=1.5)
        scatter_key = scatter_stem
        fig.savefig(scatter_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

        # ---------------------------------------------------------------------
        # Density (KDE) plot for Actual vs Predicted
        # ---------------------------------------------------------------------
        try:
            # Filter out 0.0s for density plot to focus on fault distribution
            dens_true = y_pos_true_np[y_pos_true_np > 0]
            dens_pred = y_pos_pred_np[y_pos_pred_np > 0]

            if dens_true.size > 1 and dens_pred.size > 1:
                density_path = out_dir / f"{scatter_stem}_density_kde.png"
                fig_kde, ax_kde = plt.subplots(figsize=(10, 6))

                # Compute KDE
                kde_true = gaussian_kde(dens_true)
                kde_pred = gaussian_kde(dens_pred)
                
                # Evaluation grid
                x_grid = np.linspace(
                    min(dens_true.min(), dens_pred.min()),
                    max(dens_true.max(), dens_pred.max()),
                    500
                )

                ax_kde.plot(x_grid, kde_true(x_grid), color="tab:green", label="Actual Faults", linewidth=2)
                ax_kde.fill_between(x_grid, kde_true(x_grid), color="tab:green", alpha=0.2)
                
                ax_kde.plot(x_grid, kde_pred(x_grid), color="tab:blue", label="Predicted Faults", linewidth=2, linestyle="--")
                ax_kde.fill_between(x_grid, kde_pred(x_grid), color="tab:blue", alpha=0.1)

                ax_kde.set_xlabel("Fault Position (normalized)", fontsize=12, fontweight="bold")
                ax_kde.set_ylabel("Density", fontsize=12, fontweight="bold")
                ax_kde.set_title(f"Fault Position Density (KDE) - {classifier_plot_label}", fontsize=14, fontweight="bold")
                ax_kde.legend(fontsize=11)
                ax_kde.grid(True, linestyle=":", alpha=0.3)
                fig_kde.tight_layout()
                fig_kde.savefig(density_path, dpi=150)
                plt.close(fig_kde)
                print(f"Saved density plot to {density_path}")
        except Exception as exc:
            print(f"[WARN] Failed to generate KDE plot: {exc}")

    if (radial_y_true is not None and radial_y_pred is not None) or (
            y_pos_true_np is not None and y_pos_pred_np is not None
    ):
        radial_artifacts = _plot_radial(
            classifier_plot_label,
            radial_y_true,
            radial_y_pred,
            list(range(n_classes)),
            out_dir,
            y_pos_true=y_pos_true_np,
            y_pos_pred=y_pos_pred_np,
            include_loss_reflectance=use_loss_reflectance,
        )

    localisation_paths = _plot_localisation_vs_snr(
        X_test,
        loc_indices,
        scaler,
        y_pos_test,
        pos_hat,
        out_dir,
        classifier=classifier_display_label,
        localisation_model=localisation_display_label,
        include_loss_reflectance=use_loss_reflectance,
        feature_names=meas_cols,
    )

    # ------------- random visualisations ------------- #
    rng = np.random.default_rng(42)
    available_idx = idx_to_eval.numpy()
    if num_samples <= 0 or available_idx.size == 0:
        chosen = np.empty(0, dtype=int)
    else:
        sample_goal = min(max(num_samples, LLM_SAMPLE_TARGET), available_idx.size)
        chosen = rng.choice(
            available_idx,
            size=sample_goal,
            replace=False,
        )
    llm_limit = min(30, chosen.size)
    llm_indices_all = chosen[:llm_limit] if llm_limit > 0 else np.empty(0, dtype=int)
    llm_batches = [
        llm_indices_all[i : i + LLM_SAMPLE_TARGET]
        for i in range(0, llm_indices_all.size, LLM_SAMPLE_TARGET)
    ]

    idx_eval_cpu = idx_to_eval.detach().cpu()
    pred_lookup: dict[int, int] = {}
    if preds_cls is not None:
        preds_cpu = preds_cls.detach().cpu()
        pred_lookup = {
            int(idx_eval_cpu[i].item()): int(preds_cpu[i].item())
            for i in range(idx_eval_cpu.size(0))
        }
    pos_lookup: dict[int, float] = {}
    if pos_hat is not None:
        pos_cpu = pos_hat.detach().cpu()
        loc_eval_cpu = loc_indices.detach().cpu()
        pos_lookup = {
            int(loc_eval_cpu[i].item()): float(pos_cpu[i].item())
            for i in range(loc_eval_cpu.size(0))
        }

    # ------------- Feature attribution explainability ------------- #
    classifier_name = classifier_plot_label
    llm_dir = Path("outputs/llm_output")
    llm_dir.mkdir(parents=True, exist_ok=True)

    for batch_num, llm_indices in enumerate(llm_batches, start=1):
        attr_summary_map: dict[str, List[str]] = {method: [] for method in explain_methods}
        compute_attr_methods = [m for m in explain_methods if m != "none"]
        if use_llm and preds_cls is not None and llm_indices.size > 0 and compute_attr_methods:
            try:
                bg_size = min(50, idx_eval_cpu.size(0))
                if bg_size > 0:
                    background = X_test[idx_eval_cpu[:bg_size]].numpy()
                    sample_tensor = torch.as_tensor(llm_indices, dtype=torch.long)
                    sample_block = X_test[sample_tensor].numpy()
                    for method in compute_attr_methods:
                        if method == "shap":
                            attr_summary_map[method] = _compute_shap_summaries(
                                classifier_model,
                                classifier_for_eval,
                                device,
                                background,
                                sample_block,
                                llm_indices.tolist(),
                                pred_lookup,
                                meas_cols,
                                pos_count=pos_count,
                            )
                        elif method == "lime":
                            attr_summary_map[method] = _compute_lime_summaries(
                                classifier_model,
                                classifier_for_eval,
                                device,
                                background,
                                sample_block,
                                llm_indices.tolist(),
                                pred_lookup,
                                meas_cols,
                                pos_count=pos_count,
                            )
                        else:
                            raise ValueError(
                                f"Unsupported explainability method '{method}'."
                            )
            except Exception as exc:  # pragma: no cover - fallback path
                print(f"[WARN] Explainability computation failed: {exc}")

        img_paths: list[Path] = []
        num_points = pos_count
        loss_idx: int | None = None
        reflectance_idx: int | None = None
        if use_loss_reflectance:
            lower_name_map = {name.lower(): i for i, name in enumerate(meas_cols)}
            loss_idx = lower_name_map.get("loss")
            reflectance_idx = lower_name_map.get("reflectance")
        for idx in llm_indices:
            idx_int = int(idx)
            amp_scaled = X_test[idx_int][1 : 1 + num_points].detach().cpu().numpy()
            amp = (
                amp_scaled * scaler.scale_[1 : 1 + num_points]
                + scaler.mean_[1 : 1 + num_points]
            )
            snr_scaled = X_test[idx_int][0].item()
            snr = float(snr_scaled * scaler.scale_[0] + scaler.mean_[0])
            loss_val: float | None = None
            reflectance_val: float | None = None
            if loss_idx is not None:
                loss_scaled = X_test[idx_int][loss_idx].item()
                loss_val = float(
                    loss_scaled * scaler.scale_[loss_idx] + scaler.mean_[loss_idx]
                )
            if reflectance_idx is not None:
                refl_scaled = X_test[idx_int][reflectance_idx].item()
                reflectance_val = float(
                    refl_scaled * scaler.scale_[reflectance_idx]
                    + scaler.mean_[reflectance_idx]
                )
            t_cls = int(y_cls_test[idx_int].item())
            p_cls = pred_lookup.get(idx_int)
            t_pos = float(y_pos_test[idx_int].item())
            p_pos = pos_lookup.get(idx_int)
            img_paths.append(
                _visualise_sample(
                    classifier=classifier,
                    amps=amp,
                    snr=snr,
                    loss_db=loss_val,
                    reflectance_db=reflectance_val,
                    true_cls=t_cls,
                    pred_cls=p_cls,
                    true_pos=t_pos,
                    pred_pos=p_pos,
                    idx=idx_int,
                    out_dir=out_dir,
                    classifier_label=classifier_display_label,
                    localisation_label=localisation_display_label,
                )
            )

        explain_outputs: dict[str, tuple[str, str, str, bool]] = {}
        if use_llm and img_paths:
            for method in explain_methods:
                try:
                    explain_pair = _llm_explain_with_self_reflection(
                        img_paths,
                        openai_model="gpt-5",
                        classifier_type=classifier_plot_label,
                        attribution_summaries=attr_summary_map.get(method),
                        attribution_method=method,
                    )
                except Exception as exc:  # pragma: no cover - API errors
                    print(f"[WARN] LLM explanation ({method}) failed: {exc}")
                    continue
                if explain_pair:
                    explain_outputs[method] = explain_pair

        for method, explain_tuple in explain_outputs.items():
            direct_text, refined_text, digest_text, rag_flag = explain_tuple
            method_slug = method.lower()
            explanation_file = llm_dir / (
                f"llm_explanation_{method_slug}_batch{batch_num}.txt"
            )
            i = 1
            while explanation_file.exists():
                explanation_file = llm_dir / (
                    f"llm_explanation_{method_slug}_batch{batch_num}_{i}.txt"
                )
                i += 1

            header = (
                f"LLM explanation for eval subset {'with' if rag_flag else 'without'} RAG "
                f"for {classifier_name} in {mode} mode ({method.upper()} attributions)"
                f" covering {len(llm_indices)} samples (batch {batch_num}):\n\n"
            )
            combined = (
                header
                + "=== DIRECT ===\n"
                + direct_text.strip()
                + "\n\n=== SELF-REFLECTION (REVISED) ===\n"
                + refined_text.strip()
                + "\n\n=== FIELD OPS DIGEST ===\n"
                + digest_text.strip()
                + "\n"
            )
            explanation_file.write_text(combined, encoding="utf-8")
            print(
                "LLM explanation (direct + self-reflection + ops digest) "
                f"saved to {explanation_file.name}"
            )  # noqa: T201
            if wandb_run:
                wandb_run.log({f"llm_{method_slug}_batch{batch_num}": wandb.Html(combined)})

        if wandb_run:
            if img_paths:
                wandb_run.log(
                    {
                        f"{wandb_prefix}sample_traces_batch{batch_num}": [
                            wandb.Image(str(path)) for path in img_paths
                        ],
                    }
                )
            for method, summaries in attr_summary_map.items():
                if summaries:
                    table = wandb.Table(columns=["summary"])
                    for summary in summaries:
                        table.add_data(summary)
                    wandb_run.log({f"{method}_summaries_batch{batch_num}": table})

    if wandb_run:
        metrics_payload: dict[str, float] = {}
        if acc is not None:
            metrics_payload["accuracy"] = float(acc)
        if rmse is not None:
            metrics_payload["rmse"] = float(rmse)
        if auc_val is not None:
            metrics_payload["auc"] = float(auc_val)
        if metrics_payload:
            wandb_run.log(metrics_payload)

        if cm_path and cm_path.exists():
            wandb_run.log({f"{wandb_prefix}confusion_matrix": wandb.Image(str(cm_path))})
        if radial_artifacts:
            ra = radial_artifacts.get("radial_accuracy")
            if ra and ra.exists():
                wandb_run.log({f"{wandb_prefix}radial_accuracy": wandb.Image(str(ra))})
            acc_bar = radial_artifacts.get("accuracy_per_class_bar")
            if acc_bar and acc_bar.exists():
                wandb_run.log({f"{wandb_prefix}accuracy_per_class_bar": wandb.Image(str(acc_bar))})
            loc_err = radial_artifacts.get("localisation_error_per_class")
            if loc_err and loc_err.exists():
                wandb_run.log({f"{wandb_prefix}localisation_error_per_class": wandb.Image(str(loc_err))})
        if localisation_paths:
            for key, path in localisation_paths.items():
                if path and path.exists():
                    wandb_run.log(
                        {f"{wandb_prefix}localisation_vs_{key}": wandb.Image(str(path))}
                    )
        if scatter_path and scatter_path.exists():
            scatter_log_name = scatter_key or Path(scatter_path).stem
            wandb_run.log({f"{wandb_prefix}{scatter_log_name}": wandb.Image(str(scatter_path))})
        if img_paths:
            wandb_run.log({f"{wandb_prefix}sample_traces": [wandb.Image(str(path)) for path in img_paths],})
        for method, summaries in attr_summary_map.items():
            if summaries:
                table = wandb.Table(columns=["summary"])
                for summary in summaries:
                    table.add_data(summary)
                wandb_run.log({f"{method}_summaries": table})

    if wandb_run:
        wandb_run.finish()



if __name__ == "__main__":
    main()
