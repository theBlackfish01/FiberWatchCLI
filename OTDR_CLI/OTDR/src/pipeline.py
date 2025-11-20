"""Binary ➜ anomaly-only TCN ➜ TST evaluation pipeline."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import json

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    roc_auc_score,
    root_mean_squared_error,
)
from sklearn.preprocessing import StandardScaler

from data_helper import (
    load_raw_dataframe,
    make_splits,
    measurement_columns,
    summarise_feature_layout,
    tensorise_splits,
)
from model_functions.tcn import OTDR_TCN, predict as predict_tcn
from model_functions.tcn_binary import OTDR_TCNBinary, predict as predict_tcn_binary
from model_functions.tst import TimeSeriesTransformer, predict as predict_tst

warnings.filterwarnings("ignore", category=FutureWarning)  # noqa: T201

__all__ = [
    "PipelineResult",
    "Stage1Result",
    "Stage2Result",
    "Stage3Result",
    "load_classifier",
    "load_classifier_meta",
    "remap_anomaly_only_targets",
    "run_cascade",
    "run_full_tcn_pipeline",
]


@dataclass
class Stage1Result:
    accuracy: float
    auc: float | None
    predicted_faults: int
    total_samples: int
    truth_faults: int
    predictions: torch.Tensor
    probabilities: torch.Tensor


@dataclass
class Stage2Result:
    predictions: dict[int, int]
    accuracy: float | None
    evaluated_samples: int


@dataclass
class Stage3Result:
    rmse: float | None
    mae: float | None
    median_ae: float | None
    bias: float | None
    evaluated_samples: int
    plot_paths: dict[str, Path]


@dataclass
class PipelineResult:
    stage1: Stage1Result
    stage2: Stage2Result
    stage3: Stage3Result
    summary_lines: list[str]
    confusion_matrix: np.ndarray
    confusion_matrix_path: Path
    classification_report: str
    final_predictions: torch.Tensor
    binary_confusion_matrix: np.ndarray | None = None
    binary_confusion_matrix_path: Path | None = None


def _unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    ordered: list[Path] = []
    for p in paths:
        p = p.resolve()
        if p not in seen:
            ordered.append(p)
            seen.add(p)
    return ordered


def _resolve_device(preferred: str | None) -> torch.device:
    if preferred is None:
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    try:
        return torch.device(preferred)
    except (RuntimeError, ValueError) as exc:  # pragma: no cover - defensive path
        raise click.BadParameter(f"Invalid device specification: {preferred}") from exc


def load_classifier(
    kind: str,
    cls_path: Path,
    *,
    seq_len: int,
    n_classes: int,
    device: torch.device,
    input_channels: int | None = None,
):
    if kind == "tcn":
        model = OTDR_TCN(n_classes=n_classes, in_ch=input_channels or 2)
    elif kind == "tcn_binary":
        model = OTDR_TCNBinary(in_ch=input_channels or 2)
    elif kind == "tst":
        model = TimeSeriesTransformer(seq_len=seq_len)
    else:  # pragma: no cover - guarded by CLI choices
        raise ValueError("classifier kind must be 'tcn', 'tcn_binary' or 'tst'")
    model.load_state_dict(torch.load(cls_path, map_location=device))
    return model.eval().to(device)


def load_classifier_meta(cls_path: Path) -> dict[str, object] | None:
    meta_path = cls_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def remap_anomaly_only_targets(
    X: torch.Tensor,
    y_cls: torch.Tensor,
    y_pos: torch.Tensor,
    meta: dict[str, object],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int], torch.Tensor]:
    return remap_anomaly_only_targets_with_options(
        X, y_cls, y_pos, meta, validate_presence=True
    )


def remap_anomaly_only_targets_with_options(
    X: torch.Tensor,
    y_cls: torch.Tensor,
    y_pos: torch.Tensor,
    meta: dict[str, object],
    *,
    validate_presence: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[int, int], torch.Tensor]:
    mapping_raw = meta.get("class_index_map")
    if mapping_raw is not None:
        mapping = {int(k): int(v) for k, v in mapping_raw.items()}
    else:
        original_classes = meta.get("original_classes")
        if not original_classes:
            raise ValueError(
                "Anomaly-only TCN metadata must include 'class_index_map' or 'original_classes'."
            )
        mapping = {int(orig): idx for idx, orig in enumerate(original_classes)}

    mask = torch.zeros_like(y_cls, dtype=torch.bool)
    for orig in mapping:
        mask |= y_cls == int(orig)

    selected = torch.nonzero(mask, as_tuple=True)[0]
    if validate_presence and selected.numel() == 0:
        raise ValueError(
            "No samples with the anomaly classes required by the anomaly-only TCN were found."
        )

    X_sel = X[selected]
    y_pos_sel = y_pos[selected]
    y_cls_sel = y_cls[selected]

    remapped = torch.empty_like(y_cls_sel, dtype=torch.long)
    for orig, idx in mapping.items():
        remapped[y_cls_sel == int(orig)] = int(idx)

    return X_sel, remapped, y_pos_sel, mapping, selected


def _run_tst_localisation(
    *,
    class_feature: torch.Tensor,
    candidates: torch.Tensor,
    tst_path: Path,
    device: torch.device,
    default_n_classes: int,
    out_dir: Path,
    true_positions: torch.Tensor,
) -> Stage3Result:
    if candidates.size(0) == 0:
        return Stage3Result(
            rmse=None,
            mae=None,
            median_ae=None,
            bias=None,
            evaluated_samples=0,
            plot_paths={},
        )

    tst_input = torch.cat([class_feature, candidates], dim=1)
    tst_model = load_classifier(
        "tst",
        tst_path,
        seq_len=tst_input.shape[1],
        n_classes=default_n_classes,
        device=device,
    )
    pos_hat = predict_tst(tst_model, tst_input, device=device)
    evaluated = int(pos_hat.size(0))
    pos_true = true_positions.cpu().numpy()
    pos_pred = pos_hat.cpu().numpy()

    rmse = root_mean_squared_error(pos_true, pos_pred) if evaluated > 0 else None
    mae = mean_absolute_error(pos_true, pos_pred) if evaluated > 0 else None
    median_ae = float(np.median(np.abs(pos_pred - pos_true))) if evaluated > 0 else None
    bias = float(np.mean(pos_pred - pos_true)) if evaluated > 0 else None

    plot_paths: dict[str, Path] = {}
    if evaluated > 0:
        scatter_path = out_dir / "tst_localisation_scatter.png"
        fig, ax = plt.subplots()
        ax.scatter(pos_true, pos_pred, alpha=0.6, edgecolor="none")
        diag_min = float(min(pos_true.min(), pos_pred.min()))
        diag_max = float(max(pos_true.max(), pos_pred.max()))
        ax.plot(
            [diag_min, diag_max],
            [diag_min, diag_max],
            linestyle="--",
            color="tab:red",
            label="Ideal",
        )
        ax.set_xlabel("True fault position")
        ax.set_ylabel("Predicted fault position")
        ax.set_title("TST localisation – predictions vs. ground truth")
        ax.legend()
        fig.tight_layout()
        fig.savefig(scatter_path, dpi=150)
        plt.close(fig)
        plot_paths["scatter"] = scatter_path

        error_path = out_dir / "tst_localisation_error_hist.png"
        fig, ax = plt.subplots()
        errors = pos_pred - pos_true
        ax.hist(errors, bins=20, color="tab:blue", alpha=0.75)
        ax.set_xlabel("Prediction error")
        ax.set_ylabel("Count")
        ax.set_title("TST localisation – error distribution")
        fig.tight_layout()
        fig.savefig(error_path, dpi=150)
        plt.close(fig)
        plot_paths["error_hist"] = error_path

    return Stage3Result(
        rmse=rmse,
        mae=mae,
        median_ae=median_ae,
        bias=bias,
        evaluated_samples=evaluated,
        plot_paths=plot_paths,
    )


def _load_scaler_metadata(
    *,
    checkpoint_dirs: Sequence[Path],
    use_loss_reflectance: bool,
) -> tuple[StandardScaler, list[str] | None]:
    scaler = StandardScaler()
    suffix = "_lr" if use_loss_reflectance else ""
    candidates: list[Path] = []
    for base in _unique_paths(checkpoint_dirs):
        if suffix:
            candidates.append(base / f"scaler{suffix}.json")
        candidates.append(base / "scaler.json")

    for candidate in candidates:
        if not candidate.exists():
            continue
        meta = json.loads(candidate.read_text())
        mean = meta.get("mean") or meta.get("scaler_mean")
        scale = meta.get("scale") or meta.get("scaler_scale")
        if mean is None or scale is None:
            continue
        scaler.mean_ = np.asarray(mean, dtype=np.float32)
        scaler.scale_ = np.asarray(scale, dtype=np.float32)
        meta_use_lr = meta.get("use_loss_reflectance")
        if meta_use_lr is not None and bool(meta_use_lr) != use_loss_reflectance:
            raise ValueError(
                "Scaler metadata was generated with a different loss/reflectance configuration."
            )
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
        features = meta.get("feature_names") or meta.get("active_features")
        return scaler, list(features) if features is not None else None

    raise FileNotFoundError(
        "Could not locate scaler metadata (expected scaler.json alongside checkpoints)."
    )


def run_cascade(
    *,
    X_test: torch.Tensor,
    y_cls_test: torch.Tensor,
    y_pos_test: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    binary_path: Path,
    anomaly_path: Path,
    tst_path: Path,
    default_n_classes: int,
    pos_count: int,
    input_channels: int,
) -> PipelineResult:
    total_samples = int(X_test.size(0))
    actual_anomalies = int((y_cls_test != 0).sum().item())

    # ---------- Stage 1: binary anomaly filter ---------- #
    binary_meta = load_classifier_meta(binary_path) or {}
    meta_pos = binary_meta.get("pos_count")
    if meta_pos is not None and int(meta_pos) != pos_count:
        raise ValueError("Binary TCN checkpoint expects a different number of position columns.")
    meta_channels = binary_meta.get("input_channels")
    if meta_channels is not None and int(meta_channels) != input_channels:
        raise ValueError("Binary TCN checkpoint expects a different input channel arrangement.")

    binary_model = load_classifier(
        "tcn_binary",
        binary_path,
        seq_len=X_test.shape[1],
        n_classes=2,
        device=device,
        input_channels=input_channels,
    )
    binary_logits = predict_tcn_binary(
        binary_model,
        X_test,
        device=device,
        pos_count=pos_count,
    )
    binary_probs = torch.softmax(binary_logits, dim=1)
    binary_preds = binary_probs.argmax(1)
    binary_truth = (y_cls_test != 0).to(dtype=torch.long)

    binary_acc = accuracy_score(
        binary_truth.cpu().numpy(),
        binary_preds.cpu().numpy(),
    )
    binary_auc: float | None = None
    if torch.unique(binary_truth).numel() == 2:
        try:
            binary_auc = roc_auc_score(
                binary_truth.cpu().numpy(),
                binary_probs[:, 1].cpu().numpy(),
            )
        except ValueError:  # pragma: no cover - degenerate class distribution
            binary_auc = None

    anomaly_indices = torch.nonzero(binary_preds == 1, as_tuple=True)[0]

    stage1 = Stage1Result(
        accuracy=binary_acc,
        auc=binary_auc,
        predicted_faults=int(anomaly_indices.numel()),
        total_samples=total_samples,
        truth_faults=actual_anomalies,
        predictions=binary_preds,
        probabilities=binary_probs,
    )

    binary_truth_np = binary_truth.cpu().numpy()
    binary_pred_np = binary_preds.cpu().numpy()
    binary_report = classification_report(binary_truth_np, binary_pred_np, digits=3)
    binary_cm = confusion_matrix(binary_truth_np, binary_pred_np)
    ConfusionMatrixDisplay(binary_cm).plot(
        include_values=True, cmap="Blues", colorbar=False
    )
    plt.title("Confusion Matrix – Stage 1 binary filter")
    plt.tight_layout()
    binary_cm_path = out_dir / "confusion_matrix_binary.png"
    plt.savefig(binary_cm_path, dpi=150)
    plt.close()

    # ---------- Stage 2: anomaly-only multi-class TCN ---------- #
    anomaly_meta = load_classifier_meta(anomaly_path)
    if anomaly_meta is None:
        raise ValueError(
            "Anomaly-only TCN checkpoint metadata is required for the orchestrated pipeline."
        )
    meta_pos = anomaly_meta.get("pos_count")
    if meta_pos is not None and int(meta_pos) != pos_count:
        raise ValueError("Anomaly TCN checkpoint expects a different number of position columns.")
    meta_channels = anomaly_meta.get("input_channels")
    if meta_channels is not None and int(meta_channels) != input_channels:
        raise ValueError("Anomaly TCN checkpoint expects a different input channel arrangement.")

    _, _, _, mapping, _ = remap_anomaly_only_targets_with_options(
        X_test,
        y_cls_test,
        y_pos_test,
        anomaly_meta,
        validate_presence=False,
    )
    inv_mapping = {int(v): int(k) for k, v in mapping.items()}
    anomaly_n_classes = len(mapping)

    anomaly_model = load_classifier(
        "tcn",
        anomaly_path,
        seq_len=X_test.shape[1],
        n_classes=anomaly_n_classes,
        device=device,
        input_channels=input_channels,
    )

    stage2_pred_lookup: dict[int, int] = {}
    stage2_accuracy: float | None = None
    stage2_eval_count = 0
    if anomaly_indices.numel() > 0:
        stage2_logits, _ = predict_tcn(
            anomaly_model,
            X_test[anomaly_indices],
            device=device,
            pos_count=pos_count,
        )
        stage2_preds_remap = stage2_logits.argmax(1).cpu()
        stage2_preds_orig = [inv_mapping[int(cls.item())] for cls in stage2_preds_remap]
        stage2_pred_lookup = {
            int(idx.item()): int(stage2_preds_orig[pos])
            for pos, idx in enumerate(anomaly_indices.cpu())
        }

        # Accuracy only on samples whose ground-truth class belongs to the anomaly mapping
        stage2_truth: list[int] = []
        stage2_preds_eval: list[int] = []
        for idx in anomaly_indices.cpu().tolist():
            true_cls = int(y_cls_test[idx].item())
            if true_cls in mapping:
                stage2_truth.append(true_cls)
                stage2_preds_eval.append(stage2_pred_lookup[idx])
        stage2_eval_count = len(stage2_truth)
        if stage2_truth:
            stage2_accuracy = accuracy_score(stage2_truth, stage2_preds_eval)

    stage2 = Stage2Result(
        predictions=stage2_pred_lookup,
        accuracy=stage2_accuracy,
        evaluated_samples=stage2_eval_count,
    )

    # ---------- Stage 3: TST localisation ---------- #
    class_feature = torch.tensor(
        [float(stage2_pred_lookup.get(int(idx.item()), 0)) for idx in anomaly_indices],
        dtype=X_test.dtype,
    ).unsqueeze(1)
    stage3 = _run_tst_localisation(
        class_feature=class_feature,
        candidates=X_test[anomaly_indices],
        tst_path=tst_path,
        device=device,
        default_n_classes=default_n_classes,
        out_dir=out_dir,
        true_positions=y_pos_test[anomaly_indices],
    )

    # ---------- Aggregate predictions ---------- #
    final_preds = torch.zeros_like(y_cls_test)
    for idx, pred_cls in stage2_pred_lookup.items():
        final_preds[idx] = int(pred_cls)

    y_true_np = y_cls_test.cpu().numpy()
    y_pred_np = final_preds.cpu().numpy()

    report = classification_report(y_true_np, y_pred_np, digits=3)
    cm = confusion_matrix(y_true_np, y_pred_np)
    ConfusionMatrixDisplay(cm).plot(include_values=True, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix – Binary➜TCN➜TST pipeline")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix_pipeline.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    auc_str = f"{stage1.auc:.3f}" if stage1.auc is not None else "N/A"

    summary_lines = [
        (
            "Stage 1 – Binary anomaly filter: "
            f"accuracy={stage1.accuracy:.3f}, auc={auc_str}, "
            f"predicted {stage1.predicted_faults}/{stage1.total_samples} traces as faulty ("
            f"ground-truth faults: {stage1.truth_faults})."
        ),
        "Stage 1 – Binary confusion matrix saved to: " + str(binary_cm_path),
        "Stage 1 – Binary classification report:\n" + binary_report,
    ]

    if stage2.predictions:
        acc_str = f"{stage2.accuracy:.3f}" if stage2.accuracy is not None else "N/A"
        summary_lines.append(
            "Stage 2 – Anomaly-only TCN: "
            f"accuracy={acc_str} over {stage2.evaluated_samples} mapped faults; "
            f"issued predictions for {len(stage2.predictions)} traces."
        )
    else:
        summary_lines.append(
            "Stage 2 – Anomaly-only TCN: no anomaly predictions received from the binary filter."
        )

    if stage3.evaluated_samples > 0:
        rmse_str = f"{stage3.rmse:.3f}" if stage3.rmse is not None else "N/A"
        mae_str = f"{stage3.mae:.3f}" if stage3.mae is not None else "N/A"
        med_str = f"{stage3.median_ae:.3f}" if stage3.median_ae is not None else "N/A"
        bias_str = f"{stage3.bias:.3f}" if stage3.bias is not None else "N/A"
        summary_lines.append(
            "Stage 3 – Time-series transformer localisation: "
            f"RMSE={rmse_str} m, MAE={mae_str} m, median |error|={med_str} m, "
            f"bias={bias_str} m over {stage3.evaluated_samples} traces."
        )
        if stage3.plot_paths:
            summary_lines.append(
                "Stage 3 – Visualisations: "
                + ", ".join(f"{name}={path}" for name, path in stage3.plot_paths.items())
            )
    else:
        summary_lines.append(
            "Stage 3 – Time-series transformer localisation: skipped (no anomaly candidates)."
        )

    summary_lines.append("Confusion matrix (rows=true, cols=pred):\n" + np.array2string(cm))
    summary_lines.append("Overall – chained prediction classification report:\n" + report)

    return PipelineResult(
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        summary_lines=summary_lines,
        confusion_matrix=cm,
        confusion_matrix_path=cm_path,
        binary_confusion_matrix=binary_cm,
        binary_confusion_matrix_path=binary_cm_path,
        classification_report=report,
        final_predictions=final_preds,
    )


def run_full_tcn_pipeline(
    *,
    X_test: torch.Tensor,
    y_cls_test: torch.Tensor,
    y_pos_test: torch.Tensor,
    device: torch.device,
    out_dir: Path,
    full_tcn_path: Path,
    tst_path: Path,
    default_n_classes: int,
    pos_count: int,
    input_channels: int,
) -> PipelineResult:
    total_samples = int(X_test.size(0))
    actual_anomalies = int((y_cls_test != 0).sum().item())

    full_meta = load_classifier_meta(full_tcn_path) or {}
    meta_pos = full_meta.get("pos_count")
    if meta_pos is not None and int(meta_pos) != pos_count:
        raise ValueError("Full TCN checkpoint expects a different number of position columns.")
    meta_channels = full_meta.get("input_channels")
    if meta_channels is not None and int(meta_channels) != input_channels:
        raise ValueError("Full TCN checkpoint expects a different input channel arrangement.")
    meta_classes = full_meta.get("n_classes")
    n_classes = int(meta_classes) if meta_classes is not None else default_n_classes

    full_model = load_classifier(
        "tcn",
        full_tcn_path,
        seq_len=X_test.shape[1],
        n_classes=n_classes,
        device=device,
        input_channels=input_channels,
    )

    logits, _ = predict_tcn(
        full_model,
        X_test,
        device=device,
        pos_count=pos_count,
    )
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(1)

    overall_acc = accuracy_score(
        y_cls_test.cpu().numpy(),
        preds.cpu().numpy(),
    )

    stage1 = Stage1Result(
        accuracy=overall_acc,
        auc=None,
        predicted_faults=int((preds != 0).sum().item()),
        total_samples=total_samples,
        truth_faults=actual_anomalies,
        predictions=preds,
        probabilities=probs,
    )

    anomaly_mask = y_cls_test != 0
    anomaly_indices = torch.nonzero(preds != 0, as_tuple=True)[0]
    stage2_lookup = {
        int(idx): int(preds[idx].item())
        for idx in anomaly_indices.cpu().tolist()
    }
    stage2_accuracy: float | None = None
    anomaly_eval = int(anomaly_mask.sum().item())
    if anomaly_eval > 0:
        stage2_accuracy = accuracy_score(
            y_cls_test[anomaly_mask].cpu().numpy(),
            preds[anomaly_mask].cpu().numpy(),
        )

    stage2 = Stage2Result(
        predictions=stage2_lookup,
        accuracy=stage2_accuracy,
        evaluated_samples=anomaly_eval,
    )

    class_feature = preds[anomaly_indices].to(dtype=X_test.dtype).unsqueeze(1)
    stage3 = _run_tst_localisation(
        class_feature=class_feature,
        candidates=X_test[anomaly_indices],
        tst_path=tst_path,
        device=device,
        default_n_classes=default_n_classes,
        out_dir=out_dir,
        true_positions=y_pos_test[anomaly_indices],
    )

    final_preds = preds.clone()

    y_true_np = y_cls_test.cpu().numpy()
    y_pred_np = final_preds.cpu().numpy()

    report = classification_report(y_true_np, y_pred_np, digits=3)
    cm = confusion_matrix(y_true_np, y_pred_np)
    ConfusionMatrixDisplay(cm).plot(include_values=True, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix – Full TCN➜TST pipeline")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix_pipeline.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    summary_lines = [
        (
            "Stage 1 – Full multi-class TCN: "
            f"accuracy={stage1.accuracy:.3f}, predicted {stage1.predicted_faults}/{stage1.total_samples} "
            f"traces as faulty (ground-truth faults: {stage1.truth_faults})."
        )
    ]

    if stage2.evaluated_samples > 0:
        acc_str = f"{stage2.accuracy:.3f}" if stage2.accuracy is not None else "N/A"
        summary_lines.append(
            "Stage 2 – Anomaly subset evaluation: "
            f"accuracy={acc_str} across {stage2.evaluated_samples} ground-truth anomaly traces; "
            f"issued predictions for {len(stage2.predictions)} traces."
        )
    else:
        summary_lines.append(
            "Stage 2 – Anomaly subset evaluation: skipped (no ground-truth anomalies present)."
        )

    if stage3.evaluated_samples > 0:
        rmse_str = f"{stage3.rmse:.3f}" if stage3.rmse is not None else "N/A"
        mae_str = f"{stage3.mae:.3f}" if stage3.mae is not None else "N/A"
        med_str = f"{stage3.median_ae:.3f}" if stage3.median_ae is not None else "N/A"
        bias_str = f"{stage3.bias:.3f}" if stage3.bias is not None else "N/A"
        summary_lines.append(
            "Stage 3 – Time-series transformer localisation: "
            f"RMSE={rmse_str} m, MAE={mae_str} m, median |error|={med_str} m, "
            f"bias={bias_str} m over {stage3.evaluated_samples} traces."
        )
        if stage3.plot_paths:
            summary_lines.append(
                "Stage 3 – Visualisations: "
                + ", ".join(f"{name}={path}" for name, path in stage3.plot_paths.items())
            )
    else:
        summary_lines.append(
            "Stage 3 – Time-series transformer localisation: skipped (no anomaly predictions)."
        )

    summary_lines.append("Confusion matrix (rows=true, cols=pred):\n" + np.array2string(cm))
    summary_lines.append("Overall – chained prediction classification report:\n" + report)

    return PipelineResult(
        stage1=stage1,
        stage2=stage2,
        stage3=stage3,
        summary_lines=summary_lines,
        confusion_matrix=cm,
        confusion_matrix_path=cm_path,
        binary_confusion_matrix=None,
        binary_confusion_matrix_path=None,
        classification_report=report,
        final_predictions=final_preds,
    )


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--data",
    "data_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/OTDR_DATA.csv"),
    show_default=True,
    help="Path to the cleaned OTDR dataset (CSV or Parquet).",
)
@click.option(
    "--binary-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Checkpoint for the binary TCN anomaly detector.",
)
@click.option(
    "--anomaly-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Checkpoint for the anomaly-only multi-class TCN.",
)
@click.option(
    "--tst-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Checkpoint for the TST localisation model.",
)
@click.option(
    "--use-full-tcn",
    is_flag=True,
    help=(
        "Run the full multi-class TCN➜TST pipeline (bypasses the binary and anomaly-only cascade)."
    ),
)
@click.option(
    "--full-tcn-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Checkpoint for the full multi-class TCN (used with --use-full-tcn).",
)
@click.option(
    "--out-dir",
    type=str,
    default="pipeline_eval",
    show_default=True,
    help="Folder under outputs/ where pipeline artifacts will be written.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="cuda | cuda:0 | mps | cpu | leave empty for auto-detect.",
)
@click.option(
    "--use-loss-reflectance",
    is_flag=True,
    help=(
        "Append 'loss' and 'Reflectance' to the measurement vector and load models "
        "trained with those leakage-prone features."
    ),
)
def main(
    data_path: Path,
    binary_path: Path | None,
    anomaly_path: Path | None,
    tst_path: Path | None,
    use_full_tcn: bool,
    full_tcn_path: Path | None,
    out_dir: str,
    device: str | None,
    use_loss_reflectance: bool,
) -> None:
    """Run the inference pipeline (binary➜anomaly➜TST or full TCN➜TST)."""

    out_dir_path = Path("outputs") / out_dir
    out_dir_path.mkdir(parents=True, exist_ok=True)

    feature_suffix = "_lr" if use_loss_reflectance else ""

    df = load_raw_dataframe(data_path)
    _, _, test_df = make_splits(df)

    if full_tcn_path is not None and not use_full_tcn:
        raise click.BadOptionUsage(
            "--full-tcn-path",
            "--full-tcn-path can only be used together with --use-full-tcn.",
        )

    binary_default = Path("models") / (
        f"tcn_binary{feature_suffix}.pt" if feature_suffix else "tcn_binary.pt"
    )
    anomaly_default = Path("models") / (
        f"tcn_anomaly{feature_suffix}.pt" if feature_suffix else "tcn_anomaly.pt"
    )
    tst_default = Path("models") / (
        f"tst{feature_suffix}.pt" if feature_suffix else "tst.pt"
    )
    full_tcn_default = Path("models") / (
        f"tcn_full{feature_suffix}.pt" if feature_suffix else "tcn_full.pt"
    )

    resolved_tst_path = Path(tst_path) if tst_path else tst_default

    if use_full_tcn:
        resolved_full_tcn = Path(full_tcn_path) if full_tcn_path else full_tcn_default
        checkpoint_dirs = [resolved_full_tcn.parent, resolved_tst_path.parent]
    else:
        resolved_binary = Path(binary_path) if binary_path else binary_default
        resolved_anomaly = Path(anomaly_path) if anomaly_path else anomaly_default
        resolved_full_tcn = None
        checkpoint_dirs = [
            resolved_binary.parent,
            resolved_anomaly.parent,
            resolved_tst_path.parent,
        ]

    scaler, feature_names_meta = _load_scaler_metadata(
        checkpoint_dirs=checkpoint_dirs,
        use_loss_reflectance=use_loss_reflectance,
    )

    try:
        requested_cols = measurement_columns(
            test_df,
            include_loss_reflectance=use_loss_reflectance,
        )
    except KeyError as exc:
        raise click.BadOptionUsage("--use-loss-reflectance", str(exc)) from exc

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
                "Requested feature configuration does not match the saved scaler metadata."
            )
    else:
        meas_cols = requested_cols

    if len(meas_cols) != scaler.n_features_in_:
        raise ValueError(
            "Scaler metadata dimensionality does not match selected measurement columns."
        )

    layout = summarise_feature_layout(meas_cols)
    pos_count = int(layout["pos_count"])
    extra_scalar_count = len(layout["extra_features"])
    input_channels = 1 + 1 + extra_scalar_count

    if pos_count <= 0:
        raise ValueError("No positional measurement columns (P*) were detected in the dataset.")

    splits = tensorise_splits(
        test_df,
        test_df,
        test_df,
        scaler,
        measurement_override=meas_cols,
    )
    X_test = splits["test"].X
    y_cls_test = splits["test"].y_class
    y_pos_test = splits["test"].y_pos

    default_n_classes = int(df["Class"].max() + 1)

    if use_full_tcn:
        if not resolved_full_tcn.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_full_tcn}")
        if not resolved_tst_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_tst_path}")
    else:
        if not resolved_binary.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_binary}")
        if not resolved_anomaly.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_anomaly}")
        if not resolved_tst_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {resolved_tst_path}")

    resolved_device = _resolve_device(device)
    click.echo(f"[PIPELINE] Using device: {resolved_device}")

    if use_full_tcn:
        result = run_full_tcn_pipeline(
            X_test=X_test,
            y_cls_test=y_cls_test,
            y_pos_test=y_pos_test,
            device=resolved_device,
            out_dir=out_dir_path,
            full_tcn_path=resolved_full_tcn,
            tst_path=resolved_tst_path,
            default_n_classes=default_n_classes,
            pos_count=pos_count,
            input_channels=input_channels,
        )
    else:
        result = run_cascade(
            X_test=X_test,
            y_cls_test=y_cls_test,
            y_pos_test=y_pos_test,
            device=resolved_device,
            out_dir=out_dir_path,
            binary_path=resolved_binary,
            anomaly_path=resolved_anomaly,
            tst_path=resolved_tst_path,
            default_n_classes=default_n_classes,
            pos_count=pos_count,
            input_channels=input_channels,
        )

    click.echo("\n".join(result.summary_lines))
    click.echo(f"Confusion matrix saved to {result.confusion_matrix_path}")


if __name__ == "__main__":
    main()
