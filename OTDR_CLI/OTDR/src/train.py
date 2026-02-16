from __future__ import annotations

"""
Training script for OTDR models.

------------
# Train GRUAE
python -m src.train --mode gru_ae

# Train TCN 
python -m src.train --mode tcn

# Train all 
python -m src.train --mode all

# Train Siamese
python -m src.train --mode siamese
"""

from pathlib import Path
from time import perf_counter
from dataclasses import fields, is_dataclass
import json
from typing import Tuple, Optional

import click
import numpy as np
import torch
from sklearn.metrics import accuracy_score, root_mean_squared_error, roc_auc_score, roc_curve, classification_report

# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------

from data_helper import (
    SplitTensors,
    load_raw_dataframe,
    make_splits,
    fit_scaler,
    tensorise_splits,
    measurement_columns,
    summarise_feature_layout,
    build_feature_config,
)

from model_functions.gruae import (
    VectorGRUAE,
    TrainConfig as AEConfig,
    train_gru_ae,
    reconstruction_error,
)
from model_functions.tcn import (
    OTDR_TCN,
    TrainConfig as TCNConfig,
    train_tcn,
    predict as predict_tcn
)
from model_functions.tcn_binary import (
    OTDR_TCNBinary,
    TrainConfig as TCNBinaryConfig,
    train_tcn_binary,
    predict as predict_tcn_binary,
)
from model_functions.tst import (
    TimeSeriesTransformer,
    TrainConfig as TSTConfig,
    train_tst,
    predict as predict_tst,
)
from model_functions.siamese import (
    train_siamese,
    TrainConfig as SiameseConfig,
)

from pathlib import Path
from time import perf_counter
from dataclasses import fields, is_dataclass
import json
from typing import Tuple, Optional

import click
import numpy as np
import torch
from sklearn.metrics import accuracy_score, root_mean_squared_error, roc_auc_score, roc_curve, classification_report

# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------

from data_helper import (
    SplitTensors,
    load_raw_dataframe,
    make_splits,
    fit_scaler,
    tensorise_splits,
    measurement_columns,
    summarise_feature_layout,
    build_feature_config,
)

from model_functions.gruae import (
    VectorGRUAE,
    TrainConfig as AEConfig,
    train_gru_ae,
    reconstruction_error,
)
from model_functions.tcn import (
    OTDR_TCN,
    TrainConfig as TCNConfig,
    train_tcn,
    predict as predict_tcn
)
from model_functions.tcn_binary import (
    OTDR_TCNBinary,
    TrainConfig as TCNBinaryConfig,
    train_tcn_binary,
    predict as predict_tcn_binary,
)
from model_functions.tst import (
    TimeSeriesTransformer,
    TrainConfig as TSTConfig,
    train_tst,
    predict as predict_tst,
)

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)  # noqa: T201


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _config_to_dict(cfg) -> dict:
    """Serialise dataclass config objects into JSON-friendly dictionaries."""

    if not is_dataclass(cfg):
        return {}

    serialised: dict[str, object] = {}
    for f in fields(cfg):
        value = getattr(cfg, f.name)
        if isinstance(value, Path):
            serialised[f.name] = str(value)
        elif isinstance(value, torch.device):
            serialised[f.name] = str(value)
        else:
            serialised[f.name] = value
    return serialised


def _resolve_device(requested: Optional[str]) -> torch.device:
    """
    Choose an appropriate torch.device.
    Priority: requested (if available) -> CUDA -> MPS -> CPU.
    """
    # Normalize
    if requested:
        req = requested.lower()
        if req.startswith("cuda"):
            if torch.cuda.is_available():
                return torch.device(req if ":" in req else "cuda:0")
            else:
                print("[WARN] Requested CUDA, but CUDA is not available. Falling back to auto.")
        elif req == "mps":
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                print("[WARN] Requested MPS, but MPS is not available. Falling back to auto.")
        elif req == "cpu":
            return torch.device("cpu")
        else:
            # Try to construct and hope for the best
            try:
                dev = torch.device(req)
                return dev
            except Exception:
                print(f"[WARN] Unrecognized device '{requested}'. Falling back to auto.")

    # Auto
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _print_device_info(device: torch.device) -> None:
    if device.type == "cuda":
        idx = torch.cuda.current_device()
        name = torch.cuda.get_device_name(idx)
        print(f"[INFO] Using device: cuda:{idx} ({name})")
    elif device.type == "mps":
        print("[INFO] Using device: mps (Apple Silicon)")
    else:
        print("[INFO] Using device: cpu")


# ----------------------------- GRU-AE eval ----------------------------------#

def _evaluate_gru_ae(
        ae: VectorGRUAE,
        threshold: float,
        X_test: torch.Tensor,
        y_test_cls: torch.Tensor,
) -> Tuple[float, float]:
    errs = reconstruction_error(ae, X_test)
    y_true = (y_test_cls != 0).numpy().astype(int)
    y_score = errs.numpy()
    auc = roc_auc_score(y_true, y_score)
    fpr, tpr, _ = roc_curve(y_true, y_score)
    acc = accuracy_score(y_true, (y_score > threshold).astype(int))
    print(f"[GRU-AE] Test AUC={auc:.3f}  Acc@thr={acc:.3f}")
    return auc, acc


# ----------------------------- TCN eval -------------------------------------#

def _evaluate_tcn(
        tcn: OTDR_TCN,
        X_test: torch.Tensor,
        y_test_cls: torch.Tensor,
        y_test_pos: torch.Tensor,
        *,
        pos_count: int,
) -> Tuple[float, float]:
    logits, pos_hat = predict_tcn(tcn, X_test, pos_count=pos_count)
    preds = logits.argmax(dim=1).detach().cpu().numpy()
    y_true = y_test_cls.detach().cpu().numpy()
    cls_acc = accuracy_score(y_true, preds)
    rmse = root_mean_squared_error(
        y_test_pos.numpy().ravel(), pos_hat.numpy()
    )
    print(f"[TCN]    Test Acc={cls_acc:.3f}  RMSE={rmse:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, preds))
    return cls_acc, rmse


def _evaluate_tcn_binary(
        tcn: OTDR_TCNBinary,
        X_test: torch.Tensor,
        y_test_cls: torch.Tensor,
        *,
        pos_count: int,
) -> Tuple[float, float]:
    logits = predict_tcn_binary(tcn, X_test, pos_count=pos_count)
    preds = logits.argmax(1).numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_true = (y_test_cls != 0).numpy().astype(int)
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    print(f"[TCN-B]  Test Acc={acc:.3f}  AUC={auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_true, preds))
    return acc, auc


# ----------------------------- TST eval -------------------------------------#

def _evaluate_tst(
        tst: TimeSeriesTransformer,
        X_test: torch.Tensor,
        y_test_cls: torch.Tensor,
        y_test_pos: torch.Tensor,
) -> float:
    pos_hat = predict_tst(tst, X_test)
    y_true_pos = y_test_pos.numpy().ravel()
    y_pred_pos = pos_hat.numpy().ravel()
    rmse = root_mean_squared_error(y_true_pos, y_pred_pos)
    classes = torch.unique(y_test_cls).cpu().numpy()
    cls_summary = ", ".join(str(int(c)) for c in classes)
    print(f"[TST]    Test RMSE={rmse:.3f} | Classes={cls_summary}")
    per_class = []
    class_labels = y_test_cls.cpu().numpy()
    for cls in classes:
        mask = class_labels == cls
        if np.any(mask):
            cls_rmse = root_mean_squared_error(
                y_true_pos[mask], y_pred_pos[mask]
            )
            per_class.append(f"{int(cls)}: {cls_rmse:.3f}")
    if per_class:
        print("Per-class RMSEs: " + ", ".join(per_class))

    return rmse


def _faulty_only(split: SplitTensors, normal_label: int = 0) -> SplitTensors:
    """Return tensors containing only faulty samples (label != normal_label)."""

    mask = split.y_class != normal_label
    if mask.sum().item() == 0:
        raise ValueError(
            "No faulty samples available – cannot train/evaluate the TST on an empty set."
        )
    return SplitTensors(
        X=split.X[mask],
        y_class=split.y_class[mask],
        y_pos=split.y_pos[mask],
    )


def _binary_labels(split: SplitTensors, normal_label: int = 0) -> SplitTensors:
    """Map the multiclass labels to binary normal(0)/anomaly(1)."""

    binary = (split.y_class != normal_label).to(dtype=torch.long)
    return SplitTensors(X=split.X, y_class=binary, y_pos=split.y_pos)


def _faulty_only_relabel(
        split: SplitTensors,
        *,
        normal_label: int = 0,
        classes: torch.Tensor | None = None,
) -> tuple[SplitTensors, torch.Tensor]:
    """Faulty-only tensors with class labels remapped to a contiguous range."""

    mask = split.y_class != normal_label
    if mask.sum().item() == 0:
        raise ValueError(
            "No faulty samples available – cannot train/evaluate on an empty set."
        )

    y_cls = split.y_class[mask]
    if classes is None:
        classes = torch.unique(y_cls, sorted=True)
    else:
        # Ensure ``classes`` tensor lives on CPU for indexing
        classes = classes.to(y_cls.device)

    remapped = torch.searchsorted(classes, y_cls)
    if torch.any(remapped >= classes.numel()):
        raise ValueError("Class mapping produced out-of-range indices.")
    if not torch.all(classes[remapped] == y_cls):
        missing = torch.unique(y_cls[classes[remapped] != y_cls])
        raise ValueError(
            "Encountered class labels without a defined mapping: "
            f"{missing.tolist()}"
        )

    return (
        SplitTensors(
            X=split.X[mask],
            y_class=remapped.to(dtype=torch.long),
            y_pos=split.y_pos[mask],
        ),
        classes.detach().clone(),
    )


def _with_class_feature(split: SplitTensors) -> SplitTensors:
    """Append the class label as an explicit feature for localisation models."""

    class_column = split.y_class.to(dtype=split.X.dtype).unsqueeze(1)
    return SplitTensors(
        X=torch.cat([class_column, split.X], dim=1),
        y_class=split.y_class,
        y_pos=split.y_pos,
    )


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--mode",
    type=click.Choice(["gru_ae", "tcn", "binary", "tst", "all", "siamese"], case_sensitive=False),
    required=True,
    help="Which model(s) to train.",
)
@click.option(
    "--data", "data_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=True),
    default="data/processed",
    help="Path to dataset (CSV file or Parquet directory).",
)
@click.option(
    "--out-dir",
    type=click.Path(file_okay=False, dir_okay=True),
    default="models",
    help="Directory for saved weights & metadata.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="cuda | cuda:0 | mps | cpu | leave empty for auto-detect.",
)
@click.option(
    "--train-noise-level",
    type=float,
    default=0.0,
    help="Standard deviation of Gaussian noise added to features.",
)
@click.option(
    "--tcn-anomaly-only/--tcn-all-data",
    "tcn_anomaly_only",
    default=False,
    help="Train the TCN using only anomaly samples (Class != 0).",
)
@click.option(
    "--use-loss-reflectance",
    is_flag=True,
    default=False,
    help="Include 'loss' and 'Reflectance' columns in the input features.",
)
@click.option("--epochs", default=None, type=int, help="Override default epochs (Siamese only).")
@click.option("--batch-size", default=None, type=int, help="Override batch size (Siamese only).")
@click.option("--lr", default=None, type=float, help="Override learning rate (Siamese only).")
def main(
    mode: str,
    data_path: str,
    out_dir: str,
    device: str | None,
    train_noise_level: float,
    tcn_anomaly_only: bool,
    use_loss_reflectance: bool,
    epochs: int | None,
    batch_size: int | None,
    lr: float | None,
):
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)
    
    processed_path = Path(data_path)
    
    # ----------------------------- Device ---------------------------------#

    # ----------------------------- Data -----------------------------------#
    df = load_raw_dataframe(data_path)
    train_df, val_df, test_df = make_splits(df)

    # measurement columns: P1..Pn + SNR
    measurements = measurement_columns(
        train_df,
        include_loss_reflectance=use_loss_reflectance,
    )
    layout = summarise_feature_layout(measurements)
    pos_count = int(layout["pos_count"])
    extra_scalar_count = int(layout["extra_scalar_count"])
    feature_suffix = "_lr" if use_loss_reflectance else ""
    feature_config = build_feature_config(
        measurements,
        use_loss_reflectance=use_loss_reflectance
    )
    scaler = fit_scaler(train_df[measurements].values.astype(np.float32))
    splits = tensorise_splits(
        train_df,
        val_df,
        test_df,
        scaler,
        measurement_override=measurements,
    )

    if train_noise_level > 0:
        splits = {
            name: SplitTensors(
                X=split.X + torch.randn_like(split.X) * float(train_noise_level),
                y_class=split.y_class,
                y_pos=split.y_pos,
            )
            for name, split in splits.items()
        }
        print(
            f"[INFO] Added Gaussian noise to dataset splits with σ={train_noise_level:.4f}."
        )

    # ----------------------------- Device ---------------------------------#
    device = _resolve_device(device)
    _print_device_info(device)

    # Persist the training scaler separately for evaluation / future runs
    scaler_meta = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "n_features_in": int(scaler.mean_.shape[0]),
        "feature_names": measurements,
        "source_data": str(data_path),
        "active_features": measurements,
        "use_loss_reflectance": bool(use_loss_reflectance),
        "feature_config": feature_config,
        "feature_config_signature": feature_config["signature"],
    }
    scaler_name = f"scaler{feature_suffix}.json" if feature_suffix else "scaler.json"
    with open(out_dir / scaler_name, "w") as fp:
        json.dump(scaler_meta, fp, indent=2)

    # ----------------------------- GRU-AE ----------------------------------#
    if mode in {"gru_ae", "all"}:
        NORMAL = 0
        norm_idx = (splits["train"].y_class == NORMAL).nonzero(as_tuple=True)[0]
        X_norm = splits["train"].X[norm_idx]

        ae = VectorGRUAE(feat_dim=X_norm.shape[1])
        ae_path = out_dir / f"gru_ae{feature_suffix}.pt" if feature_suffix else out_dir / "gru_ae.pt"
        ae_cfg = AEConfig(save_path=ae_path, device=device)
        val_norm_idx = (splits["val"].y_class == NORMAL).nonzero(as_tuple=True)[0]
        X_val_norm = splits["val"].X[val_norm_idx]
        start_time = perf_counter()
        ae, thresh = train_gru_ae(ae, X_norm, X_val_norm, cfg=ae_cfg)
        print(f"[GRU-AE] Training completed in {perf_counter() - start_time:.2f}s")

        print(f"[GRU-AE] Threshold={thresh:.5f}")
        eval_start = perf_counter()
        _evaluate_gru_ae(ae, thresh, splits["test"].X, splits["test"].y_class)
        print(f"[GRU-AE] Evaluation completed in {perf_counter() - eval_start:.2f}s")

        gru_meta = {
            "threshold": float(thresh),
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "feature_names": measurements,
            "source_data": str(data_path),
            "model_kwargs": {
                "feat_dim": int(ae.feat_dim),
                "hidden": int(ae.hidden),
                "latent": int(ae.latent),
                "layers": int(ae.layers),
                "bidir": bool(ae.bidir),
                "dropout": float(ae.dropout_p),
            },
            "train_config": {
                "epochs": ae_cfg.epochs,
                "batch_size": ae_cfg.batch_size,
                "lr": ae_cfg.lr,
                "patience": ae_cfg.patience,
                "lr_patience": ae_cfg.lr_patience,
                "lr_factor": ae_cfg.lr_factor,
                "min_lr": ae_cfg.min_lr,
                "quantile": ae_cfg.quantile,
                "weight_decay": ae_cfg.weight_decay,
                "grad_clip": ae_cfg.grad_clip,
            },
        }
        gru_meta["active_features"] = measurements
        gru_meta["use_loss_reflectance"] = bool(use_loss_reflectance)
        gru_meta["feature_suffix"] = feature_suffix
        gru_meta["feature_config"] = feature_config
        gru_meta["feature_config_signature"] = feature_config["signature"]
        with open(ae_path.with_suffix(".json"), "w") as fp:
            json.dump(gru_meta, fp, indent=2)

    # ----------------------------- TCN -------------------------------------#
    if mode in {"tcn", "all"}:
        anomaly_classes_list: list[int] | None = None
        if tcn_anomaly_only:
            train_faulty, anomaly_classes = _faulty_only_relabel(
                splits["train"], normal_label=0
            )
            val_faulty, _ = _faulty_only_relabel(
                splits["val"], normal_label=0, classes=anomaly_classes
            )
            test_faulty, _ = _faulty_only_relabel(
                splits["test"], normal_label=0, classes=anomaly_classes
            )
            anomaly_classes_list = [int(c) for c in anomaly_classes.tolist()]
            class_map = ", ".join(
                f"{int(orig)}→{idx}" for idx, orig in enumerate(anomaly_classes.tolist())
            )
            print(
                "[TCN] Training with anomaly-only data. Class remapping: "
                f"{class_map}"
            )
            train_split = train_faulty
            val_split = val_faulty
            test_split = test_faulty
        else:
            train_split = splits["train"]
            val_split = splits["val"]
            test_split = splits["test"]

        n_classes = int(train_split.y_class.max().item() + 1)
        in_channels = 1 + 1 + extra_scalar_count
        tcn = OTDR_TCN(n_classes=n_classes, in_ch=in_channels)
        tcn_name = "tcn_anomaly" if tcn_anomaly_only else "tcn_full"
        tcn_name = f"{tcn_name}{feature_suffix}" if feature_suffix else tcn_name
        tcn_save_path = out_dir / f"{tcn_name}.pt"
        tcn_cfg = TCNConfig(save_path=tcn_save_path, device=device, pos_count=pos_count)
        start_time = perf_counter()
        tcn = train_tcn(
            tcn,
            train_split.X,
            train_split.y_class,
            train_split.y_pos,
            val_split.X,
            val_split.y_class,
            val_split.y_pos,
            cfg=tcn_cfg,
        )
        eval_start = perf_counter()
        print(f"[TCN]    Training completed in {perf_counter() - start_time:.2f}s")
        _evaluate_tcn(
            tcn,
            test_split.X,
            test_split.y_class,
            test_split.y_pos,
            pos_count=pos_count,
        )
        print(f"[TCN]    Evaluation completed in {perf_counter() - eval_start:.2f}s")

        class_labels = sorted(int(c) for c in torch.unique(train_split.y_class).tolist())
        tcn_meta = {
            "variant": "anomaly_only" if tcn_anomaly_only else "full",
            "feature_names": measurements,
            "source_data": str(data_path),
            "normal_label": 0,
            "n_classes": int(n_classes),
            "class_labels": class_labels,
            "train_config": _config_to_dict(tcn_cfg),
            "active_features": measurements,
            "use_loss_reflectance": bool(use_loss_reflectance),
            "pos_count": pos_count,
            "extra_scalar_count": extra_scalar_count,
            "input_channels": in_channels,
            "feature_suffix": feature_suffix,
            "feature_config": feature_config,
            "feature_config_signature": feature_config["signature"],
        }
        if tcn_anomaly_only:
            if not anomaly_classes_list:
                raise RuntimeError("Anomaly class mapping missing for metadata emission.")
            tcn_meta["original_classes"] = anomaly_classes_list
            tcn_meta["class_index_map"] = {
                str(orig): idx for idx, orig in enumerate(anomaly_classes_list)
            }
        else:
            tcn_meta["original_classes"] = class_labels
            tcn_meta["class_index_map"] = {str(cls): cls for cls in class_labels}

        with open(tcn_save_path.with_suffix(".json"), "w") as fp:
            json.dump(tcn_meta, fp, indent=2)

    if mode == "tcn_binary":
        train_bin = _binary_labels(splits["train"])
        val_bin = _binary_labels(splits["val"])
        test_bin = _binary_labels(splits["test"])

        in_channels = 1 + 1 + extra_scalar_count
        tcn_binary = OTDR_TCNBinary(in_ch=in_channels)
        tcn_bin_name = "tcn_binary"
        tcn_bin_name = f"{tcn_bin_name}{feature_suffix}" if feature_suffix else tcn_bin_name
        tcn_binary_path = out_dir / f"{tcn_bin_name}.pt"
        tcn_bin_cfg = TCNBinaryConfig(
            save_path=tcn_binary_path,
            device=device,
            pos_count=pos_count,
        )
        start_time = perf_counter()
        tcn_binary = train_tcn_binary(
            tcn_binary,
            train_bin.X,
            train_bin.y_class,
            val_bin.X,
            val_bin.y_class,
            cfg=tcn_bin_cfg,
        )
        print(f"[TCN-B]  Training completed in {perf_counter() - start_time:.2f}s")
        eval_start = perf_counter()
        _evaluate_tcn_binary(
            tcn_binary,
            test_bin.X,
            test_bin.y_class,
            pos_count=pos_count,
        )
        print(f"[TCN-B]  Evaluation completed in {perf_counter() - eval_start:.2f}s")

        tcn_binary_meta = {
            "variant": "binary",
            "feature_names": measurements,
            "source_data": str(data_path),
            "class_labels": [0, 1],
            "normal_label": 0,
            "positive_label": 1,
            "train_config": _config_to_dict(tcn_bin_cfg),
            "active_features": measurements,
            "use_loss_reflectance": bool(use_loss_reflectance),
            "pos_count": pos_count,
            "extra_scalar_count": extra_scalar_count,
            "input_channels": in_channels,
            "feature_suffix": feature_suffix,
            "feature_config": feature_config,
            "feature_config_signature": feature_config["signature"],
        }
        with open(tcn_binary_path.with_suffix(".json"), "w") as fp:
            json.dump(tcn_binary_meta, fp, indent=2)

    # ----------------------------- Siamese ---------------------------------#
    if mode == "siamese":
        print("\n=== Training Siamese Network ===")
        cfg = SiameseConfig()
        if epochs:
            cfg.epochs = epochs
        if batch_size:
            cfg.batch_size = batch_size
        if lr:
            cfg.lr = lr
        if device:
            cfg.device = device
            
        cfg.use_loss_reflectance = use_loss_reflectance
        cfg.noise_level = float(train_noise_level)
            
        train_siamese(processed_path, cfg, OTDR_TCN)

    # ----------------------------- TST -------------------------------------#
    if mode in {"tst", "all"}:
        fault_train = _with_class_feature(_faulty_only(splits["train"]))
        fault_val = _with_class_feature(_faulty_only(splits["val"]))
        fault_test = _with_class_feature(_faulty_only(splits["test"]))

        tst = TimeSeriesTransformer(seq_len=fault_train.X.shape[1])
        tst_name = f"tst{feature_suffix}" if feature_suffix else "tst"
        tst_cfg = TSTConfig(
            save_path=out_dir / f"{tst_name}.pt",
            device=device,
        )
        start_time = perf_counter()
        tst = train_tst(
            model=tst,
            train_tensor=fault_train.X,
            train_y_cls=fault_train.y_class,
            train_y_pos=fault_train.y_pos,
            val_tensor=fault_val.X,
            val_y_cls=fault_val.y_class,
            val_y_pos=fault_val.y_pos,
            cfg=tst_cfg,
        )
        print(f"[TST]    Training completed in {perf_counter() - start_time:.2f}s")
        eval_start = perf_counter()
        _evaluate_tst(tst, fault_test.X, fault_test.y_class, fault_test.y_pos)
        print(f"[TST]    Evaluation completed in {perf_counter() - eval_start:.2f}s")


if __name__ == "__main__":
    main()
