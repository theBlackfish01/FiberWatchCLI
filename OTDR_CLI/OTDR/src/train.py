from __future__ import annotations

"""
Training script for OTDR models.

Run examples
------------
# Train GRUAE
python -m src.train --mode gru_ae

# Train TCN 
python -m src.train --mode tcn

# Train all 
python -m src.train --mode all
"""

from pathlib import Path
import json
import re
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
from model_functions.tabnet import (
    OTDR_TabNet,
    TrainConfig as TabNetConfig,
    train_tabnet,
    predict as predict_tabnet,
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
) -> Tuple[float, float]:
    logits, pos_hat = predict_tcn(tcn, X_test)
    cls_acc = accuracy_score(y_test_cls.numpy(), logits.argmax(1).numpy())
    rmse = root_mean_squared_error(
        y_test_pos.numpy().ravel(), pos_hat.numpy()
    )
    print(f"[TCN]    Test Acc={cls_acc:.3f}  RMSE={rmse:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_cls.cpu().numpy(), pos_hat.numpy().round()))
    return cls_acc, rmse


def _evaluate_tcn_binary(
        tcn: OTDR_TCNBinary,
        X_test: torch.Tensor,
        y_test_cls: torch.Tensor,
) -> Tuple[float, float]:
    logits = predict_tcn_binary(tcn, X_test)
    preds = logits.argmax(1).numpy()
    probs = torch.softmax(logits, dim=1)[:, 1].numpy()
    y_true = y_test_cls.numpy()
    acc = accuracy_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    print(f"[TCN-B]  Test Acc={acc:.3f}  AUC={auc:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test_cls.numpy(), preds))
    return acc, auc


# ----------------------------- TST eval -------------------------------------#

def _evaluate_tst(
        tst: TimeSeriesTransformer,
        X_test: torch.Tensor,
        y_test_cls: torch.Tensor,
        y_test_pos: torch.Tensor,
) -> float:
    pos_hat = predict_tst(tst, X_test)
    rmse = root_mean_squared_error(
        y_test_pos.numpy().ravel(), pos_hat.numpy()
    )
    classes = torch.unique(y_test_cls).cpu().numpy()
    cls_summary = ", ".join(str(int(c)) for c in classes)
    print(f"[TST]    Test RMSE={rmse:.3f} | Classes={cls_summary}")
    print("\nClassification Report:")
    print(classification_report(y_test_cls.cpu().numpy(), pos_hat.numpy().round()))

    return rmse


# ----------------------------- TabNet eval ----------------------------------#

def _evaluate_tabnet(
        tabnet: OTDR_TabNet,
        X_test: torch.Tensor,
        y_test_cls: torch.Tensor,
        y_test_pos: torch.Tensor,
) -> Tuple[float, float]:
    logits, pos_hat = predict_tabnet(tabnet, X_test)
    cls_acc = accuracy_score(y_test_cls.numpy(), logits.argmax(1).numpy())
    rmse = root_mean_squared_error(
        y_test_pos.numpy().ravel(), pos_hat.numpy()
    )
    print(f"[TabNet] Test Acc={cls_acc:.3f}  RMSE={rmse:.3f}")
    return cls_acc, rmse


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

@click.command(
    context_settings=dict(help_option_names=["-h", "--help"])
)
@click.option(
    "--mode",
    type=click.Choice(["gru_ae", "tcn", "tcn_binary", "tst", "tab", "all"], case_sensitive=False),
    required=True,
    help="Which component(s) to train.",
)
@click.option(
    "--data", "data_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/OTDR_DATA.csv"),
    show_default=True,
    help="Path to cleaned OTDR dataset (CSV or Parquet).",
)
@click.option(
    "--out-dir",
    type=str,
    default="models",
    show_default=True,
    help="Directory for saved weights & metadata.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="cuda | cuda:0 | mps | cpu | leave empty for auto-detect.",
)
@click.option(
    "--tcn-anomaly-only/--tcn-all-data",
    "tcn_anomaly_only",
    default=False,
    help="Train the TCN using only anomaly samples (Class != 0).",
)
def main(mode, data_path, out_dir, device, tcn_anomaly_only) -> None:
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # ----------------------------- Data -----------------------------------#
    df = load_raw_dataframe(data_path)
    train_df, val_df, test_df = make_splits(df)

    # measurement columns: P1..Pn + SNR
    measurements = ["SNR"] + [c for c in train_df.columns if re.fullmatch(r"P\d+", c)]
    scaler = fit_scaler(train_df[measurements].values.astype(np.float32))
    splits = tensorise_splits(train_df, val_df, test_df, scaler)

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
    }
    with open(out_dir / "scaler.json", "w") as fp:
        json.dump(scaler_meta, fp, indent=2)

    # ----------------------------- GRU-AE ----------------------------------#
    if mode in {"gru_ae", "all"}:
        NORMAL = 0
        norm_idx = (splits["train"].y_class == NORMAL).nonzero(as_tuple=True)[0]
        X_norm = splits["train"].X[norm_idx]

        ae = VectorGRUAE(feat_dim=X_norm.shape[1])
        ae_cfg = AEConfig(save_path=out_dir / "gru_ae.pt", device=device)
        val_norm_idx = (splits["val"].y_class == NORMAL).nonzero(as_tuple=True)[0]
        X_val_norm = splits["val"].X[val_norm_idx]
        ae, thresh = train_gru_ae(ae, X_norm, X_val_norm, cfg=ae_cfg)

        print(f"[GRU-AE] Threshold={thresh:.5f}")
        _evaluate_gru_ae(ae, thresh, splits["test"].X, splits["test"].y_class)

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
        with open(out_dir / "gru_ae.json", "w") as fp:
            json.dump(gru_meta, fp, indent=2)

    # ----------------------------- TCN -------------------------------------#
    if mode in {"tcn", "all"}:
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
        tcn = OTDR_TCN(n_classes=n_classes)
        tcn_cfg = TCNConfig(save_path=out_dir / "tcn.pt", device=device)
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
        _evaluate_tcn(tcn, test_split.X, test_split.y_class, test_split.y_pos)

    if mode == "tcn_binary":
        train_bin = _binary_labels(splits["train"])
        val_bin = _binary_labels(splits["val"])
        test_bin = _binary_labels(splits["test"])

        tcn_binary = OTDR_TCNBinary()
        tcn_bin_cfg = TCNBinaryConfig(save_path=out_dir / "tcn_binary.pt", device=device)
        tcn_binary = train_tcn_binary(
            tcn_binary,
            train_bin.X,
            train_bin.y_class,
            val_bin.X,
            val_bin.y_class,
            cfg=tcn_bin_cfg,
        )
        _evaluate_tcn_binary(tcn_binary, test_bin.X, test_bin.y_class)

    # ----------------------------- TST -------------------------------------#
    if mode in {"tst", "all"}:
        fault_train = _with_class_feature(_faulty_only(splits["train"]))
        fault_val = _with_class_feature(_faulty_only(splits["val"]))
        fault_test = _with_class_feature(_faulty_only(splits["test"]))

        tst = TimeSeriesTransformer(seq_len=fault_train.X.shape[1])
        tst_cfg = TSTConfig(save_path=out_dir / "tst.pt", device=device)
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
        _evaluate_tst(tst, fault_test.X, fault_test.y_class, fault_test.y_pos)

    # ----------------------------- TabNet ----------------------------------#
    if mode in {"tab"}:
        n_classes = int(df["Class"].max() + 1)
        tabnet = OTDR_TabNet(n_classes=n_classes)
        tabnet_cfg = TabNetConfig(save_path=out_dir / "tabnet.pt", device=device)
        tabnet = train_tabnet(
            tabnet,
            splits["train"].X,
            splits["train"].y_class,
            splits["train"].y_pos,
            splits["val"].X,
            splits["val"].y_class,
            splits["val"].y_pos,
            cfg=tabnet_cfg,
        )
        _evaluate_tabnet(tabnet, splits["test"].X, splits["test"].y_class, splits["test"].y_pos)


if __name__ == "__main__":
    main()
