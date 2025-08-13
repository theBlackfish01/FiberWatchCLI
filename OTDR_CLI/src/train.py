from __future__ import annotations

"""End-to-end training script for OTDR models.

Run examples
------------
# Train *only* the GRU Auto-Encoder
python -m src.train --mode gru_ae

# Train *only* the TCN multitask classifier
python -m src.train --mode tcn

# Train *only* the Transformer multitask classifier
python -m src.train --mode tst

# Train *only* the TabNet multitask classifier
python -m src.train --mode tab

# Train all sequentially (GRU-AE ➜ TCN ➜ TST ➜ TabNet)
python -m src.train --mode all
"""

from pathlib import Path
import json
import re
from typing import Tuple

import click
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score, roc_curve

# ---------------------------------------------------------------------------
# Local project imports
# ---------------------------------------------------------------------------

from data_helper import (
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
from model_functions.tcn import OTDR_TCN, TrainConfig as TCNConfig, train_tcn, predict as predict_tcn
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
    rmse = mean_squared_error(y_test_pos.numpy(), pos_hat.numpy())
    print(f"[TCN]    Test Acc={cls_acc:.3f}  RMSE={rmse:.3f}")
    return cls_acc, rmse


# ----------------------------- TST eval -------------------------------------#

def _evaluate_tst(
    tst: TimeSeriesTransformer,
    X_test: torch.Tensor,
    y_test_cls: torch.Tensor,
    y_test_pos: torch.Tensor,
) -> Tuple[float, float]:
    logits, pos_hat = predict_tst(tst, X_test)
    cls_acc = accuracy_score(y_test_cls.numpy(), logits.argmax(1).numpy())
    rmse = mean_squared_error(y_test_pos.numpy(), pos_hat.numpy())
    print(f"[TST]    Test Acc={cls_acc:.3f}  RMSE={rmse:.3f}")
    return cls_acc, rmse

# ----------------------------- TabNet eval ----------------------------------#

def _evaluate_tabnet(
    tabnet: OTDR_TabNet,
    X_test: torch.Tensor,
    y_test_cls: torch.Tensor,
    y_test_pos: torch.Tensor,
) -> Tuple[float, float]:
    logits, pos_hat = predict_tabnet(tabnet, X_test)
    cls_acc = accuracy_score(y_test_cls.numpy(), logits.argmax(1).numpy())
    rmse = mean_squared_error(y_test_pos.numpy(), pos_hat.numpy())
    print(f"[TabNet] Test Acc={cls_acc:.3f}  RMSE={rmse:.3f}")
    return cls_acc, rmse


# ---------------------------------------------------------------------------
# Main driver (Click CLI)
# ---------------------------------------------------------------------------

@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--mode",
    type=click.Choice(["gru_ae", "tcn", "tst", "tab", "all"], case_sensitive=False),
    required=True,
    help="Which component(s) to train.",
)
@click.option(
    "--data", "data_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/OTDR_data.csv"),
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
    help="cuda | cpu | leave empty for auto-detect.",
)
def main(mode, data_path, out_dir, device) -> None:  # noqa: C901 – single-entry script
    out_dir = Path(out_dir)
    _ensure_dir(out_dir)

    # ----------------------------- Data -----------------------------------#
    df = load_raw_dataframe(data_path)
    train_df, val_df, test_df = make_splits(df)

    # measurement columns: P1..Pn + SNR
    measurements = [c for c in train_df.columns if re.fullmatch(r"P\d+", c)] + ["SNR"]
    scaler = fit_scaler(train_df[measurements].values.astype(np.float32))
    splits = tensorise_splits(train_df, val_df, test_df, scaler)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None else torch.device(device)
    )
    print(f"[INFO] Using device: {device}")

    # ----------------------------- GRU-AE ----------------------------------#
    if mode in {"gru_ae", "all"}:
        NORMAL = 0
        norm_idx = (splits["train"].y_class == NORMAL).nonzero(as_tuple=True)[0]
        X_norm = splits["train"].X[norm_idx]

        ae = VectorGRUAE(feat_dim=X_norm.shape[1])
        ae_cfg = AEConfig(save_path=out_dir / "gru_ae.pt", device=device)
        ae, thresh = train_gru_ae(ae, X_norm, splits["val"].X, cfg=ae_cfg)
        print(f"[GRU-AE] Threshold={thresh:.5f}")
        _evaluate_gru_ae(ae, thresh, splits["test"].X, splits["test"].y_class)

        # Save scaler stats + threshold for inference
        meta = {
            "threshold": thresh,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
        }
        with open(out_dir / "gru_ae.json", "w") as fp:
            json.dump(meta, fp, indent=2)

    # ----------------------------- TCN -------------------------------------#
    if mode in {"tcn", "all"}:
        n_classes = int(df["Class"].max() + 1)
        tcn = OTDR_TCN(n_classes=n_classes)
        tcn_cfg = TCNConfig(save_path=out_dir / "tcn.pt", device=device)
        tcn = train_tcn(
            tcn,
            splits["train"].X,
            splits["train"].y_class,
            splits["train"].y_pos,
            splits["val"].X,
            splits["val"].y_class,
            splits["val"].y_pos,
            cfg=tcn_cfg,
        )
        _evaluate_tcn(tcn, splits["test"].X, splits["test"].y_class, splits["test"].y_pos)

    # ----------------------------- TST -------------------------------------#
    if mode in {"tst", "all"}:
        n_classes = int(df["Class"].max() + 1)
        tst = TimeSeriesTransformer(seq_len=splits["train"].X.shape[1], n_classes=n_classes)
        tst_cfg = TSTConfig(save_path=out_dir / "tst.pt", device=device)
        tst = train_tst(
            tst,
            splits["train"].X,
            splits["train"].y_class,
            splits["train"].y_pos,
            splits["val"].X,
            splits["val"].y_class,
            splits["val"].y_pos,
            cfg=tst_cfg,
        )
        _evaluate_tst(tst, splits["test"].X, splits["test"].y_class, splits["test"].y_pos)

    # ----------------------------- TabNet ----------------------------------#
    if mode in {"tab", "all"}:
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
