from __future__ import annotations
"""
Phi-OTDR training CLI (CNN/TCN) with quick test evaluation + confusion matrix.

Usage (examples):
  # Train CNN (default paths)
  python train.py train --model cnn --epochs 30

  # Train TCN with inferred channel count
  python train.py train --model tcn --epochs 30

  # Explicit dataset roots/lists and output dir
  python train.py train --model cnn \
      --train-root data/das_data/train --train-list data/das_data/train/label.txt \
      --test-root  data/das_data/test  --test-list  data/das_data/test/label.txt \
      --out-dir models --batch-size 64 --lr 1e-3 --weight-decay 1e-5

  # Visualize a few raw samples before training (saved under outputs/raw_samples)
  python train.py train --model cnn --viz-samples 8
"""

from pathlib import Path
from typing import Optional, Tuple

import click
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from data_handler import (
    make_dataloaders, LoaderConfig, CLASS_NAMES, save_sample_images
)
from model_functions.cnn import (
    CNN, TrainConfig as CNNConfig, train_cnn, predict as predict_cnn
)
from model_functions.tcn import (
    TCN, TrainConfig as TCNConfig, train_tcn, predict as predict_tcn
)
# add near top
from model_functions.tft import TemporalFusionTransformer as TFT, TrainConfig as TFTConfig, train_tft, predict as predict_tft



# ------------------------------ Helpers ------------------------------ #

def _plot_cm(cm, out_path: Path, title: str) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    disp.plot(include_values=True, cmap="Blues", colorbar=False, ax=ax, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _infer_in_channels(dl) -> Optional[int]:
    """Peek a few samples to infer channel count C from tensors shaped (T, C)."""
    for i in range(len(dl.dataset)):
        sample = dl.dataset[i]
        if sample is not None:
            return int(sample["data"].shape[1])
    return None


@torch.no_grad()
def _quick_test_eval(
        model,
        predict_fn,
        test_loader,
        device: torch.device,
        cm_path: Path,
        title: str,
) -> None:
    model.eval().to(device)
    y_true, y_pred = [], []
    for batch in test_loader:
        if batch is None:
            continue
        x = batch["data"].unsqueeze(1).to(device, dtype=torch.float32)  # (B,1,T,C)
        y = batch["label"].to(device, dtype=torch.long)
        logits = predict_fn(model, x)
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())

    if not y_true:
        print("[WARN] No valid test samples after filtering.")
        return

    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    print(f"[{title}] Test Acc={acc:.3f}")
    _plot_cm(cm, cm_path, f"{title} – Confusion Matrix (test set)")


# -------------------------------- CLI -------------------------------- #

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """Phi-OTDR training CLI (CNN/TCN)."""
    pass


@cli.command("train")
@click.option("--model", type=click.Choice(["cnn", "tcn", "tft"]), required=True, help="Model to train.")
@click.option("--train-root", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "data" / "das_data" / "train",
              show_default=True, help="Root folder with training .mat files.")
@click.option("--test-root", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "data" / "das_data" / "test",
              show_default=True, help="Root folder with test .mat files.")
@click.option("--train-list", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "data" / "das_data" / "train" / "label.txt",
              show_default=True, help="Training label.txt (relative .mat path + class id).")
@click.option("--test-list", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "data" / "das_data" / "test" / "label.txt",
              show_default=True, help="Test label.txt (relative .mat path + class id).")
@click.option("--out-dir", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "models",
              show_default=True, help="Directory to save model weights.")
@click.option("--epochs", type=int, default=30, show_default=True)
@click.option("--batch-size", type=int, default=64, show_default=True)
@click.option("--lr", type=float, default=1e-3, show_default=True)
@click.option("--weight-decay", type=float, default=1e-5, show_default=True)
@click.option("--device", type=str, default=None, help="cuda|cpu (auto if omitted).")
@click.option("--viz-samples", type=int, default=6, show_default=True,
              help="Save N random raw samples before training (for sanity-check).")
@click.option("--in-channels", type=int, default=None,
              help="Override channel count for TCN (otherwise inferred from data).")
def train_cmd(
        model: str,
        train_root: Path,
        test_root: Path,
        train_list: Path,
        test_list: Path,
        out_dir: Path,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        device: str | None,
        viz_samples: int,
        in_channels: int | None,
):
    """Train a CNN or TCN on Φ-OTDR data and run a quick test-set evaluation."""
    here = Path(__file__).resolve().parent
    models_dir = Path(out_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    dev = torch.device("cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu"))
    print("[INFO] Using device:", dev)

    # Data + quick visualization
    train_loader, test_loader = make_dataloaders(
        train_root, train_list, test_root, test_list,
        LoaderConfig(batch_size=batch_size)
    )
    # Save a few raw samples to inspect the .mat structure visually
    save_sample_images(train_loader.dataset, here / "outputs" / "raw_samples", num=viz_samples)

    # ---------------------------- CNN path ---------------------------- #
    if model == "cnn":
        net = CNN(n_classes=len(CLASS_NAMES))
        cfg = CNNConfig(
            save_path=models_dir / "cnn.pt",
            device=dev,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
        )
        net = train_cnn(net, train_loader, test_loader, cfg)

        # Load best weights if saved (compat for torch versions)
        try:
            state = torch.load(cfg.save_path, map_location=dev, weights_only=True)
        except TypeError:
            state = torch.load(cfg.save_path, map_location=dev)
        net.load_state_dict(state)

        _quick_test_eval(
            model=net, predict_fn=predict_cnn, test_loader=test_loader,
            device=dev, cm_path=here / "outputs" / "confusion_matrix_train_cnn.png",
            title="CNN"
        )
    elif model == "tft":
        C = in_channels or _infer_in_channels(train_loader)
        if C is None:
            raise click.ClickException("Unable to infer in_channels; pass --in-channels.")
        net = TFT(in_channels=C, n_classes=len(CLASS_NAMES))
        cfg = TFTConfig(
            save_path=models_dir / "tft.pt",
            device=dev,
            epochs=epochs,
            lr=lr,
            weight_decay=weight_decay,
            in_channels=C,
        )
        train_tft(net, train_loader, test_loader, cfg)

        # Load best weights if saved (compat for torch versions)
        try:
            state = torch.load(cfg.save_path, map_location=dev, weights_only=True)
        except TypeError:
            state = torch.load(cfg.save_path, map_location=dev)
        net.load_state_dict(state)

        _quick_test_eval(
            model=net, predict_fn=predict_tft, test_loader=test_loader,
            device=dev, cm_path=here / "outputs" / "confusion_matrix_train_tft.png",
            title="TFT"
        )

    # ---------------------------- TCN path ---------------------------- #
    else:  # "tcn"
        # Determine input channels (C)
        C = in_channels or _infer_in_channels(train_loader)
        if C is None:
            raise click.ClickException(
                "Unable to infer in_channels from training data. "
                "Pass --in-channels explicitly."
            )

        net = TCN(in_channels=C, n_classes=len(CLASS_NAMES))
        cfg = TCNConfig(
            save_path=models_dir / "tcn.pt",
            device=dev,
            epochs=epochs, lr=lr, weight_decay=weight_decay,
            in_channels=C,
        )
        train_tcn(net, train_loader, test_loader, cfg)

        # Load best weights if saved (compat for torch versions)
        try:
            state = torch.load(cfg.save_path, map_location=dev, weights_only=True)
        except TypeError:
            state = torch.load(cfg.save_path, map_location=dev)
        net.load_state_dict(state)

        _quick_test_eval(
            model=net, predict_fn=predict_tcn, test_loader=test_loader,
            device=dev, cm_path=here / "outputs" / "confusion_matrix_train_tcn.png",
            title="TCN"
        )

    # ------------------------- Skipped report ------------------------- #
    tr = train_loader.dataset
    te = test_loader.dataset
    total_skipped = tr.skipped_missing + tr.skipped_broken + te.skipped_missing + te.skipped_broken
    print(
        f"Skipped files — "
        f"train: missing={tr.skipped_missing}, broken={tr.skipped_broken}; "
        f"test:  missing={te.skipped_missing}, broken={te.skipped_broken}; "
        f"total={total_skipped}"
    )


if __name__ == "__main__":
    cli()
