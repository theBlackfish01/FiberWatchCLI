from __future__ import annotations

import argparse
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data_handler import make_dataloaders, LoaderConfig, CLASS_NAMES, save_sample_images
from model_functions.cnn import CNN, TrainConfig as CNNConfig, train_cnn, predict as predict_cnn

def _plot_cm(cm, out_path: Path, title: str):
    fig = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        include_values=True, cmap="Blues", colorbar=False
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def main():
    here = Path(__file__).resolve().parent
    default_root = here / "data" / "das_data"

    ap = argparse.ArgumentParser(description="Train Phi-OTDR models")
    ap.add_argument("--mode", choices=["cnn"], required=True)
    ap.add_argument("--train-root", type=Path, default=default_root / "train")
    ap.add_argument("--test-root",  type=Path, default=default_root / "test")
    ap.add_argument("--train-list", type=Path, default=default_root / "train" / "label.txt")
    ap.add_argument("--test-list",  type=Path, default=default_root / "test"  / "label.txt")
    ap.add_argument("--out-dir", type=Path, default=here / "models")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--device", default=None)
    ap.add_argument("--viz-samples", type=int, default=6)
    args = ap.parse_args()

    models_dir = Path(args.out_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device is None and torch.cuda.is_available()) else (args.device or "cpu"))

    # Data + quick visualization
    train_loader, test_loader = make_dataloaders(
        args.train_root, args.train_list, args.test_root, args.test_list,
        LoaderConfig(batch_size=args.batch_size)  # num_workers=0, collate filters Nones
    )
    save_sample_images(train_loader.dataset, here / "outputs" / "raw_samples", num=args.viz_samples)

    # Train
    if args.mode == "cnn":
        model = CNN(n_classes=len(CLASS_NAMES))
        cfg = CNNConfig(
            save_path=models_dir / "cnn.pt",
            device=device,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
        )
        model = train_cnn(model, train_loader, test_loader, cfg)
        print("Using Device: ", device)
        # Evaluate on test set using best weights
        try:
            model.load_state_dict(torch.load(cfg.save_path, map_location=device, weights_only=True))
        except Exception:
            pass
        model.to(device).eval()
        y_true, y_pred = [], []
        with torch.no_grad():
            for batch in test_loader:
                if batch is None:
                    continue
                x = batch["data"].unsqueeze(1).to(device, dtype=torch.float32)
                y = batch["label"].to(device, dtype=torch.long)
                logits = predict_cnn(model, x)
                y_true.extend(y.cpu().tolist())
                y_pred.extend(logits.argmax(1).cpu().tolist())

        if len(y_true) > 0:
            acc = accuracy_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
            print(f"[CNN] Test Acc={acc:.3f}")
            _plot_cm(cm, here / "outputs" / "confusion_matrix_train.png", "CNN – Confusion Matrix (test set)")
        else:
            print("[WARN] No valid test samples after filtering.")

    # Report skipped counts
    tr = train_loader.dataset
    te = test_loader.dataset
    total_skipped = tr.skipped_missing + tr.skipped_broken + te.skipped_missing + te.skipped_broken
    print(
        f"Skipped files — "
        f"train: missing={tr.skipped_missing}, broken={tr.skipped_broken}; "
        f"test: missing={te.skipped_missing}, broken={te.skipped_broken}; "
        f"total={total_skipped}"
    )

if __name__ == "__main__":
    main()
