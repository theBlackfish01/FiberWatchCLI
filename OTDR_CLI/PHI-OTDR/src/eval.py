from __future__ import annotations

import argparse
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data_handler import make_dataloaders, LoaderConfig, CLASS_NAMES, save_sample_images
from model_functions.cnn import CNN, predict as predict_cnn

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

    ap = argparse.ArgumentParser(description="Evaluate Phi-OTDR models & visualise outputs")
    ap.add_argument("--model", choices=["cnn"], required=True)
    ap.add_argument("--weights", type=Path, default=here / "models" / "cnn.pt")
    ap.add_argument("--test-root", type=Path, default=default_root / "test")
    ap.add_argument("--test-list", type=Path, default=default_root / "test" / "label.txt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", type=Path, default=here / "outputs" / "eval_outputs")
    ap.add_argument("--num-samples", type=int, default=6)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if (args.device is None and torch.cuda.is_available()) else (args.device or "cpu"))

    # Only need the test loader; train loader will share the same filtering logic
    _, test_loader = make_dataloaders(
        train_root=args.test_root, train_list=args.test_list,
        test_root=args.test_root,  test_list=args.test_list,
        cfg=LoaderConfig(batch_size=args.batch_size)
    )

    # Load model
    model = CNN(n_classes=len(CLASS_NAMES))
    model.load_state_dict(torch.load(args.weights, map_location=device, weights_only=True))
    model.eval().to(device)

    # Inference
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
        print(f"Eval size = {len(y_true)} | Acc = {acc:.3f}")
        _plot_cm(cm, out_dir / "confusion_matrix.png", "CNN – Confusion Matrix (test set)")
        # also dump a few visual samples
        save_sample_images(test_loader.dataset, out_dir / "samples", num=args.num_samples)
    else:
        print("[WARN] No valid test samples after filtering.")

    # Report skipped counts
    te = test_loader.dataset
    total_skipped = te.skipped_missing + te.skipped_broken
    print(f"Skipped files — test: missing={te.skipped_missing}, broken={te.skipped_broken}; total={total_skipped}")

if __name__ == "__main__":
    main()
