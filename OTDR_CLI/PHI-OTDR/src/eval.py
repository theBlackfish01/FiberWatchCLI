from __future__ import annotations

import argparse
import base64
import os
from pathlib import Path
from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from data_handler import make_dataloaders, LoaderConfig, CLASS_NAMES, save_sample_images
from model_functions.cnn import CNN, predict as predict_cnn

# ---- Optional config import for API key (falls back to env var) ----
try:
    import config.config as cfg  # project-level config
    _OPENAI_KEY = getattr(cfg, "OPENAI_API_KEY", None)
except Exception:
    _OPENAI_KEY = None

# ---- LLM (OpenAI) ---------------------------------------------------
def _b64(path: Path) -> str:
    """Return a data-URL for PNG/JPEG at `path`."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    enc = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{enc}"

def _get_openai_client():
    key = _OPENAI_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        print("OPENAI_API_KEY not set – skipping LLM explanation")
        return None
    from openai import OpenAI
    return OpenAI(api_key=key)

def _llm_explain_phi(img_paths: List[Path], model_name: str = "gpt-4o-mini") -> str | None:
    """
    Ask a vision-capable LLM to describe common patterns and misclassifications
    across provided Phi-OTDR heatmaps. Returns text or None if unavailable.
    """
    client = _get_openai_client()
    if client is None:
        return None

    system_prompt = (
        "You are an expert in distributed acoustic sensing (Φ-OTDR). "
        "Each figure is a time–channel heatmap (x=time index, y=channel) with a title "
        "that includes ground truth and predicted class. "
        "Briefly explain common patterns you see across the images, typical signatures "
        "for each class (background, digging, knocking, shaking, watering, walking), "
        "and any likely reasons for misclassifications. Keep it concise and practical."
    )

    user_parts = [{"type": "text", "text": "Here are the selected samples:"}]
    user_parts += [{"type": "image_url", "image_url": {"url": _b64(p)}} for p in img_paths]

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_parts},
        ],
        max_tokens=500,
    )
    return resp.choices[0].message.content.strip()

# ---- Viz helpers -----------------------------------------------------
def _plot_cm(cm, out_path: Path, title: str):
    fig = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES).plot(
        include_values=True, cmap="Blues", colorbar=False
    )
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def _visualise_heatmap(arr: np.ndarray, true_idx: int, pred_idx: int, idx: int, out_dir: Path) -> Path:
    """
    Save a single heatmap figure for one sample (T x C), with title
    showing true/predicted classes.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(arr.T, aspect="auto", origin="lower")
    ax.set_title(f"Sample #{idx} | True={CLASS_NAMES[true_idx]}  Pred={CLASS_NAMES[pred_idx]}")
    ax.set_ylabel("Channel")
    ax.set_xlabel("Time index")
    fig.colorbar(im, ax=ax, shrink=0.8, label="normalized amplitude")
    fig.tight_layout()
    path = out_dir / f"sample_{idx}_t{true_idx}_p{pred_idx}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

# ---- Main ------------------------------------------------------------
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
    ap.add_argument("--num-samples", type=int, default=6, help="random samples to visualise")
    ap.add_argument("--llm-model", type=str, default="gpt-4o-mini", help="vision model for explanations")
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
    # NOTE: if your torch version doesn't support weights_only, remove that kwarg.
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

    # ---------------- LLM explanation (no RAG) ----------------
    # Pick random samples, render labeled heatmaps, and ask the LLM.
    try:
        ds = test_loader.dataset
        rng = np.random.default_rng(42)
        indices = rng.choice(len(ds), size=min(args.num_samples, len(ds)), replace=False)
        llm_imgs_dir = out_dir / "samples_llm"
        img_paths: List[Path] = []

        for i in indices:
            sample = ds[i]
            arr = sample["data"].numpy()
            t_lab = int(sample["label"].item())
            with torch.no_grad():
                x1 = sample["data"].unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
                pred = predict_cnn(model, x1).argmax(1).item()
            img_paths.append(_visualise_heatmap(arr, t_lab, int(pred), int(i), llm_imgs_dir))

        explanation = _llm_explain_phi(img_paths, model_name=args.llm_model)
        if explanation:
            llm_dir = here / "outputs" / "llm_output"
            llm_dir.mkdir(parents=True, exist_ok=True)
            out_file = llm_dir / "phi_otdr_llm_explanation.txt"
            k = 1
            while out_file.exists():
                out_file = llm_dir / f"phi_otdr_llm_explanation_{k}.txt"
                k += 1
            header = "LLM explanation for Φ-OTDR eval subset (no RAG):\n\n"
            out_file.write_text(header + explanation, encoding="utf-8")
            print(f"LLM explanation saved to {out_file.name}")
    except Exception as e:
        print(f"[WARN] LLM explanation step skipped due to error: {e}")

    # Report skipped counts (from data_handler)
    te = test_loader.dataset
    total_skipped = getattr(te, "skipped_missing", 0) + getattr(te, "skipped_broken", 0)
    print(f"Skipped files — test: missing={getattr(te, 'skipped_missing', 0)}, "
          f"broken={getattr(te, 'skipped_broken', 0)}; total={total_skipped}")

if __name__ == "__main__":
    main()
