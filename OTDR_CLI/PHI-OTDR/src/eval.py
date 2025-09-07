from __future__ import annotations

"""
Phi-OTDR evaluation & visualization (CNN/TCN) with LLM explanation + FAISS RAG (auto).

Usage examples:
  python eval.py eval --model cnn --weights models/cnn.pt
  python eval.py eval --model tcn --weights models/tcn.pt --num-samples 8 --skip-llm
  python eval.py eval --model cnn --llm-model gpt-4o-mini

Notes:
- RAG is automatically enabled if a FAISS index exists at:
    PHI-OTDR/src/corpus/index.faiss          (default)
    PHI-OTDR/src/corpus/chunks.jsonl         (sidecar store)
    PHI-OTDR/src/corpus/chunks.meta.json     (auto-detected embed model)
  Build these once via:
    python PHI-OTDR/src/rag.py build --corpus PHI-OTDR/src/corpus
"""

import base64
import json
import os
from pathlib import Path
from typing import List, Tuple, Optional

import click
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

from data_handler import make_dataloaders, LoaderConfig, CLASS_NAMES, save_sample_images
from model_functions.cnn import CNN, predict as predict_cnn
from model_functions.tcn import TCN, predict as predict_tcn
from rag import retrieve  # FAISS-backed retriever

# ---------- Optional API key via project config; falls back to env ----------
try:
    import config.config as cfg  # project-level config (OTDR_CLI/src/config/config.py)
    _OPENAI_KEY = getattr(cfg, "OPENAI_API_KEY", None)
except Exception:
    _OPENAI_KEY = None


# ============================== RAG defaults ============================== #

def _default_rag_paths() -> tuple[Path, Path, Path]:
    """Return (index.faiss, chunks.jsonl, chunks.meta.json) under src/corpus/."""
    here = Path(__file__).resolve().parent
    corpus = here / "corpus"
    index_path = corpus / "index.faiss"
    store_path = corpus / "chunks.jsonl"
    meta_path = corpus / "chunks.meta.json"  # written by rag.build()
    return index_path, store_path, meta_path


def _load_embed_model_from_meta(meta_path: Path) -> Optional[str]:
    """Read the embedding model name used at index build time."""
    try:
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        return meta.get("embed_model", None)
    except Exception:
        return None


def _auto_retrieve_snippets(k: int = 5) -> tuple[List[str], bool]:
    """
    If the FAISS index+store exist, run a simple task-specific query and
    return top-k snippet texts. Returns (snippets, used_rag_flag).
    """
    index_path, store_path, meta_path = _default_rag_paths()
    if not (index_path.is_file() and store_path.is_file()):
        return [], False

    embed_model = _load_embed_model_from_meta(meta_path)
    if embed_model is None:
        # Fall back to default used in rag.py; may still work if matching build-time model
        embed_model = "text-embedding-3-large"

    class_list = ", ".join(CLASS_NAMES)
    query = (
        "Phi-OTDR / Distributed Acoustic Sensing (DAS) event signatures, "
        "typical time–channel heatmap patterns, confusion cases and diagnostics for classes: "
        f"{class_list}. Include tips on distinguishing similar temporal envelopes and channel energy distributions."
    )
    try:
        hits = retrieve(
            query=query,
            k=k,
            index_path=index_path,
            store_path=store_path,
            embed_model=embed_model,
        )
        # Limit each chunk length to keep prompt size reasonable
        snippets = [h["text"][:900] for h in hits if "text" in h and h["text"].strip()]
        if snippets:
            print(f"[RAG] Retrieved {len(snippets)} snippets from {index_path.name}")
            return snippets, True
    except Exception as e:
        print(f"[RAG] Retrieval failed: {e}")

    return [], False


# ============================== LLM helpers =============================== #

def _b64(path: Path) -> str:
    """Return a data-URL (PNG/JPEG) ready for image_url."""
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


def _llm_explain_phi(
    img_paths: List[Path],
    model_name: str = "gpt-4o-mini",
    rag_snippets: List[str] | None = None,
) -> str | None:
    """
    Ask a vision-capable LLM to describe patterns and misclassifications.
    If rag_snippets are provided, prepend a numbered reference block and ask the model to cite [1], [2], …
    """
    client = _get_openai_client()
    if client is None:
        return None

    system_prompt = (
        "You are an expert in distributed acoustic sensing (Φ-OTDR). "
        "Each image shows a time–channel heatmap with overlaid text and probability bars. "
        "Explain typical signatures for the classes "
        "(background, digging, knocking, watering, shaking, walking). "
        "Note any misclassifications and plausible causes (e.g., similar temporal envelopes "
        "or channel energy distributions), and provide practical diagnostic tips. "
        "When helpful, cite the provided reference snippets like [1], [2]…"
    )

    ref_block = ""
    if rag_snippets:
        joined = "\n\n".join(f"[{i + 1}] {s}" for i, s in enumerate(rag_snippets))
        ref_block = "Reference snippets:\n" + joined

    user_parts = []
    if ref_block:
        user_parts.append({"type": "text", "text": ref_block})
    user_parts.append({"type": "text", "text": "Here are the selected samples:"})
    user_parts += [{"type": "image_url", "image_url": {"url": _b64(p)}} for p in img_paths]

    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_parts},
        ],
        max_tokens=700,
    )
    return resp.choices[0].message.content.strip()


# =============================== Viz utils =============================== #

def _plot_cm(cm: np.ndarray, out_path: Path, title: str) -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    disp.plot(include_values=True, cmap="Blues", colorbar=False, ax=ax, xticks_rotation=45)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _heatmap_basic(arr: np.ndarray, true_idx: int, pred_idx: int, idx: int, out_dir: Path) -> Path:
    """Standard heatmap with True/Pred in title."""
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


def _heatmap_llm_sheet(
    arr: np.ndarray,
    logits: np.ndarray,
    true_idx: int,
    idx: int,
    out_dir: Path,
) -> Path:
    """
    LLM-friendly sheet:
      - left: heatmap (downsampled if long)
      - right-top: per-class probability bars with labels
      - right-bottom: text box with concise stats: T,C, mean/max, top-energy channels, pred/prob
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    T, C = arr.shape
    # Downsample in time for readability if very long
    max_T = 1500
    arr_ds = arr[:: int(np.ceil(T / max_T))] if T > max_T else arr

    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    pred_idx = int(np.argmax(probs))
    pred_prob = float(probs[pred_idx])

    # Channel energy (RMS) — helps explain which channels are active
    rms = np.sqrt(np.mean(arr * arr, axis=0))
    top_k_idx = np.argsort(rms)[-3:][::-1]  # top 3
    top_k_desc = ", ".join([f"ch{int(i)}={rms[i]:.2f}" for i in top_k_idx])

    fig = plt.figure(figsize=(11, 4.5))
    # Grid spec: left (heatmap) wide; right split into bars + textbox
    gs = fig.add_gridspec(2, 2, width_ratios=[3.0, 2.2], height_ratios=[1, 1])

    # Left: heatmap
    ax0 = fig.add_subplot(gs[:, 0])
    im = ax0.imshow(arr_ds.T, aspect="auto", origin="lower")
    ax0.set_title(f"Sample #{idx} | True: {CLASS_NAMES[true_idx]} | Pred: {CLASS_NAMES[pred_idx]} ({pred_prob:.2f})")
    ax0.set_ylabel("Channel")
    ax0.set_xlabel("Time index")
    cb = fig.colorbar(im, ax=ax0, shrink=0.8)
    cb.set_label("norm. amplitude")

    # Right top: per-class probabilities (horizontal bars)
    ax1 = fig.add_subplot(gs[0, 1])
    y = np.arange(len(CLASS_NAMES))
    ax1.barh(y, probs, align="center")
    ax1.set_yticks(y)
    ax1.set_yticklabels(CLASS_NAMES)
    ax1.invert_yaxis()
    ax1.set_xlim(0, 1)
    ax1.set_xlabel("probability")
    ax1.set_title("Model probabilities")

    # Right bottom: text box with concise stats
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.axis("off")
    txt = (
        f"Stats:\n"
        f"- Shape: T={T}, C={C}\n"
        f"- Mean={arr.mean():.3f}, Std={arr.std():.3f}, Max={arr.max():.3f}\n"
        f"- Top channels (RMS): {top_k_desc}\n"
        f"- Prediction: {CLASS_NAMES[pred_idx]}  (p={pred_prob:.2f})\n"
        f"- Ground truth: {CLASS_NAMES[true_idx]}"
    )
    ax2.text(0.02, 0.95, txt, va="top", ha="left", fontsize=10, family="monospace")

    fig.tight_layout()
    path = out_dir / f"llm_sheet_{idx}_t{true_idx}_p{pred_idx}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ================================ Core ================================= #

def _infer_in_channels(dl) -> int | None:
    """Peek a few samples to infer channel count C from (T, C)."""
    for i in range(len(dl.dataset)):
        sample = dl.dataset[i]
        if sample is not None:
            return int(sample["data"].shape[1])
    return None


def _load_model(model_name: str, in_channels: int, n_classes: int, weights: Path, device: torch.device):
    """Construct model by name and load weights (with safe fallback)."""
    if model_name == "cnn":
        model = CNN(n_classes=n_classes)
        predict_fn = predict_cnn
    else:  # "tcn"
        model = TCN(in_channels=in_channels, n_classes=n_classes)
        predict_fn = predict_tcn

    # Older torch may not support weights_only=True
    try:
        state = torch.load(weights, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weights, map_location=device)
    model.load_state_dict(state)
    model.eval().to(device)
    return model, predict_fn


def _evaluate_and_visualize(
    model_name: str,
    weights: Path,
    test_root: Path,
    test_list: Path,
    batch_size: int,
    device: torch.device,
    out_dir: Path,
    num_samples: int,
    llm_model: str,
    skip_llm: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Loader
    _, test_loader = make_dataloaders(
        train_root=test_root, train_list=test_list,
        test_root=test_root, test_list=test_list,
        cfg=LoaderConfig(batch_size=batch_size)
    )

    in_channels = _infer_in_channels(test_loader)
    if in_channels is None:
        print("[WARN] No valid test samples after filtering.")
        te = test_loader.dataset
        total_skipped = te.skipped_missing + te.skipped_broken
        print(f"Skipped files — test: missing={te.skipped_missing}, broken={te.skipped_broken}; total={total_skipped}")
        return

    # Model
    n_classes = len(CLASS_NAMES)
    model, predict_fn = _load_model(model_name, in_channels, n_classes, weights, device)

    # Inference
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            x = batch["data"].unsqueeze(1).to(device, dtype=torch.float32)  # (B,1,T,C) for both models
            y = batch["label"].to(device, dtype=torch.long)
            logits = predict_fn(model, x)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(logits.argmax(1).cpu().tolist())

    # Metrics & confusion matrix
    if y_true:
        acc = accuracy_score(y_true, y_pred)
        cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
        print(f"Eval size = {len(y_true)} | Acc = {acc:.3f}")
        _plot_cm(cm, out_dir / "confusion_matrix.png",
                 f"{model_name.upper()} – Confusion Matrix (test set)")
        # Save raw sample heatmaps for quick inspection
        save_sample_images(test_loader.dataset, out_dir / "samples_raw", num=num_samples)
    else:
        print("[WARN] No valid test samples after filtering.")

    # LLM-friendly sheets + basic pred heatmaps
    rng = np.random.default_rng(42)
    chosen = rng.choice(len(test_loader.dataset), size=min(num_samples, len(test_loader.dataset)), replace=False)
    basic_img_dir = out_dir / "samples_pred"
    llm_img_dir = out_dir / "samples_llm"
    basic_img_dir.mkdir(parents=True, exist_ok=True)
    llm_img_dir.mkdir(parents=True, exist_ok=True)

    all_llm_imgs: List[Path] = []
    for idx in chosen:
        sample = test_loader.dataset[idx]
        if sample is None:
            continue
        arr = sample["data"].numpy()
        t_lab = int(sample["label"].item())
        with torch.no_grad():
            x1 = sample["data"].unsqueeze(0).unsqueeze(0).to(device, dtype=torch.float32)
            logits1 = predict_fn(model, x1).squeeze(0).cpu().numpy()
            p_idx = int(np.argmax(logits1))
        _heatmap_basic(arr, t_lab, p_idx, int(idx), basic_img_dir)
        llm_path = _heatmap_llm_sheet(arr, logits1, t_lab, int(idx), llm_img_dir)
        all_llm_imgs.append(llm_path)

    # ---- Auto RAG (if index present) ----
    rag_snippets: List[str] = []
    used_rag = False
    if not skip_llm and all_llm_imgs:
        rag_snippets, used_rag = _auto_retrieve_snippets(k=5)

    # LLM explanation (optional)
    if not skip_llm and all_llm_imgs:
        try:
            explanation = _llm_explain_phi(all_llm_imgs, model_name=llm_model, rag_snippets=rag_snippets or None)
            if explanation:
                llm_dir = out_dir.parent / "llm_output"
                llm_dir.mkdir(parents=True, exist_ok=True)
                out_file = llm_dir / f"phi_otdr_{model_name}_llm_explanation.txt"
                k = 1
                while out_file.exists():
                    out_file = llm_dir / f"phi_otdr_{model_name}_llm_explanation_{k}.txt"
                    k += 1
                header = (
                    f"LLM explanation for Φ-OTDR eval subset "
                    f"({model_name.upper()}, {'with RAG' if used_rag else 'no RAG'}):\n\n"
                )
                out_file.write_text(header + explanation, encoding="utf-8")
                print(f"LLM explanation saved to {out_file.name}")
        except Exception as e:
            print(f"[WARN] LLM explanation step skipped due to error: {e}")

    # Skipped report
    te = test_loader.dataset
    total_skipped = te.skipped_missing + te.skipped_broken
    print(f"Skipped files — test: missing={te.skipped_missing}, broken={te.skipped_broken}; total={total_skipped}")


# ================================ CLI ================================= #

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """Phi-OTDR evaluation CLI (CNN/TCN) with visualization and LLM explanations (+ auto RAG)."""
    pass


@cli.command("eval")
@click.option("--model", type=click.Choice(["cnn", "tcn"]), required=True, help="Model type to evaluate.")
@click.option("--weights", type=click.Path(path_type=Path), default=None,
              help="Path to model weights. Defaults to models/<model>.pt.")
@click.option("--test-root", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "data" / "das_data" / "test",
              show_default=True, help="Root folder containing test .mat files.")
@click.option("--test-list", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "data" / "das_data" / "test" / "label.txt",
              show_default=True, help="label.txt mapping relative paths to class IDs.")
@click.option("--batch-size", type=int, default=64, show_default=True, help="Batch size.")
@click.option("--device", type=str, default=None, help="cuda|cpu (auto if omitted).")
@click.option("--out-dir", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "outputs" / "eval_outputs",
              show_default=True, help="Directory to save metrics/plots.")
@click.option("--num-samples", type=int, default=6, show_default=True,
              help="Number of random samples to visualize.")
@click.option("--llm-model", type=str, default="gpt-4o-mini", show_default=True,
              help="OpenAI vision model for explanations.")
@click.option("--skip-llm", is_flag=True, help="Skip the LLM explanation step.")
def eval_cmd(
    model: str,
    weights: Path | None,
    test_root: Path,
    test_list: Path,
    batch_size: int,
    device: str | None,
    out_dir: Path,
    num_samples: int,
    llm_model: str,
    skip_llm: bool,
):
    """Run evaluation, render plots, and (optionally) generate an LLM explanation (auto RAG if available)."""
    here = Path(__file__).resolve().parent
    # Default weights by model if not provided
    if weights is None:
        weights = here / "models" / f"{model}.pt"

    # Device selection
    dev = torch.device("cuda" if (device is None and torch.cuda.is_available()) else (device or "cpu"))

    _evaluate_and_visualize(
        model_name=model,
        weights=weights,
        test_root=test_root,
        test_list=test_list,
        batch_size=batch_size,
        device=dev,
        out_dir=out_dir,
        num_samples=num_samples,
        llm_model=llm_model,
        skip_llm=skip_llm,
    )


if __name__ == "__main__":
    cli()
