from __future__ import annotations

import base64

"""Evaluation + visual explanation script for OTDR pipeline.

Two modes:

1. **pipeline** – GRU‑AE anomaly detection ➜ selected samples → classifier (TCN/TST)
2. **direct**  – classifier directly on the full test set

For a handful of random traces this script draws amplitude + prediction
overlays and asks an OpenAI LLM to provide a natural‑language explanation.
"""

from pathlib import Path
import argparse
import json
import re
from typing import List, Tuple
from contextlib import ExitStack

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, ConfusionMatrixDisplay

# Local imports
from data_helper import load_raw_dataframe, make_splits, fit_scaler, tensorise_splits
from model_functions.gruae import VectorGRUAE, reconstruction_error
from model_functions.tcn import OTDR_TCN, predict as predict_tcn
from model_functions.tst import TimeSeriesTransformer, predict as predict_tst
import config.config as cfg
from pathlib import Path
from typing import List, Optional
from rag import retrieve
from openai import OpenAI
client = OpenAI(api_key=cfg.OPENAI_API_KEY)


import warnings
warnings.filterwarnings("ignore", category=FutureWarning)  # noqa: T201

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
    scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(meta["scaler_scale"], dtype=np.float32)
    threshold = float(meta["threshold"])

    # feature dim = scaler_mean length
    feat_dim = scaler_mean.size
    ae = VectorGRUAE(feat_dim=feat_dim)
    ae.load_state_dict(torch.load(det_path, map_location=device))
    ae.eval().to(device)
    return ae, threshold, scaler_mean, scaler_scale


def _load_classifier(kind: str, cls_path: Path, seq_len: int, n_classes: int, device: torch.device):
    if kind == "tcn":
        model = OTDR_TCN(n_classes=n_classes)
        model.load_state_dict(torch.load(cls_path, map_location=device))
    elif kind == "tst":
        model = TimeSeriesTransformer(seq_len=seq_len, n_classes=n_classes)
        model.load_state_dict(torch.load(cls_path, map_location=device))
    else:
        raise ValueError("classifier kind must be 'tcn' or 'tst'")
    return model.eval().to(device)


def _visualise_sample(
    amps: np.ndarray,
    snr: float,
    true_cls: int,
    pred_cls: int,
    true_pos: float,
    pred_pos: float,
    idx: int,
    out_dir: Path,
):
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(amps.size), amps, label="Amplitude")
    plt.title(
        f"Sample #{idx} | TrueC={true_cls} PredC={pred_cls} | "
        f"TruePos={true_pos:.3f}m  PredPos={pred_pos:.3f}m | SNR={snr:.2f}"
    )
    plt.xlabel("P-index")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    fname = out_dir / f"sample_{idx}.png"
    plt.savefig(fname, dpi=150)
    plt.close()
    return fname


def _b64(path: Path) -> str:
    """Return a data‑URL (PNG/JPEG) ready for image_url."""
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    with path.open("rb") as f:
        enc = base64.b64encode(f.read()).decode("ascii")
    return f"data:{mime};base64,{enc}"


def _llm_explain(
    img_paths: List[Path],
    openai_model: str = "gpt-4o-mini",
) -> tuple[str, bool] | None:
    """
    Ask a vision‑capable chat model for a concise explanation of common
    patterns in the supplied OTDR trace images, **augmented with RAG‑retrieved
    reference snippets**.

    Returns the explanation text, or None if OPENAI_API_KEY isn’t set.
    """
    api_key = cfg.OPENAI_API_KEY
    if not api_key:
        print("OPENAI_API_KEY not set – skipping LLM explanation")
        return None

    client = OpenAI(api_key=api_key)

    # ---------- RAG: retrieve reference snippets -------------------------
    query = "OTDR fault plots – " + ", ".join(p.stem for p in img_paths)
    try:
        retrieved = retrieve(query, k=5)          # ← may raise / be empty
    except Exception as exc:
        print(f"RAG retrieval failed: {exc}")
        retrieved = []

    ref_block = "\n\n".join(f"[{i+1}] {r['text']}" for i, r in enumerate(retrieved))

    # ---------- build messages ------------------------------------------
    system_prompt = (
        "You are an optical‑fibre fault‑analysis expert. "
        "Given the following figures (each shows amplitude over P‑points with "
        "predictions vs ground truth in the title), write a concise explanation "
        "of common patterns you observe, including typical failure modes and "
        "any misclassifications. Explain the type of fault, position, possible causes "
        "and possible solutions. Provide brief answers.\n\n"
        "Use the reference snippets and image when required, "
        "citing snippets like [1], [2] where appropriate.\n\n"
        "Fault Classes are labelled as follows:\n"
        "id\tfault type \ttypical signs\n"
        "0\tnormal / no fault\tloss = 0, Position = 0\n"
        "1\tfibre cut / strong attenuation\thigh loss, reflectance 0\n"
        "2\tdirty connector\tsmall reflectance spike, low loss\n"
        "3\toptical tap (eavesdrop)\treflectance ≈ 0, moderate loss\n"
        "4\tmacrobend / sharp bend\tmedium loss, no reflectance\n"
        "5\tbad splice\treflectance < 0 dB, moderate loss\n"
        "6\thigh‑reflective break\treflectance −0.3 … −0.5 dB\n"
        "7\tunknown / mixed\tthe catch‑all class"
    )

    # first part: reference snippets (if any) + lead‑in text
    user_parts = [
        {"type": "text",
         "text": "Reference snippets:\n" + (ref_block or "*<no snippets retrieved>*")},
        {"type": "text",
         "text": "Here are the selected samples for inspection:"},
    ] + [
        {   # the images themselves
            "type": "image_url",
            "image_url": {"url": _b64(p)},
        }
        for p in img_paths
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_parts},
    ]

    # ---------- chat completion -----------------------------------------
    resp = client.chat.completions.create(
        model=openai_model,
        messages=messages,
        max_tokens=600,   # limit the response length
    )

    rag_flag = False
    if retrieved:
        rag_flag = True
        print("RAG retrieval successful, using retrieved snippets in LLM prompt.")

    return resp.choices[0].message.content.strip(), rag_flag




# --------------------------------------------------
# Main eval flow
# --------------------------------------------------

def main() -> None:  # noqa: C901
    ap = argparse.ArgumentParser(description="Evaluate OTDR models & visualise outputs")
    ap.add_argument("--mode", choices=["pipeline", "direct"], required=True)
    ap.add_argument("--classifier", choices=["tcn", "tst"], default="tcn")
    ap.add_argument("--data", default="data/OTDR_data.csv")
    ap.add_argument("--detector", default="models/gru_ae.pt")
    ap.add_argument("--cls-path", default=None, help="Classifier weights path; defaults based on --classifier")
    ap.add_argument("--num-samples", type=int, default=2, help="Random samples to visualise & explain")
    ap.add_argument("--out-dir", default="eval_outputs")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ---------- #
    df = load_raw_dataframe(args.data)
    _, _, test_df = make_splits(df)

    meas_cols = [c for c in test_df.columns if re.fullmatch(r"P\d+", c)] + ["SNR"]
    scaler = fit_scaler(test_df[meas_cols].values.astype(np.float32))  # fit on test for standardisation only here
    splits = tensorise_splits(test_df, test_df, test_df, scaler)  # only need "test" key
    X_test = splits["test"].X
    y_cls_test = splits["test"].y_class
    y_pos_test = splits["test"].y_pos

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device is None else torch.device(args.device)
    )

    # ---------- load models ---------- #
    cls_path = Path(args.cls_path) if args.cls_path else Path("models") / ("tcn.pt" if args.classifier == "tcn" else "tst.pt")

    n_classes = int(df["Class"].max() + 1)
    classifier = _load_classifier(args.classifier, cls_path, seq_len=X_test.shape[1], n_classes=n_classes, device=device)

    if args.mode == "pipeline":
        ae, threshold, _, _ = _load_gru_ae(Path(args.detector), device)
        errs = reconstruction_error(ae, X_test, device=device)
        is_fault = errs > threshold
        # if all healthy, fallback to sample of test
        idx_to_eval = torch.nonzero(is_fault).squeeze(-1)
        if idx_to_eval.numel() == 0:
            idx_to_eval = torch.arange(0, min(1000, X_test.size(0)))
    else:  # direct
        idx_to_eval = torch.arange(X_test.size(0))

    # ------------- inference ------------- #
    logits, pos_hat = (
        predict_tcn(classifier, X_test[idx_to_eval])
        if args.classifier == "tcn"
        else predict_tst(classifier, X_test[idx_to_eval])
    )
    preds_cls = logits.argmax(1)

    # metrics
    acc = accuracy_score(y_cls_test[idx_to_eval].numpy(), preds_cls.numpy())
    rmse = mean_squared_error(y_pos_test[idx_to_eval].numpy(), pos_hat.numpy())
    print(f"Eval subset size = {idx_to_eval.size(0)} | Acc = {acc:.3f} | RMSE = {rmse:.3f}")  # noqa: T201

    # Confusion matrix plot
    cm = confusion_matrix(y_cls_test[idx_to_eval].numpy(), preds_cls.numpy())
    fig_cm = ConfusionMatrixDisplay(cm).plot(include_values=True, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix – Eval subset")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ------------- random visualisations ------------- #
    rng = np.random.default_rng(42)
    chosen = rng.choice(idx_to_eval.numpy(), size=min(args.num_samples, idx_to_eval.size(0)), replace=False)
    img_paths = []
    num_points = X_test.shape[1] - 1  # number of P-points in the traces
    for idx in chosen:
        amp = X_test[idx][:num_points].numpy() * scaler.scale_[:num_points] + scaler.mean_[:num_points]
        snr = float(X_test[idx][num_points].item() * scaler.scale_[num_points] + scaler.mean_[num_points])
        t_cls = int(y_cls_test[idx].item())
        p_cls = int(preds_cls[idx_to_eval == idx][0].item())
        t_pos = float(y_pos_test[idx].item())
        p_pos = float(pos_hat[idx_to_eval == idx][0].item())
        img_paths.append(_visualise_sample(amp, snr, t_cls, p_cls, t_pos, p_pos, int(idx), out_dir))

    # ------------- LLM explanation ------------- #
    explanation, rag_flag = _llm_explain(img_paths)
    llm_dir = Path("llm_output")
    llm_dir.mkdir(parents=True, exist_ok=True)
    if explanation:
        explanation_file = llm_dir / "llm_explanation.txt"
        i = 1
        while explanation_file.exists():
            explanation_file = llm_dir / f"llm_explanation_{i}.txt"
            i += 1
        if rag_flag:
            explanation = f"LLM explanation for eval subset with RAG:\n\n{explanation}"
        else:
            explanation = f"LLM explanation for eval subset without RAG:\n\n{explanation}"
        explanation_file.write_text(explanation)
        print(f"LLM explanation saved to {explanation_file.name}")  # noqa: T201

if __name__ == "__main__":
    main()

# eval input arguments

# if pipeline, use pipeline

# if model, then pass to model

# grab output

# results

# LLM explanation