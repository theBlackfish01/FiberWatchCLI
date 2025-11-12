# OTDR/src/eval.py

"""
Evaluation script

Two modes:

1. Pipeline – GRU‑AE anomaly detection ➜ selected samples → classifier (TCN/TST)
2. Direct – classifier directly on the full test set

LLM explanation of random samples using vision‑capable GPT‑4o‑mini with RAG
"""

from __future__ import annotations
import base64
import click
import json
import re
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from sklearn.metrics import accuracy_score, root_mean_squared_error, confusion_matrix, ConfusionMatrixDisplay, classification_report
from data_helper import load_raw_dataframe, make_splits, tensorise_splits
from model_functions.gruae import VectorGRUAE, reconstruction_error
from model_functions.tcn import OTDR_TCN, predict as predict_tcn
from model_functions.tst import TimeSeriesTransformer, predict as predict_tst
from model_functions.tabnet import OTDR_TabNet, predict as predict_tabnet
import config.config as cfg
from pathlib import Path
from rag import retrieve
from openai import OpenAI
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)  # noqa: T201
client = OpenAI(api_key=cfg.OPENAI_API_KEY)

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
    if "scaler_mean" in meta and "scaler_scale" in meta:
        scaler_mean = np.array(meta["scaler_mean"], dtype=np.float32)
        scaler_scale = np.array(meta["scaler_scale"], dtype=np.float32)
    elif "mean" in meta and "scale" in meta:
        # backwards compatibility with early metadata files
        scaler_mean = np.array(meta["mean"], dtype=np.float32)
        scaler_scale = np.array(meta["scale"], dtype=np.float32)
    else:
        raise KeyError("GRU-AE metadata must contain scaler_mean/scale entries")
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
        model = TimeSeriesTransformer(seq_len=seq_len)
        model.load_state_dict(torch.load(cls_path, map_location=device))
    elif kind == "tab":
        model = OTDR_TabNet(n_classes=n_classes)
        model.load_state_dict(torch.load(cls_path, map_location=device))
    else:
        raise ValueError("classifier kind must be 'tcn', 'tab' or 'tst'")
    return model.eval().to(device)


def _visualise_sample(
        amps: np.ndarray,
        snr: float,
        true_cls: int,
        pred_cls: int | None,
        true_pos: float,
        pred_pos: float,
        idx: int,
        out_dir: Path,
):
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(amps.size), amps, label="Amplitude")
    pred_label = "N/A" if pred_cls is None else str(pred_cls)
    plt.title(
        f"Sample #{idx} | TrueC={true_cls} PredC={pred_label} | "
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


def _make_predict_fn(model, classifier: str, device: torch.device):
    """Wrap the classifier into a numpy → probability function for SHAP."""

    def _predict(x: np.ndarray) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        data = torch.from_numpy(arr)
        if classifier == "tcn":
            logits, _ = predict_tcn(model, data, device=device)
        elif classifier == "tst":
            raise RuntimeError("TST model does not provide classification logits.")
        else:
            logits, _ = predict_tabnet(model, data, device=device)
        probs = torch.softmax(logits, dim=1)
        return probs.numpy()

    return _predict


def _extract_shap_vector(shap_exp: shap.Explanation, sample_idx: int, class_idx: int) -> np.ndarray:
    """Return the SHAP vector for the given sample/class regardless of layout."""

    values = shap_exp.values
    if values.ndim == 1:
        return np.asarray(values)
    if values.ndim == 2:
        return np.asarray(values[sample_idx])
    if values.ndim == 3:
        return np.asarray(values[sample_idx, :, class_idx])
    raise ValueError("Unexpected SHAP values shape")


def _extract_base_value(shap_exp: shap.Explanation, sample_idx: int, class_idx: int) -> float:
    base_vals = np.asarray(shap_exp.base_values)
    if base_vals.ndim == 0:
        return float(base_vals)
    if base_vals.ndim == 1:
        return float(base_vals[sample_idx])
    if base_vals.ndim == 2:
        return float(base_vals[sample_idx, class_idx])
    raise ValueError("Unexpected SHAP base_values shape")


def _compute_shap_summaries(
        model,
        classifier: str,
        device: torch.device,
        background: np.ndarray,
        samples: np.ndarray,
        sample_indices: List[int],
        pred_lookup: dict[int, int],
        feature_names: List[str],
) -> List[str]:
    """Compute SHAP attributions and return formatted summaries per sample."""

    if samples.size == 0:
        return []

    bg = np.asarray(background, dtype=np.float32)
    if bg.ndim == 1:
        bg = bg.reshape(1, -1)
    sample_arr = np.asarray(samples, dtype=np.float32)

    masker = shap.maskers.Independent(bg)
    predict_fn = _make_predict_fn(model, classifier, device)
    explainer = shap.Explainer(predict_fn, masker, algorithm="permutation")
    max_evals = 2 * sample_arr.shape[1] + 2048
    shap_exp = explainer(sample_arr, max_evals=max_evals)

    feature_list = ", ".join(feature_names)
    dataset_context = (
        "Model input is a numpy.ndarray[float32] shaped "
        f"({sample_arr.shape[0]}, {sample_arr.shape[1]}) with ordered features: "
        f"{feature_list}. Columns 'Reflectance' and 'loss' are intentionally excluded to avoid data leakage."
    )
    print(f"[SHAP] {dataset_context}")

    summaries: List[str] = [dataset_context]
    for local_idx, global_idx in enumerate(sample_indices):
        pred_cls = int(pred_lookup[int(global_idx)])
        shap_vec = _extract_shap_vector(shap_exp, local_idx, pred_cls)
        base_val = _extract_base_value(shap_exp, local_idx, pred_cls)

        predicted_prob = float(np.clip(base_val + shap_vec.sum(), 0.0, 1.0))
        top_idx = np.argsort(np.abs(shap_vec))[::-1]
        top_k = top_idx[:5]

        print(
            f"[SHAP] Sample {global_idx} → class {pred_cls}: base prob {base_val:.3f}, "
            f"pred prob {predicted_prob:.3f}."
        )
        print("        Top feature contributions (Δprobability):")
        for rank, j in enumerate(top_k, start=1):
            direction = "raises" if shap_vec[j] >= 0 else "lowers"
            print(
                f"          #{rank}: {feature_names[j]} {shap_vec[j]:+.4f} ({direction} class {pred_cls} probability)"
            )

        shap_contribs = ", ".join(
            f"#{idx + 1} {feature_names[j]} ({shap_vec[j]:+.3f})" for idx, j in enumerate(top_k)
        )
        pos_total = float(np.sum(shap_vec[shap_vec > 0]))
        neg_total = float(np.sum(shap_vec[shap_vec < 0]))
        summary = (
            f"Sample {global_idx} → class {pred_cls} | base prob {base_val:.3f} → predicted {predicted_prob:.3f}. "
            f"Top drivers: {shap_contribs}. Σpositive={pos_total:+.3f}, Σnegative={neg_total:+.3f}."
        )
        summaries.append(summary)

    return summaries


def _llm_explain(
        img_paths: List[Path], classifier_type: str = "tcn",
        openai_model: str = "gpt-4o-mini",
        shap_summaries: List[str] | None = None,
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
        retrieved = retrieve(query, k=5)  # ← may raise / be empty
    except Exception as exc:
        print(f"RAG retrieval failed. {exc}")
        retrieved = []

    ref_block = "\n\n".join(f"[{i + 1}] {r['text']}" for i, r in enumerate(retrieved))

    # ---------- build messages ------------------------------------------
    system_prompt = (
        "You are an optical-fibre fault-analysis expert. "
        "Given the following figures (each shows amplitude over P-points with "
        f"predictions vs ground truth in the title predicted using a {classifier_type} machine learning model), "
        "write a concise explanation for each figure "
        "of common patterns you observe, including typical failure modes and "
        "any misclassifications. Explain the type of fault, position, possible causes "
        "and possible solutions. Provide brief answers.\n\n"
        "Use the reference snippets, the SHAP feature attributions, and each image when required, "
        "citing snippets like [1], [2] where appropriate. When SHAP highlights features, incorporate that evidence in your reasoning. "
        "Specify the information provided by the SHAP values.\n\n"
        "Fault Classes are labelled as follows:\n"
        "id\tfault type\t\t\ttypical signs\n"
        "0\tnormal / no fault\t\tbaseline trace, loss ≈ 0, position = 0\n"
        "1\tfiber tapping\t\tlocalized disturbance, moderate loss due to coupler, reflectance can be low/absent\n"
        "2\tbad splice\t\t\tlocalized event with excess loss, small/possible reflection\n"
        "3\tbending event\t\tgradual/medium loss, usually no clear reflectance peak\n"
        "4\tdirty connector\t\tconnector-like event with extra loss and messy/variable reflectance\n"
        "5\tfiber cut\t\t\tabrupt large loss/end-of-trace, may appear near end position\n"
        "6\tPC connector\t\tclean connector-type reflective event, expected position\n"
        "7\treflector\t\t\tstrongly reflective event, high reflectance value\n"
    )

    # first part: reference snippets (if any) + lead‑in text
    user_parts: List[dict] = [
        {"type": "text",
         "text": "Reference snippets:\n" + (ref_block or "*<no snippets retrieved>*")},
    ]
    if shap_summaries:
        shap_text = "\n".join(shap_summaries)
        user_parts.append({"type": "text", "text": "SHAP attributions per sample:\n" + shap_text})
    user_parts.append({"type": "text", "text": "Here are the selected samples for inspection:"})
    user_parts += [
                     {  # the images themselves
                         "type": "image_url",
                         "image_url": {"url": _b64(p)},
                     }
                     for p in img_paths
                 ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_parts},
    ]

    # ---------- chat completion -----------------------------------------
    resp = client.chat.completions.create(
        model=openai_model,
        messages=messages,
        # max_completion_tokens=1000 # limit the response length
    )

    rag_flag = False
    if retrieved:
        rag_flag = True
        print("RAG retrieval successful, using retrieved snippets in LLM prompt.")

    return resp.choices[0].message.content.strip(), rag_flag


# --------------------------------------------------
# Main eval flow
# --------------------------------------------------
@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--mode",
    type=click.Choice(["pipeline", "direct"], case_sensitive=False),
    required=True,
    help="Evaluation mode: pipeline (GRU-AE filter) or direct (full test set).",
)
@click.option(
    "--classifier",
    type=click.Choice(["tcn", "tst", "tab"], case_sensitive=False),
    required=True,
    help="Classifier to use.",
)
@click.option(
    "--data", "data_path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("data/OTDR_DATA.csv"),
    show_default=True,
    help="Path to the dataset CSV. Must have 'Class', 'SNR', 'Position', and P{N} columns.",
)
@click.option(
    "--detector",
    type=click.Path(dir_okay=False, path_type=Path),
    default=Path("models/gru_ae.pt"),
    show_default=True,
    help="Path to GRU-AE weights.",
)
@click.option(
    "--cls-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to classifier weights; defaults by --classifier.",
)
@click.option(
    "--num-samples",
    type=int,
    default=4,
    show_default=True,
    help="Random samples to visualise & explain.",
)
@click.option(
    "--out-dir",
    type=str,
    default="eval_outputs",
    show_default=True,
    help="Folder name under outputs/ for artifacts.",
)
@click.option(
    "--device",
    type=str,
    default=None,
    help="cuda | cpu | leave empty for auto-detect.",
)
def main(mode, classifier, data_path, detector, cls_path, num_samples, out_dir, device):  # noqa: C901
    out_dir = Path("outputs") / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- data ---------- #
    df = load_raw_dataframe(data_path)
    _, _, test_df = make_splits(df)

    meas_cols = [c for c in test_df.columns if re.fullmatch(r"P\d+", c)] + ["SNR"]
    leakage_cols = {"Reflectance", "loss", "Loss"}
    leaked = [c for c in meas_cols if c in leakage_cols]
    if leaked:
        raise ValueError(
            "Measurement column selection must not include leakage features, found: "
            + ", ".join(leaked)
        )
    print(
        "[INFO] Using measurement columns (ordered): "
        + ", ".join(meas_cols)
        + ". 'Reflectance' and 'loss' are excluded from model inputs."
    )

    # test scaling only using training dataset info
    scaler_path = Path(detector).parent / "scaler.json"
    if scaler_path.exists():
        meta = json.loads(scaler_path.read_text())
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(meta["mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(meta["scale"], dtype=np.float32)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
    else:
        meta = json.loads(Path(detector).with_suffix(".json").read_text())
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(meta["scaler_mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(meta["scaler_scale"], dtype=np.float32)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = scaler.mean_.shape[0]
    splits = tensorise_splits(test_df, test_df, test_df, scaler)  # only need "test" key
    X_test = splits["test"].X
    y_cls_test = splits["test"].y_class
    y_pos_test = splits["test"].y_pos

    if classifier == "tst":
        fault_mask = y_cls_test != 0
        if fault_mask.sum().item() == 0:
            raise ValueError("No faulty samples available in the test set for TST evaluation.")
        X_test = X_test[fault_mask]
        y_cls_test = y_cls_test[fault_mask]
        y_pos_test = y_pos_test[fault_mask]

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None else torch.device(device)
    )
    print("[INFO] Using device:", device)

    # ---------- load models ---------- #
    cls_default = "tabnet.pt" if classifier == "tab" else ("tcn.pt" if classifier == "tcn" else "tst.pt")
    cls_path = Path(cls_path or Path("models") / cls_default)

    n_classes = int(df["Class"].max() + 1)
    classifier_model = _load_classifier(classifier, cls_path, seq_len=X_test.shape[1], n_classes=n_classes,
                                        device=device)

    if mode == "pipeline":
        ae, threshold, _, _ = _load_gru_ae(Path(detector), device)
        errs = reconstruction_error(ae, X_test, device=device)
        is_fault = errs > threshold
        # if all healthy, fallback to sample of test
        idx_to_eval = torch.nonzero(is_fault).squeeze(-1)
        if idx_to_eval.numel() == 0:
            idx_to_eval = torch.arange(0, min(1000, X_test.size(0)))
    else:  # direct
        idx_to_eval = torch.arange(X_test.size(0))

    # ------------- inference ------------- #
    if classifier == "tcn":
        logits, pos_hat = predict_tcn(classifier_model, X_test[idx_to_eval])
        preds_cls = logits.argmax(1)
    elif classifier == "tst":
        pos_hat = predict_tst(classifier_model, X_test[idx_to_eval])
        preds_cls = None
    else:  # tab
        logits, pos_hat = predict_tabnet(classifier_model, X_test[idx_to_eval])
        preds_cls = logits.argmax(1)

    rmse = root_mean_squared_error(y_pos_test[idx_to_eval].numpy(), pos_hat.numpy())
    if preds_cls is not None:
        acc = accuracy_score(y_cls_test[idx_to_eval].numpy(), preds_cls.numpy())
        print(
            f"Eval subset size = {idx_to_eval.size(0)} | Acc = {acc:.3f} | RMSE = {rmse:.3f}"
        )  # noqa: T201
        y_true = y_cls_test[idx_to_eval].numpy()
        y_pred = preds_cls.numpy()
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=3))

        # Confusion matrix plot
        cm = confusion_matrix(y_cls_test[idx_to_eval].numpy(), preds_cls.numpy())
        ConfusionMatrixDisplay(cm).plot(include_values=True, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix – Eval subset")
        plt.tight_layout()
        cm_path = out_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
    else:
        print(
            f"Eval subset size = {idx_to_eval.size(0)} | RMSE = {rmse:.3f}"
        )  # noqa: T201

    # ------------- random visualisations ------------- #
    rng = np.random.default_rng(42)
    chosen = rng.choice(idx_to_eval.numpy(), size=min(num_samples, idx_to_eval.size(0)), replace=False)

    # ------------- SHAP explainability ------------- #
    shap_summaries: List[str] = []
    if preds_cls is not None:
        try:
            idx_eval_cpu = idx_to_eval.detach().cpu()
            preds_cpu = preds_cls.detach().cpu()
            pred_lookup = {int(idx_eval_cpu[i].item()): int(preds_cpu[i].item()) for i in range(idx_eval_cpu.size(0))}
            bg_size = min(50, idx_eval_cpu.size(0))
            if bg_size > 0 and chosen.size > 0:
                background = X_test[idx_eval_cpu[:bg_size]].numpy()
                sample_tensor = torch.as_tensor(chosen, dtype=torch.long)
                shap_samples = X_test[sample_tensor].numpy()
                shap_summaries = _compute_shap_summaries(
                    classifier_model,
                    classifier,
                    device,
                    background,
                    shap_samples,
                    chosen.tolist(),
                    pred_lookup,
                    meas_cols,
                )
        except Exception as exc:  # pragma: no cover - fallback path
            print(f"[WARN] SHAP computation failed: {exc}")
            shap_summaries = []

    img_paths = []
    num_points = X_test.shape[1] - 1  # number of P-points in the traces
    for idx in chosen:
        amp = X_test[idx][:num_points].numpy() * scaler.scale_[:num_points] + scaler.mean_[:num_points]
        snr = float(X_test[idx][num_points].item() * scaler.scale_[num_points] + scaler.mean_[num_points])
        t_cls = int(y_cls_test[idx].item())
        if preds_cls is None:
            p_cls = None
        else:
            p_cls = int(preds_cls[idx_to_eval == idx][0].item())
        t_pos = float(y_pos_test[idx].item())
        p_pos = float(pos_hat[idx_to_eval == idx][0].item())
        img_paths.append(_visualise_sample(amp, snr, t_cls, p_cls, t_pos, p_pos, int(idx), out_dir))

    # ------------- LLM explanation ------------- #
    explanation, rag_flag = _llm_explain(img_paths, openai_model= "gpt-5", classifier_type=classifier, shap_summaries=shap_summaries)
    classifier_name = classifier.upper()
    llm_dir = Path("outputs/llm_output")
    llm_dir.mkdir(parents=True, exist_ok=True)
    if explanation:
        explanation_file = llm_dir / "llm_explanation_shap.txt"
        i = 1
        while explanation_file.exists():
            explanation_file = llm_dir / f"llm_explanation_shap_{i}.txt"
            i += 1
        if rag_flag:
            explanation = f"LLM explanation for eval subset with RAG for {classifier_name} in {mode} mode:\n\n{explanation}"
        else:
            explanation = f"LLM explanation for eval subset without RAG for {classifier_name} in {mode} mode:\n\n{explanation}"
        explanation_file.write_text(explanation, encoding='utf-8')
        print(f"LLM explanation saved to {explanation_file.name}")  # noqa: T201

if __name__ == "__main__":
    main()
