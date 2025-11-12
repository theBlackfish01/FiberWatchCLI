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
from typing import Any, List, Tuple
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from sklearn.metrics import (
    accuracy_score,
    root_mean_squared_error,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    roc_auc_score,
)
from data_helper import (
    load_raw_dataframe,
    make_splits,
    tensorise_splits,
    measurement_columns,
)
from model_functions.gruae import VectorGRUAE, reconstruction_error
from model_functions.tcn import OTDR_TCN, predict as predict_tcn
from model_functions.tcn_binary import OTDR_TCNBinary, predict as predict_tcn_binary
from model_functions.tst import TimeSeriesTransformer, predict as predict_tst
from model_functions.tabnet import OTDR_TabNet, predict as predict_tabnet
import config.config as cfg
from pathlib import Path
from rag import retrieve
from openai import OpenAI
import warnings


warnings.filterwarnings("ignore", category=FutureWarning)  # noqa: T201
client = OpenAI(api_key=cfg.OPENAI_API_KEY)


def _extract_response_text(resp: Any) -> str:
    """Best-effort extraction of text content from the Responses API result."""

    output_text = getattr(resp, "output_text", None)
    if isinstance(output_text, str) and output_text.strip():
        return output_text.strip()

    texts: list[str] = []
    for item in getattr(resp, "output", []) or []:
        for content in getattr(item, "content", []) or []:
            text_obj = getattr(content, "text", None)
            if isinstance(text_obj, str):
                texts.append(text_obj)
            else:
                value = getattr(text_obj, "value", None)
                if isinstance(value, str):
                    texts.append(value)

    if texts:
        return "\n".join(t.strip() for t in texts if t.strip())

    raise RuntimeError("No textual content returned by the Responses API")


def _call_responses_api(
        llm_client: OpenAI,
        model: str,
        system_text: str,
        user_content: list[dict[str, Any]],
) -> str:
    """Invoke the modern Responses API for multimodal prompts."""

    resp = llm_client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ],
    )
    return _extract_response_text(resp)

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
    elif kind == "tcn_binary":
        model = OTDR_TCNBinary()
        model.load_state_dict(torch.load(cls_path, map_location=device))
    elif kind == "tst":
        model = TimeSeriesTransformer(seq_len=seq_len)
        model.load_state_dict(torch.load(cls_path, map_location=device))
    elif kind == "tab":
        model = OTDR_TabNet(n_classes=n_classes)
        model.load_state_dict(torch.load(cls_path, map_location=device))
    else:
        raise ValueError("classifier kind must be 'tcn', 'tcn_binary', 'tab' or 'tst'")
    return model.eval().to(device)


def _load_classifier_meta(cls_path: Path) -> dict[str, Any] | None:
    meta_path = cls_path.with_suffix(".json")
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text())


def _remap_anomaly_only_targets(
    X: torch.Tensor,
    y_cls: torch.Tensor,
    y_pos: torch.Tensor,
    meta: dict[str, Any],
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[int, int],
    torch.Tensor,
]:
    mapping_raw = meta.get("class_index_map")
    if mapping_raw is not None:
        mapping = {int(k): int(v) for k, v in mapping_raw.items()}
    else:
        original_classes = meta.get("original_classes")
        if not original_classes:
            raise ValueError(
                "Anomaly-only TCN metadata must include 'class_index_map' or 'original_classes'."
            )
        mapping = {int(orig): idx for idx, orig in enumerate(original_classes)}

    mask = torch.zeros_like(y_cls, dtype=torch.bool)
    for orig in mapping:
        mask |= y_cls == int(orig)

    selected = torch.nonzero(mask, as_tuple=True)[0]
    if selected.numel() == 0:
        raise ValueError(
            "No samples with the anomaly classes required by the anomaly-only TCN were found."
        )

    X_sel = X[selected]
    y_pos_sel = y_pos[selected]
    y_cls_sel = y_cls[selected]

    remapped = torch.empty_like(y_cls_sel)
    for orig, idx in mapping.items():
        remapped[y_cls_sel == int(orig)] = int(idx)

    return X_sel, remapped.to(dtype=torch.long), y_pos_sel, mapping, selected


def _visualise_sample(
        amps: np.ndarray,
        snr: float,
        true_cls: int,
        pred_cls: int | None,
        true_pos: float,
        pred_pos: float | None,
        idx: int,
        out_dir: Path,
):
    plt.figure(figsize=(10, 8))
    plt.plot(np.arange(amps.size), amps, label="Amplitude")
    pred_label = "N/A" if pred_cls is None else str(pred_cls)
    pred_pos_str = "N/A" if pred_pos is None else f"{pred_pos:.3f}"
    plt.title(
        f"Sample #{idx} | TrueC={true_cls} PredC={pred_label} | "
        f"TruePos={true_pos:.3f}m  PredPos={pred_pos_str}m | SNR={snr:.2f}"
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
        elif classifier == "tcn_binary":
            logits = predict_tcn_binary(model, data, device=device)
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

def _llm_explain_with_self_reflection(
        img_paths: List[Path],
        classifier_type: str = "tcn",
        openai_model: str = "gpt-4o-mini",
        shap_summaries: List[str] | None = None,
) -> tuple[str, str, bool] | None:
    """
    DIRECT pass -> SELF-REFLECTION pass with explicit TrueC/PredC handling.
    Returns (direct_text, refined_text, rag_used_flag) or None if no API key.
    """
    api_key = cfg.OPENAI_API_KEY
    if not api_key:
        print("OPENAI_API_KEY not set – skipping LLM explanation")
        return None

    client = OpenAI(api_key=api_key)

    # ---------- Fault class block (shared) ---------------------------------
    fault_classes_block = (
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

    # ---------- RAG retrieval (shared) -------------------------------------
    query = "OTDR fault plots – " + ", ".join(p.stem for p in img_paths)
    try:
        retrieved = retrieve(query, k=5)
    except Exception as exc:
        print(f"RAG retrieval failed. {exc}")
        retrieved = []
    ref_block = "\n\n".join(f"[{i + 1}] {r['text']}" for i, r in enumerate(retrieved))
    rag_flag = bool(retrieved)
    if rag_flag:
        print("RAG retrieval successful, using retrieved snippets in LLM prompt.")

    # ---------- SHAP text (shared) -----------------------------------------
    shap_text = "\n".join(shap_summaries) if shap_summaries else ""

    # ---------- DIRECT pass -------------------------------------------------
    # IMPORTANT: disambiguate TrueC vs PredC and force wording when they differ
    true_pred_rules = (
        "READ TITLES CAREFULLY: each figure title contains 'TrueC=<int> PredC=<int>'.\n"
        "- 'TrueC' is the ground-truth class.\n"
        "- 'PredC' is the model's predicted class (may be 'N/A' for regression-only models).\n"
        "- If TrueC != PredC, explicitly write: \"misclassified as <PredC> (true: <TrueC>)\".\n"
        "- NEVER swap or rename these; do not call TrueC the prediction or PredC the truth.\n"
    )

    system_direct = (
        "You are an optical-fibre fault-analysis expert. "
        "Given the following figures (OTDR amplitude over P-points; titles include TrueC/PredC and positions), "
        f"write a concise explanation for each figure predicted by a {classifier_type} model. "
        "Explain fault type, position, likely causes, and concrete next actions. "
        "Use the reference snippets and the SHAP feature attributions when available. "
        "Cite snippets like [1], [2] when used. If SHAP is present, explicitly state which features raised/lowered "
        "the predicted class probability.\n\n"
        + true_pred_rules
        + fault_classes_block
    )

    user_direct_parts: List[dict[str, Any]] = [
        {"type": "input_text", "text": "Reference snippets:\n" + (ref_block or "*<no snippets retrieved>*")},
    ]
    if shap_text:
        user_direct_parts.append({"type": "input_text", "text": "SHAP attributions per sample:\n" + shap_text})
    user_direct_parts.append({"type": "input_text", "text": "Selected samples for inspection (images):"})
    user_direct_parts += [
        {"type": "input_image", "image_url": _b64(p)} for p in img_paths
    ]

    direct_text = _call_responses_api(client, openai_model, system_direct, user_direct_parts)

    # ---------- SELF-REFLECTION pass ---------------------------------------
    # Provide SAME images so the reviewer can verify visually.
    # Repeat the TrueC/PredC rule to avoid drift.
    system_reflect = (
        "You are a meticulous QA reviewer for optical-fibre explanations. "
        "You will receive: (a) the same context (reference snippets, SHAP summaries, and the images), and (b) a DRAFT explanation. "
        "OUTPUT ONLY an improved explanation that:\n"
        "1) Matches SHAP signs (positive SHAP → increases predicted class probability; negative → decreases).\n"
        "2) Mentions the top-k absolute SHAP contributors (k≈5) in plain English.\n"
        "3) Grounds standards/definitions with citations [i] that exist in the provided snippet list.\n"
        "4) Avoids hallucinated numbers; if a number isn’t present, use cautious wording or a justified range.\n"
        "5) Keeps the operator section actionable (2–3 steps).\n\n"
        + true_pred_rules
        + fault_classes_block
    )

    reflect_user_content: List[dict[str, Any]] = [
        {"type": "input_text", "text": "Reference snippets:\n" + (ref_block or "*<no snippets retrieved>*")},
    ]
    if shap_text:
        reflect_user_content.append({"type": "input_text", "text": "SHAP attributions per sample:\n" + shap_text})
    reflect_user_content.append({"type": "input_text", "text": "Images (verify titles with TrueC/PredC and positions):"})
    reflect_user_content += [
        {"type": "input_image", "image_url": _b64(p)} for p in img_paths
    ]
    reflect_user_content.append({"type": "input_text", "text": "DRAFT explanation to review:\n" + direct_text})

    refined_text = _call_responses_api(client, openai_model, system_reflect, reflect_user_content)

    return direct_text, refined_text, rag_flag



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
    user_parts: List[dict[str, Any]] = [
        {"type": "input_text",
         "text": "Reference snippets:\n" + (ref_block or "*<no snippets retrieved>*")},
    ]
    if shap_summaries:
        shap_text = "\n".join(shap_summaries)
        user_parts.append({"type": "input_text", "text": "SHAP attributions per sample:\n" + shap_text})
    user_parts.append({"type": "input_text", "text": "Here are the selected samples for inspection:"})
    user_parts += [
                     {
                         "type": "input_image",
                         "image_url": _b64(p),
                     }
                     for p in img_paths
                 ]

    resp_text = _call_responses_api(client, openai_model, system_prompt, user_parts)

    rag_flag = False
    if retrieved:
        rag_flag = True
        print("RAG retrieval successful, using retrieved snippets in LLM prompt.")

    return resp_text, rag_flag


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
    type=click.Choice(["tcn", "tcn_binary", "tst", "tab"], case_sensitive=False),
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
    "--tcn-anomaly-only/--tcn-all-data",
    "tcn_anomaly_only",
    default=False,
    help="When evaluating TCN models, select the anomaly-only classifier variant.",
)
@click.option(
    "--orchestrate-tst",
    is_flag=True,
    help="Chain binary TCN ➜ anomaly-only TCN ➜ TST for localisation evaluation.",
)
@click.option(
    "--cls-path",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Optional path to classifier weights; defaults by --classifier.",
)
@click.option(
    "--num-samples",
    type=click.IntRange(0, None),
    default=4,
    show_default=True,
    help="Random samples to visualise & explain (0 to skip explainability).",
)
@click.option(
    "--extra-feature",
    "extra_features",
    multiple=True,
    help=(
        "Optional additional feature columns to append to the default measurement "
        "set (repeat flag for multiple columns)."
    ),
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
def main(
    mode,
    classifier,
    data_path,
    detector,
    cls_path,
    num_samples,
    out_dir,
    device,
    tcn_anomaly_only,
    orchestrate_tst,
    extra_features,
):  # noqa: C901
    out_dir = Path("outputs") / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if classifier != "tcn" and tcn_anomaly_only:
        raise click.BadOptionUsage(
            "--tcn-anomaly-only",
            "The anomaly-only flag is only applicable when --classifier=tcn.",
        )

    extras = tuple(extra_features)

    # ---------- data ---------- #
    df = load_raw_dataframe(data_path)
    _, _, test_df = make_splits(df)

    scaler_path = Path(detector).parent / "scaler.json"
    scaler = StandardScaler()
    feature_names_meta: list[str] | None = None
    if scaler_path.exists():
        scaler_meta = json.loads(scaler_path.read_text())
        feature_names_meta = scaler_meta.get("feature_names")
        scaler.mean_ = np.asarray(scaler_meta["mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(scaler_meta["scale"], dtype=np.float32)
    else:
        detector_meta_path = Path(detector).with_suffix(".json")
        detector_meta = json.loads(detector_meta_path.read_text())
        feature_names_meta = detector_meta.get("feature_names")
        scaler.mean_ = np.asarray(detector_meta["scaler_mean"], dtype=np.float32)
        scaler.scale_ = np.asarray(detector_meta["scaler_scale"], dtype=np.float32)
    scaler.var_ = scaler.scale_ ** 2
    scaler.n_features_in_ = scaler.mean_.shape[0]

    if feature_names_meta:
        meas_cols = list(feature_names_meta)
        missing_cols = [c for c in meas_cols if c not in test_df.columns]
        if missing_cols:
            raise ValueError(
                "Dataset is missing feature columns required by the scaler metadata: "
                + ", ".join(missing_cols)
            )
        if extras:
            missing_requested = [c for c in extras if c not in meas_cols]
            if missing_requested:
                raise click.BadOptionUsage(
                    "--extra-feature",
                    "Requested additional feature(s) not present in the saved scaler metadata: "
                    + ", ".join(missing_requested),
                )
    else:
        try:
            meas_cols = measurement_columns(test_df, extras)
        except KeyError as exc:
            raise click.BadOptionUsage("--extra-feature", str(exc)) from exc

    if len(meas_cols) != scaler.n_features_in_:
        raise ValueError(
            "Scaler metadata dimensionality does not match selected measurement columns."
        )

    leakage_cols = {"Reflectance", "loss", "Loss"}
    leaked = [c for c in meas_cols if c in leakage_cols]
    if leaked and not extras:
        raise ValueError(
            "Measurement column selection must not include leakage features, found: "
            + ", ".join(leaked)
        )
    if leaked and extras:
        print(
            "[WARN] Additional features include potential leakage columns: "
            + ", ".join(leaked)
        )

    print("[INFO] Using measurement columns (ordered): " + ", ".join(meas_cols))
    if extras:
        print("[INFO] Extra features appended: " + ", ".join(extras))

    splits = tensorise_splits(
        test_df,
        test_df,
        test_df,
        scaler,
        measurement_override=meas_cols,
    )  # only need "test" key
    X_test = splits["test"].X
    y_cls_test = splits["test"].y_class
    y_pos_test = splits["test"].y_pos

    if classifier == "tst" and not orchestrate_tst:
        fault_mask = y_cls_test != 0
        if fault_mask.sum().item() == 0:
            raise ValueError("No faulty samples available in the test set for TST evaluation.")
        X_test = X_test[fault_mask]
        y_cls_test = y_cls_test[fault_mask]
        y_pos_test = y_pos_test[fault_mask]
        class_feature = y_cls_test.to(dtype=X_test.dtype).unsqueeze(1)
        tst_features = torch.cat([class_feature, X_test], dim=1)
    else:
        tst_features = None

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device is None else torch.device(device)
    )
    print("[INFO] Using device:", device)

    # ---------- load models ---------- #
    if classifier == "tab":
        cls_default = "tabnet.pt"
    elif classifier == "tcn":
        cls_default = "tcn_anomaly.pt" if tcn_anomaly_only else "tcn_full.pt"
    elif classifier == "tcn_binary":
        cls_default = "tcn_binary.pt"
    else:
        cls_default = "tst.pt"
    cls_path = Path(cls_path or Path("models") / cls_default)

    cls_meta = _load_classifier_meta(cls_path)

    default_n_classes = int(df["Class"].max() + 1)

    if orchestrate_tst:
        if classifier != "tst":
            raise click.BadOptionUsage(
                "--orchestrate-tst",
                "The chained orchestrator is only available when --classifier=tst.",
            )

        model_dir = cls_path.parent
        binary_path = model_dir / "tcn_binary.pt"
        anomaly_path = model_dir / "tcn_anomaly.pt"

        _run_tst_orchestrator(
            X_test=X_test,
            y_cls_test=y_cls_test,
            y_pos_test=y_pos_test,
            device=device,
            out_dir=out_dir,
            binary_path=binary_path,
            anomaly_path=anomaly_path,
            tst_path=cls_path,
            default_n_classes=default_n_classes,
        )
        return

    if classifier == "tcn_binary":
        n_classes = 2
    elif classifier == "tcn":
        variant = "anomaly_only" if tcn_anomaly_only else "full"
        if cls_meta and "variant" in cls_meta:
            meta_variant = cls_meta["variant"]
            if meta_variant not in {"full", "anomaly_only"}:
                raise ValueError(f"Unknown TCN variant '{meta_variant}' in metadata.")
            if meta_variant != variant:
                if tcn_anomaly_only:
                    raise ValueError(
                        "Requested anomaly-only TCN but provided checkpoint metadata "
                        "describes the full-data variant."
                    )
                else:
                    print(
                        f"[INFO] Detected '{meta_variant}' TCN variant from metadata; adjusting evaluation."
                    )
                    variant = meta_variant
        tcn_anomaly_only = variant == "anomaly_only"
        if tcn_anomaly_only:
            if cls_meta is None:
                raise ValueError(
                    "Anomaly-only TCN checkpoints must include metadata for class remapping."
                )
            class_labels_meta = cls_meta.get("class_labels") or cls_meta.get("original_classes")
            if not class_labels_meta:
                raise ValueError(
                    "Anomaly-only TCN metadata must include 'class_labels' or 'original_classes'."
                )
            n_classes = len(class_labels_meta)
        else:
            if cls_meta and "n_classes" in cls_meta:
                n_classes = int(cls_meta["n_classes"])
            else:
                n_classes = default_n_classes
    else:
        n_classes = default_n_classes

    anomaly_mapping: dict[int, int] | None = None
    if classifier == "tcn" and tcn_anomaly_only:
        X_test, y_cls_test, y_pos_test, anomaly_mapping, _ = _remap_anomaly_only_targets(
            X_test,
            y_cls_test,
            y_pos_test,
            cls_meta or {},
        )
        mapping_desc = ", ".join(
            f"{orig}→{idx}" for orig, idx in sorted(anomaly_mapping.items())
        )
        print(f"[INFO] Evaluating anomaly-only TCN with mapping: {mapping_desc}")

    seq_len = tst_features.shape[1] if tst_features is not None else X_test.shape[1]
    classifier_model = _load_classifier(
        classifier,
        cls_path,
        seq_len=seq_len,
        n_classes=n_classes,
        device=device,
    )

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
    pos_hat = None
    logits = None
    preds_cls = None
    if classifier == "tcn":
        logits, pos_hat = predict_tcn(classifier_model, X_test[idx_to_eval])
        preds_cls = logits.argmax(1)
    elif classifier == "tcn_binary":
        logits = predict_tcn_binary(classifier_model, X_test[idx_to_eval])
        preds_cls = logits.argmax(1)
    elif classifier == "tst":
        pos_hat = predict_tst(classifier_model, tst_features[idx_to_eval])
        preds_cls = None
    else:  # tab
        logits, pos_hat = predict_tabnet(classifier_model, X_test[idx_to_eval])
        preds_cls = logits.argmax(1)

    if pos_hat is not None:
        rmse = root_mean_squared_error(y_pos_test[idx_to_eval].numpy(), pos_hat.numpy())
    else:
        rmse = None

    if preds_cls is not None:
        acc = accuracy_score(y_cls_test[idx_to_eval].numpy(), preds_cls.numpy())
        if rmse is not None:
            print(
                f"Eval subset size = {idx_to_eval.size(0)} | Acc = {acc:.3f} | RMSE = {rmse:.3f}"
            )  # noqa: T201
        else:
            print(
                f"Eval subset size = {idx_to_eval.size(0)} | Acc = {acc:.3f}"
            )  # noqa: T201
        y_true = y_cls_test[idx_to_eval].numpy()
        y_pred = preds_cls.numpy()
        print("\nClassification report:")
        print(classification_report(y_true, y_pred, digits=3))
        if logits is not None and logits.shape[1] == 2:
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()
            auc_val = roc_auc_score(y_true, probs)
            print(f"AUC = {auc_val:.3f}")

        # Confusion matrix plot
        cm = confusion_matrix(y_cls_test[idx_to_eval].numpy(), preds_cls.numpy())
        ConfusionMatrixDisplay(cm).plot(include_values=True, cmap="Blues", colorbar=False)
        plt.title("Confusion Matrix – Eval subset")
        plt.tight_layout()
        cm_path = out_dir / "confusion_matrix.png"
        plt.savefig(cm_path, dpi=150)
        plt.close()
    else:
        if rmse is not None:
            print(
                f"Eval subset size = {idx_to_eval.size(0)} | RMSE = {rmse:.3f}"
            )  # noqa: T201

    # ------------- random visualisations ------------- #
    rng = np.random.default_rng(42)
    if num_samples <= 0 or idx_to_eval.size(0) == 0:
        chosen = np.empty(0, dtype=int)
    else:
        chosen = rng.choice(
            idx_to_eval.numpy(),
            size=min(num_samples, idx_to_eval.size(0)),
            replace=False,
        )

    # ------------- SHAP explainability ------------- #
    shap_summaries: List[str] = []
    if preds_cls is not None and chosen.size > 0:
        try:
            idx_eval_cpu = idx_to_eval.detach().cpu()
            preds_cpu = preds_cls.detach().cpu()
            pred_lookup = {
                int(idx_eval_cpu[i].item()): int(preds_cpu[i].item())
                for i in range(idx_eval_cpu.size(0))
            }
            bg_size = min(50, idx_eval_cpu.size(0))
            if bg_size > 0:
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

    img_paths: list[Path] = []
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
        if pos_hat is None:
            p_pos = None
        else:
            p_pos = float(pos_hat[idx_to_eval == idx][0].item())
        img_paths.append(_visualise_sample(amp, snr, t_cls, p_cls, t_pos, p_pos, int(idx), out_dir))

    explain_pair: tuple[str, str, bool] | None = None
    if img_paths:
        # ------------- LLM explanation (direct + self-reflection) ------------- #
        explain_pair = _llm_explain_with_self_reflection(
            img_paths,
            openai_model="gpt-5",  # keep your configured model string
            classifier_type=classifier,
            shap_summaries=shap_summaries
        )

    classifier_name = classifier.upper()
    llm_dir = Path("outputs/llm_output")
    llm_dir.mkdir(parents=True, exist_ok=True)

    if explain_pair:
        direct_text, refined_text, rag_flag = explain_pair
        explanation_file = llm_dir / "llm_explanation_shap.txt"
        i = 1
        while explanation_file.exists():
            explanation_file = llm_dir / f"llm_explanation_shap_{i}.txt"
            i += 1

        header = (
            f"LLM explanation for eval subset {'with' if rag_flag else 'without'} RAG "
            f"for {classifier_name} in {mode} mode:\n\n"
        )
        combined = (
                header
                + "=== DIRECT ===\n"
                + direct_text.strip()
                + "\n\n=== SELF-REFLECTION (REVISED) ===\n"
                + refined_text.strip()
                + "\n"
        )
        explanation_file.write_text(combined, encoding="utf-8")
        print(f"LLM explanation (direct + self-reflection) saved to {explanation_file.name}")  # noqa: T201


def _run_tst_orchestrator(
        *,
        X_test: torch.Tensor,
        y_cls_test: torch.Tensor,
        y_pos_test: torch.Tensor,
        device: torch.device,
        out_dir: Path,
        binary_path: Path,
        anomaly_path: Path,
        tst_path: Path,
        default_n_classes: int,
) -> None:
    """Evaluate the chained Binary-TCN ➜ anomaly-TCN ➜ TST pipeline."""

    total_samples = int(X_test.size(0))
    actual_anomalies = int((y_cls_test != 0).sum().item())

    # ---------- Stage 1: binary anomaly filter ---------- #
    binary_model = _load_classifier(
        "tcn_binary",
        binary_path,
        seq_len=X_test.shape[1],
        n_classes=2,
        device=device,
    )
    binary_logits = predict_tcn_binary(binary_model, X_test, device=device)
    binary_probs = torch.softmax(binary_logits, dim=1)
    binary_preds = binary_probs.argmax(1)
    binary_truth = (y_cls_test != 0).to(dtype=torch.long)

    binary_acc = accuracy_score(
        binary_truth.cpu().numpy(),
        binary_preds.cpu().numpy(),
    )
    binary_auc: float | None = None
    if torch.unique(binary_truth).numel() == 2:
        try:
            binary_auc = roc_auc_score(
                binary_truth.cpu().numpy(),
                binary_probs[:, 1].cpu().numpy(),
            )
        except ValueError:
            binary_auc = None

    anomaly_indices = torch.nonzero(binary_preds == 1, as_tuple=True)[0]
    anomaly_count = int(anomaly_indices.numel())

    # ---------- Stage 2: anomaly-only multi-class TCN ---------- #
    anomaly_meta = _load_classifier_meta(anomaly_path)
    if anomaly_meta is None:
        raise ValueError(
            "Anomaly-only TCN checkpoint metadata is required for the orchestrated pipeline."
        )

    _, _remapped_truth, _, mapping, _ = _remap_anomaly_only_targets(
        X_test,
        y_cls_test,
        y_pos_test,
        anomaly_meta,
    )
    inv_mapping = {int(v): int(k) for k, v in mapping.items()}
    anomaly_n_classes = len(mapping)

    anomaly_model = _load_classifier(
        "tcn",
        anomaly_path,
        seq_len=X_test.shape[1],
        n_classes=anomaly_n_classes,
        device=device,
    )

    stage2_pred_lookup: dict[int, int] = {}
    stage2_accuracy: float | None = None
    stage2_eval_count = 0
    if anomaly_indices.numel() > 0:
        stage2_logits, _ = predict_tcn(
            anomaly_model,
            X_test[anomaly_indices],
            device=device,
        )
        stage2_preds_remap = stage2_logits.argmax(1).cpu()
        stage2_preds_orig = [inv_mapping[int(cls.item())] for cls in stage2_preds_remap]
        stage2_pred_lookup = {
            int(idx.item()): int(stage2_preds_orig[pos])
            for pos, idx in enumerate(anomaly_indices.cpu())
        }

        # Accuracy only on samples whose ground-truth class belongs to the anomaly mapping
        stage2_truth: list[int] = []
        stage2_preds_eval: list[int] = []
        for idx in anomaly_indices.cpu().tolist():
            true_cls = int(y_cls_test[idx].item())
            if true_cls in mapping:
                stage2_truth.append(true_cls)
                stage2_preds_eval.append(stage2_pred_lookup[idx])
        stage2_eval_count = len(stage2_truth)
        if stage2_truth:
            stage2_accuracy = accuracy_score(stage2_truth, stage2_preds_eval)

    # ---------- Stage 3: TST localisation ---------- #
    stage3_rmse: float | None = None
    stage3_count = 0
    if anomaly_indices.numel() > 0:
        class_feature = torch.tensor(
            [float(stage2_pred_lookup.get(int(idx.item()), 0)) for idx in anomaly_indices],
            dtype=X_test.dtype,
        ).unsqueeze(1)
        tst_input = torch.cat([class_feature, X_test[anomaly_indices]], dim=1)
        tst_model = _load_classifier(
            "tst",
            tst_path,
            seq_len=tst_input.shape[1],
            n_classes=default_n_classes,
            device=device,
        )
        pos_hat = predict_tst(tst_model, tst_input, device=device)
        stage3_count = int(pos_hat.size(0))
        if stage3_count > 0:
            stage3_rmse = root_mean_squared_error(
                y_pos_test[anomaly_indices].cpu().numpy(),
                pos_hat.cpu().numpy(),
            )

    # ---------- Aggregate predictions ---------- #
    final_preds = torch.zeros_like(y_cls_test)
    for idx, pred_cls in stage2_pred_lookup.items():
        final_preds[idx] = int(pred_cls)

    y_true_np = y_cls_test.cpu().numpy()
    y_pred_np = final_preds.cpu().numpy()

    report = classification_report(y_true_np, y_pred_np, digits=3)
    cm = confusion_matrix(y_true_np, y_pred_np)
    ConfusionMatrixDisplay(cm).plot(include_values=True, cmap="Blues", colorbar=False)
    plt.title("Confusion Matrix – Binary➜TCN➜TST pipeline")
    plt.tight_layout()
    cm_path = out_dir / "confusion_matrix_orchestrator.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ---------- Textual summary ---------- #
    summary_lines = [
        (
            "Stage 1 – Binary anomaly filter: "
            f"accuracy={binary_acc:.3f}, auc={binary_auc:.3f if binary_auc is not None else 'N/A'}, "
            f"predicted {anomaly_count}/{total_samples} traces as faulty (ground-truth faults: {actual_anomalies})."
        ),
    ]

    if stage2_pred_lookup:
        acc_str = f"{stage2_accuracy:.3f}" if stage2_accuracy is not None else "N/A"
        summary_lines.append(
            "Stage 2 – Anomaly-only TCN: "
            f"accuracy={acc_str} over {stage2_eval_count} mapped faults; "
            f"issued predictions for {len(stage2_pred_lookup)} traces."
        )
    else:
        summary_lines.append(
            "Stage 2 – Anomaly-only TCN: no anomaly predictions received from the binary filter."
        )

    if stage3_count > 0:
        rmse_str = f"{stage3_rmse:.3f}" if stage3_rmse is not None else "N/A"
        summary_lines.append(
            "Stage 3 – Time-series transformer localisation: "
            f"RMSE={rmse_str} m over {stage3_count} traces."
        )
    else:
        summary_lines.append(
            "Stage 3 – Time-series transformer localisation: skipped (no anomaly candidates)."
        )

    summary_lines.append(
        "Confusion matrix (rows=true, cols=pred):\n" + np.array2string(cm)
    )
    summary_lines.append(
        "Overall – chained prediction classification report:\n" + report
    )

    print("\n".join(summary_lines))  # noqa: T201
    print(f"Confusion matrix saved to {cm_path}")  # noqa: T201

if __name__ == "__main__":
    main()
