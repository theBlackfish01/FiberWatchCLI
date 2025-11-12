# FiberWatch CLI

FiberWatch is a CLI toolkit to **train**, **evaluate**, and **explain** ML models for two sensing modes:

* **OTDR** (classic reflectometry) — 1-D fiber traces, anomaly detection + fault type/position.
* **Φ-OTDR / DAS** (distributed acoustic sensing) — time-channel mats, human activity / event classification.

Both tracks include **visual diagnostics** and an optional **LLM explainability** step (with or without RAG).

---

## What’s in the repo (conceptually)

* **OTDR/src/**

  * Models: GRU-AE (detector), TCN / TST (classifiers)
  * `train.py`, `eval.py` (argparse CLIs), `data_helper.py`
  * Optional RAG (`corpus/`, `rag.py`) to ground LLM explanations

* **PHI-OTDR/src/**

  * Models: **CNN**, **TCN**, **TFT (Temporal Fusion Transformer)**
  * `data_handler.py` for `.mat` data + label tooling
  * `train.py` / `eval.py` (Click CLIs) with visualizations and LLM explanations
  * `feature_visualisation.py` (LDA scatter of saved features)

> The code treats OTDR and Φ-OTDR differently because the **data geometry** is different.

---

## Data modalities (what the models actually “see”)

### OTDR

* Input is a **1-D amplitude trace** sampled along distance (P-points) plus a few scalars (e.g., SNR).
* Tasks:

  1. **Detection** (GRU-AE reconstruction error) 
  2. **Diagnosis + localization** (TCN/TST) — fault **type** and **position**.

### Φ-OTDR / DAS

* Each sample is a **(T × C)** matrix — **time** across the x-axis, **channels (distance bins)** on the y-axis, values represent normalized backscatter intensity/phase features.
* We render this as a **heatmap** so humans (and the LLM) see temporal envelopes across channels.
* Models consume the **raw tensor**, not a PNG. The plotted PNGs are only for inspection/explanation.

---

## Modeling approaches (why these models)

* **GRU-AE (OTDR)**: learns the manifold of *healthy* traces → high reconstruction error flags anomalies.
* **TCN**: causal, dilated temporal convolutions; strong for sequences with local motifs.
* **TST (Transformer for time series, OTDR)**: attention across the 1-D distance axis to capture long-range patterns.
* **CNN (Φ-OTDR)**: baseline that scans local time–channel patches.
* **TFT (Φ-OTDR)**: adds **variable selection**, **positional encodings**, and **attention** over time,
  then **attention pooling** for sequence classification. We cap attention context with light
  **time downsampling** inside the model to keep memory manageable on long traces.

---

## How to read the visual outputs

* **Raw heatmap** (Φ-OTDR): time vs channel with color = normalized amplitude; quick sanity check.
* **Prediction heatmap**: title annotates *True* vs *Pred* class.
* **LLM sheet** (Φ-OTDR): heatmap + per-class probability bars + a small stats box (means/std, top-energy channels).
  This is what we pass to the LLM so it can ground its explanation in the same picture you see.

---

## Results (high level)

### OTDR 

* **TCN**: Acc ≈ 0.885; **TST**: Acc ≈ 0.881
* **GRU-AE → TST pipeline**: Acc ≈ **0.958**, with strong anomaly recall; localization MSE \~ **5 cm** scale.

### Φ-OTDR — **TFT**

**Overall metrics (N=3052):**

* **Accuracy** 0.904
* **Balanced Acc** 0.903
* **MCC** 0.885, **Cohen’s κ** 0.885
* **Macro P/R/F1** 0.902 / 0.903 / 0.902
* **LogLoss** 0.288, **ROC-AUC (macro OVR)** 0.991
* **Top-3** 0.996, **Top-5** 1.000

**Per-class (precision / recall / F1, support):**

* background: **0.967 / 0.939 / 0.953** (588)
* digging: **0.923 / 0.863 / 0.892** (502)
* knocking: **0.876 / 0.903 / 0.889** (475)
* watering: **0.854 / 0.909 / 0.881** (451)
* shaking: **0.923 / 0.949 / 0.936** (546)
* walking: **0.869 / 0.853 / 0.861** (490)

**What the confusion tells us (row-normalized):**

* **Walking ↔ Digging**: \~**4.7–7.8%** leakage. Both can produce **sustained energy across nearby channels**; walking sometimes looks “bursty” like shallow digging.
* **Knocking ↔ Watering**: **7.4%** of *knocking* predicted as *watering*. These both exhibit **short, repeated envelopes**, differing mostly in **duration** and **channel spread**; borderline segments blur that line.
* **Shaking** is the **cleanest** (recall \~**94.9%**), likely because its **wideband, persistent pattern** stands out from the others.
* **Background** remains mostly intact (recall \~**93.9%**), with small spill to *knocking* (3.1%) — short transients in quiet segments can be interpreted as taps.

**Takeaways**

* TFT is **strong and well-calibrated** (low LogLoss, high ROC-AUC).
* The primary remaining errors align with **classes that share temporal envelopes**; additional features (e.g., per-channel spectral ratios, event duration priors) or **sequence-level post-filters** could squeeze extra points.

---

## LLM explainability (and why RAG helps)

* During eval we save a handful of **LLM sheets** (the composite plot) and ask a vision model for a **natural-language explanation**: what pattern indicates the predicted class, why confusions occurred, and quick **operational tips**.
* With **RAG** (for OTDR; optional for Φ-OTDR), the model cites curated materials (e.g., ITU-T style guidance) and tends to:

  * avoid hallucinated claims/citations,
  * be more **procedural** (e.g., “try bidirectional OTDR, clean/check connectors, isolate segments”),
  * and structure the write-up into clear, **actionable** bullet points.

You’ll find the generated text under `outputs/llm_output/…` alongside the saved plots.

---

## Minimal “how to run” (for context)

> Full CLIs already exist; this is just a tiny cheat-sheet.

**Φ-OTDR training (TFT):**

```bash
python PHI-OTDR/src/train.py train --model tft
```

**Φ-OTDR eval + plots (+ optional LLM):**

```bash
python PHI-OTDR/src/eval.py eval --model tcn --skip-llm   # or --model tft | cnn
python PHI-OTDR/src/eval.py eval --model tft              # runs LLM if key is set
```

**OTDR pipeline examples:**

```bash
python OTDR/src/train.py --mode all
python OTDR/src/eval.py  --mode pipeline --classifier tcn
```

> If the Φ-OTDR `.mat` roots don’t have `label.txt`, use `data_handler.py` to rebuild/validate labels from folder names.

---

## Design notes & roadmap

* **TFT memory scaling**: attention is O(T²); we **downsample time** inside the model to a safe cap and use **mixed precision** on GPU. This preserves long-context behavior without OOM on 8 GB cards.
* **Post-processing**: a simple temporal smoother or HMM-style decoder over class logits could reduce the walking/digging and knocking/watering swaps.
* **Cross-domain ideas**: bring the **GRU-AE → classifier pipeline** to Φ-OTDR to triage “active vs quiet” windows before classification.
* **Explainability**: add saliency/rollout maps per channel and render them onto the LLM sheet to show “where the model looked.”

---

*FiberWatch CLI unifies practical training, rigorous evaluation, and human-readable explanations for fiber monitoring—covering both reflectometry and distributed acoustic sensing, with results that are good out-of-the-box and a clear path to further improvement.*
