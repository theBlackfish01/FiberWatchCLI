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

**Pipeline (binary anomaly filter → anomaly-only TCN → localisation TST)**

* **Loss/reflectance augmented inputs (`--use-loss-reflectance`)**
  * Binary filter: **accuracy 1.000**, **AUC 1.000**; flagged **5,490 / 6,292** traces as faulty (ground-truth: 5,490).
  * Anomaly-only TCN: **accuracy 1.000** on the mapped faults (5,490 / 5,490 predictions).
  * Localisation TST: **RMSE 0.015 m** across 5,490 traces.
  * Overall chained classifier: **accuracy 1.000**, macro **precision / recall / F1 = 1.000 / 1.000 / 1.000** (supports per class: 802, 800, 800, 800, 800, 693, 800, 797).

* **Standard inputs**
  * Binary filter: **accuracy 0.953**, **AUC 0.993**; flagged **5,230 / 6,292** traces as faulty (ground-truth: 5,490).
  * Anomaly-only TCN: **accuracy 0.958** on 5,213 mapped faults (predictions issued for 5,230 traces).
  * Localisation TST: **RMSE 0.019 m** across 5,230 traces.
  * Overall chained classifier: **accuracy 0.919**, macro **precision / recall / F1 = 0.929 / 0.919 / 0.920**.

Historical standalone model baselines (pre-pipeline): **TCN** accuracy ≈ 0.885; **TST** accuracy ≈ 0.881.

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

## Preparing the OTDR RAG corpus

1. **Chunk the curated OTDR references** into `docs.json`:

   ```bash
   python OTDR_CLI/OTDR/src/corpus/scripts/make_chunks.py \\
       --raw-dir OTDR_CLI/OTDR/src/corpus/raw \\
       --output OTDR_CLI/OTDR/src/corpus/docs.json
   ```

   This writes token-friendly snippets (≈200 words each) inside the OTDR module tree so they can be versioned alongside the codebase.

2. **Sync the chunks to Pinecone** (uses the same `text-embedding-3-large` model as runtime RAG):

   ```bash
   python OTDR_CLI/OTDR/src/corpus/scripts/sync_pinecone.py \\
       --docs-path OTDR_CLI/OTDR/src/corpus/docs.json \\
       --namespace otdr-prod
   ```

   The helper will create the `fiberwatch` index if it is missing, embed in batches with OpenAI, and upsert chunk metadata (`text`, `source`, and `chunk_index`). Use `--raw-dir` if you prefer to regenerate chunks on the fly or `--batch-size` / `--limit-words` to tune ingestion.

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

**OTDR inference pipeline:**

```bash
cd OTDR_CLI/OTDR
python -m src.pipeline --data data/OTDR_DATA.csv
```

**OTDR training (standard features):**

```bash
cd OTDR_CLI/OTDR
python -m src.train --mode all --data data/OTDR_DATA.csv
```

**OTDR training with loss/Reflectance inputs:**

```bash
cd OTDR_CLI/OTDR
python -m src.train --mode all --use-loss-reflectance --data data/OTDR_DATA.csv
```

Both commands persist a `feature_config` block in the emitted metadata. When the loss/Reflectance flag is enabled the scaler and checkpoints receive a `_lr` suffix; evaluation automatically targets those filenames when `--use-loss-reflectance` is provided and will refuse to run if the requested feature signature diverges from the checkpoint metadata.

Key flags:

* `--binary-path / --anomaly-path / --tst-path` – swap in specific checkpoints for each cascade stage.
* `--use-loss-reflectance` – include the leakage-prone `loss`/`Reflectance` scalars. Trained checkpoints and scalers gain a `_lr` suffix and evaluation will look for those files automatically when the flag is set.

The legacy individual commands (`python -m src.train` / `python -m src.eval`) still work if you prefer manual control.

> If the Φ-OTDR `.mat` roots don’t have `label.txt`, use `data_handler.py` to rebuild/validate labels from folder names.

---

## Design notes & roadmap

* **TFT memory scaling**: attention is O(T²); we **downsample time** inside the model to a safe cap and use **mixed precision** on GPU. This preserves long-context behavior without OOM on 8 GB cards.
* **Post-processing**: a simple temporal smoother or HMM-style decoder over class logits could reduce the walking/digging and knocking/watering swaps.
* **Cross-domain ideas**: bring the **GRU-AE → classifier pipeline** to Φ-OTDR to triage “active vs quiet” windows before classification.
* **Explainability**: add saliency/rollout maps per channel and render them onto the LLM sheet to show “where the model looked.”

---

*FiberWatch CLI unifies practical training, rigorous evaluation, and human-readable explanations for fiber monitoring—covering both reflectometry and distributed acoustic sensing, with results that are good out-of-the-box and a clear path to further improvement.*
