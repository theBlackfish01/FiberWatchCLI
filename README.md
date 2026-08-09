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
* **Dataset source:** [BJTU Sensor Team — Phi-OTDR dataset and codes](https://github.com/BJTUSensor/Phi-OTDR_dataset_and_codes)
* Models consume the **raw tensor**, not a PNG. The plotted PNGs are only for inspection/explanation.

### Φ-OTDR acquisition-generalization research status

The complete BJTU inventory contains 15,418 readable windows grouped into 441 recording sessions, six classes, 21 dates, and two acquisition eras. Evaluation uses the recording session—not an individual window—as the independent unit. Training, validation, calibration, support, and query sessions are disjoint where the protocol defines those roles.

| Evaluation | Representation or method | Session macro-F1 / Enrollment-H |
|---|---|---:|
| Random complete-session split | Tested fixed-classifier representations | 0.988–0.996 macro-F1 |
| January → April–May | Absolute, registered-difference, and invariant-fused representations | 0.431–0.522 macro-F1 |
| April–May → January | Absolute, registered-difference, and invariant-fused representations | 0.404–0.699 macro-F1 |
| Five-shot session enrollment, January → April–May | Sliced-Wasserstein matching | 0.480 Enrollment-H |
| Five-shot session enrollment, April–May → January | Sliced-Wasserstein matching | 0.738 Enrollment-H |

The random-session result does not carry over to acquisition-era transfer. The measured shift is class-conditional, and the tested symmetric factorization did not improve both directions. These are retrospective results on one public dataset; they are not evidence of deployment, site, operator, subject, or interrogator generalization.

The research implementation is organized as follows:

* [`config/`](OTDR_CLI/PHI-OTDR/config/) contains versioned split manifests, protocol locks, and CRLF/LF-invariant SHA-256 sidecars.
* [`src/phi_research/`](OTDR_CLI/PHI-OTDR/src/phi_research/) contains dataset contracts, feature extraction, session aggregation, acquisition-era evaluation, open-set evaluation, neural baselines, factorization, and enrollment methods.
* [`tests/`](OTDR_CLI/PHI-OTDR/tests/) checks split disjointness, leakage controls, feature contracts, metrics, protocol hashing, CUDA cache behavior, and artifact validation.
* `experiments/`, raw `.mat` files, model checkpoints, generated reports, and local caches are intentionally ignored.

From `OTDR_CLI/PHI-OTDR`, expose the research package and inspect any frozen runner before supplying local data and output paths:

```bash
export PYTHONPATH=src
python -m phi_research.evaluation_ladder_v1 --help
python -m phi_research.shift_forensics_v1 --help
python -m phi_research.factorization_v1 --help
python -m phi_research.robust_enrollment_v1 --help
pytest -q
```

On PowerShell, use `$env:PYTHONPATH = "src"`. Neural training and frozen neural inference enforce CUDA availability and do not silently fall back to CPU.

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

### OTDR zero-shot fault classification (experimental)

The zero-shot path aligns CUDA-encoded OTDR traces with five versioned semantic
descriptions for each class. It uses only `SNR` and `P1` through `P30`; target-derived
`loss`, `Reflectance`, `Position`, and `Class` fields are never model inputs. Class 0
is the normal anchor, while all 21 unordered pairs from fault classes 1 through 7
serve as unseen-class folds.

The command is intentionally CUDA-only and exits instead of silently falling back
to CPU:

```bash
cd OTDR_CLI/OTDR
python -m src.zero_shot train-fold \
  --data src/data/OTDR_DATA.csv \
  --prototypes src/corpus/zero_shot_fault_prototypes.json \
  --holdout 1 --holdout 2 --device cuda:0
```

Run the complete pairwise benchmark with:

```bash
python -m src.zero_shot benchmark \
  --data src/data/OTDR_DATA.csv \
  --prototypes src/corpus/zero_shot_fault_prototypes.json \
  --device cuda:0
```

Artifacts under `models/zero_shot/fold_XX_YY/` include the checkpoint, scaler,
prototype embeddings, leakage-safe split manifest, CUDA metadata, predictions,
confusion matrices, calibration curve, and both conventional ZSL and generalized
ZSL metrics. The generalized headline metric is the harmonic mean of macro seen
and unseen class accuracy.

### OTDR multi-similarity one-shot classification (experimental)

The multi-similarity one-shot path learns whether two OTDR traces represent the
same class. A shared TCN encoder feeds a symmetric comparison head containing
mean L1 distance, RMS L2 distance, cosine similarity, and the full Hadamard
product. Training uses balanced same/different pairs and binary cross-entropy.
At inference, a reference gallery supports explicit unknown rejection and can be
extended with a confirmed example without retraining the network.

It uses the same leakage-safe `SNR` plus `P1` through `P30` feature contract and
the same 21 pairwise held-out-fault folds as the semantic zero-shot path. CUDA is
mandatory. The primary benchmark reserves 20% of each held-out class as a
support pool, evaluates on the disjoint 80% query set, and reports 20
deterministic one-reference draws by default:

Unknown thresholds are calibrated with five leave-one-seen-fault-out models per
outer fold. Scores are normalized by the known-validation median and IQR before
cross-fold aggregation. Uniform and seen-rich galleries receive independent
calibration. The `balanced` operating point maximizes known/unknown harmonic
mean; `normal_far` uses the final model's normal-validation 1st percentile.

```bash
cd OTDR_CLI/OTDR
python -m src.one_shot train-fold \
  --data src/data/OTDR_DATA.csv \
  --holdout 1 --holdout 2 --device cuda:0

python -m src.one_shot evaluate-detection \
  --fold-dir models/one_shot_crossfit/fold_01_02 \
  --method learned --regime uniform_one_reference \
  --operating-point balanced --device cuda:0

python -m src.one_shot evaluate-one-shot \
  --fold-dir models/one_shot_crossfit/fold_01_02 \
  --method cosine_1nn --regime uniform_one_reference \
  --operating-point normal_far --device cuda:0
```

The same encoded traces are evaluated with the learned multi-similarity head,
cosine 1NN, and Euclidean 1NN. Every fold writes normalized score tables,
known/unknown histograms, and ROC curves. Run all 21 folds with `benchmark`.

`enroll-reference` creates a new gallery from confirmed labeled traces, while
`classify` applies the calibrated rejection threshold. `classify` also accepts
an optional semantic zero-shot predictions CSV via `--semantic-suggestions`;
those labels are surfaced only for rejected traces and remain suggestions until
human confirmation and enrollment.

Learned-head ablations are available by retraining with
`--similarity-mode l1`, `l2`, `cosine`, or `product`; the default is the complete
`multi` feature set.

Artifacts under `models/one_shot_crossfit/fold_XX_YY/` include the CUDA-trained
checkpoint, fitted scaler, uniform and seen-rich galleries, calibration curve,
GPU and dataset hashes, pre-enrollment detection metrics, and post-enrollment
seen/unseen metrics across support draws. The validated 12-epoch full experiment
covered all 21 held-out pairs, both operating points, all three methods, and both
galleries. Selected aggregate results are:

| Gallery / method / operating point | Detection AUROC | Known acceptance | Unknown recall | Post-enrollment H |
|---|---:|---:|---:|---:|
| Uniform / learned / normal-FAR | 0.6108 | 0.9863 | 0.0426 | 0.2136 |
| Uniform / cosine 1NN / normal-FAR | 0.5326 | 0.9583 | 0.0527 | **0.2933** |
| Seen-rich / cosine 1NN / balanced | **0.6566** | 0.5602 | **0.6527** | 0.0719 |

These operating points expose the main trade-off: the normal-FAR setting retains
known traces but detects few unknown faults, while balanced rejection detects more
unknowns at the cost of known acceptance. The semantic zero-shot implementation
has unit and CUDA smoke validation, but no completed all-fold result is claimed
here.

### OTDR fixed-memory TabPFN enrollment

The fixed-memory runner evaluates low-shot enrollment without updating model
weights. For each held-out pair of fault classes, six base classes retain 20
group-distinct examples each. The two enrolled classes append 1, 3, or 5 labeled
examples each. Three fixed context memories are ensembled, and evaluation uses a
balanced 800-example query with 100 examples per class.

This configuration uses the scaled `SNR`, `loss`, and `Reflectance` fields. The
class label and fault position are never model inputs. CUDA is mandatory for
TabPFN execution.

Validated results across all 21 held-out fault pairs and four fresh split seeds
(84 pair/seed units, 20 support draws per unit) are:

| Shots per enrolled class | Harmonic mean H (95% CI) | Enrolled accuracy | Balanced accuracy | Pair/seed means with H >= 0.95 |
|---:|---:|---:|---:|---:|
| 1 | 0.9313 (0.9086-0.9526) | 0.8826 | 0.9706 | 33.3% |
| 3 | 0.9811 (0.9740-0.9876) | 0.9653 | 0.9913 | 94.0% |
| 5 | **0.9887 (0.9849-0.9923)** | **0.9788** | **0.9947** | **100%** |

Base-class accuracy remained 1.0000. The five-shot minimum pair/seed mean was
0.9612. Individual support selection remains relevant: 89.3% of five-shot draws
reached H >= 0.95, and the lowest single draw was 0.7302. Bad-splice recall was
the limiting per-class result at 0.9321.

Run one confirmatory unit or resume the full CUDA matrix with:

```bash
cd OTDR_CLI/OTDR

python -m src.tabpfn_incremental_memory --study confirmatory unit \
  --pair 1-2 --seed 7 --device cuda:0

python -m src.tabpfn_incremental_memory --study confirmatory matrix \
  --device cuda:0
```

The runner stores per-example probabilities, predictions, query/support group
identities, CUDA metadata, and artifact hashes. Generated experiment artifacts
and checkpoints are intentionally excluded from version control.

Key flags:

* `--binary-path / --anomaly-path / --tst-path` – swap in specific checkpoints for each cascade stage.
* `--use-loss-reflectance` – include the diagnostic `loss`/`Reflectance` scalars. Trained checkpoints and scalers gain a `_lr` suffix and evaluation will look for those files automatically when the flag is set.

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
