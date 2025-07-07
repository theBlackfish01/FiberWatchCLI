# OTDR-ML Pipeline

End-to-end anomaly detection, fault diagnosis, and localisation for Optical-Time-Domain-Reflectometry traces.

---

## 📑 Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Quick-Start](#quick-start)
4. [End-to-End Workflow](#end-to-end-workflow)

   * 4.1 📊 Data Preparation
   * 4.2 🔍 Stage 1 – GRU-AE Detector
   * 4.3 🔧 Stage 2 – TCN Diagnostician
   * 4.4 📜 Alternative – Time-Series Transformer
   * 4.5 📈 Evaluation & Visualisation
   * 4.6 🩺 Model Explainability (SHAP + ChatGPT o3)
5. [Results](#results)
6. [Reproducing the Experiments](#reproducing-the-experiments)
7. [Dependencies](#dependencies)
8. [Acknowledgements](#acknowledgements)

---

## Project Overview

This repository implements and extends the two-stage pipeline from
*“Machine-learning-based anomaly detection in optical fibre monitoring”*.

* **Stage 1 – Detection.**
  A GRU-based Auto-Encoder (GRU-AE) flags anomalous windows by reconstruction error.
* **Stage 2 – Diagnosis & Localisation.**
  *Primary* model: **Temporal Convolutional Network (TCN)**
  *Alternative* model: **Time-Series Transformer (TST)**

Additional modules provide **SHAP interpretability** and a **natural-language explanation** of SHAP output generated with *ChatGPT o3*.

---

## Repository Structure

```text
.
├─ data/                     # not included in the repo, see below
│  ├─ OTDR_DATA.csv
├─ models/                   # also not included in the repo, see below
│  ├─ gru_ae_cond.pt
│  ├─ gru_ae_deep.pt
│  ├─ best_tcn.pt
│  └─ best_tst.pt
├─ ONT_Models.ipynb          # Jupyter notebook for model training & evaluation
├─ requirements.txt
└─ README.md
```

---

## End-to-End Workflow

### 4.1 📊 Data Preparation

* **Input features**: `SNR`, `P1 … P30` (float32).
* **Targets**: `Class` (8 categories) and `Position` (m).
* **Split**: train 64 % / val 20 % / test 16 % (stratified on `Class`).
* **Normalisation**: `StandardScaler` fitted on train only, persisted to `feature_scaler.pkl`.

### 4.2 🔍 Stage 1 – GRU-AE Detector

| Parameter        | Value                       |
| ---------------- | --------------------------- |
| Layers (enc/dec) | 3                           |
| Hidden units     | 64                          |
| Latent dim       | 32                          |
| Optimiser        | Adam 1e-3                   |
| Threshold        | 0.029 MSE (94ᵗʰ percentile) |

### 4.3 🔧 Stage 2 – TCN Diagnostician

* **Architecture**: 6 dilated residual blocks (`k = 3`, dilations 1…32, 64 channels).
* **Loss**: `CE(fault) + 0.5 × MSE(position)`.
* **Training**: Adam 1e-3, gradient clipping, early stopping (patience 7).

### 4.4 📜 Alternative – Time-Series Transformer

* 4-layer encoder-only Transformer, `d_model = 128`, 4 heads, dropout 0.1.
* Trained with AdamW 2e-4, LR step-decay.

### 4.5 📈 Evaluation & Visualisation


* GRU-AE ROC curve & error histogram
* Confusion matrices (TCN / TST)
* Per-class precision-recall bars
* Localisation parity scatter & error histograms


### 4.6 🩺 Model Explainability


1. Computes SHAP values (KernelSHAP) for the TCN on 1 000 random test windows.
2. Persists `tcn_shap_values.pkl`.
3. Sends a prompt + SHAP summary to *ChatGPT o3* via the OpenAI API.
4. Stores the **plain-language narrative** in `chatgpt_o3_narrative.md`.

---

## Results

| Metric            | TCN (direct) | TST (direct) | TST after GRU-AE |
| ----------------- |--------------|--------------|------------------|
| Accuracy          | **0.893**    | 0.882        | 0.961            |
| Macro F1          | 0.892        | 0.881        | 0.917            |
| Localisation RMSE | 0.038 m      | **0.028 m**  | 0.045m           |

> *Key finding:* Attention-based TST halves localisation error relative to the TCN, but the TCN edges ahead on classification accuracy. A hard AE → TST cascade reduces throughput accuracy because detector recall = 90 %.

---


## Dependencies

* Python ≥ 3.10
* PyTorch ≥ 2.0
* scikit-learn, pandas, numpy
* matplotlib & seaborn
* shap
* tqdm
* openai ≥ 1.3 *(only for SHAP narrative generation)*

A full, hash-pinned list is provided in `requirements.txt`.

---

## Acknowledgements

Original research inspiration: **M. Tourancheau et al.**, *“Machine-learning-based anomaly detection in optical fibre monitoring,”* 2021.
SHAP library © Scott Lundberg et al.
Natural-language explanations courtesy of **ChatGPT o3**.
