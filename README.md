# FiberWatch CLI 

End-to-end, CLI-driven pipelines to **train**, **evaluate**, and **explain** machine-learning models for:

* **OTDR** (classic reflectometry; CSV/Parquet dataset)
* **Φ-OTDR / DAS** (distributed acoustic sensing; `.mat` dataset)

The repo includes a lightweight **LLM-based explainability** step during evaluation (optional).

---

## Repository layout

```
OTDR_CLI/
├─ OTDR/
│  └─ src/
│     ├─ corpus/                       # RAG corpus (used by standard OTDR eval)
│     ├─ model_functions/              # TCN, TST, GRU-AE, TabNet, …
│     ├─ data_helper.py                # loaders/scalers for standard OTDR CSV
│     ├─ train.py                      # argparse CLI (OTDR models)
│     ├─ eval.py                       # argparse CLI + LLM explainability
│     └─ rag.py                        # retrieval helpers for LLM (OTDR)
│
├─ PHI-OTDR/
│  └─ src/
│     ├─ config/                       # optional: set OPENAI_API_KEY here
│     ├─ model_functions/
│     │  ├─ __init__.py
│     │  ├─ cnn.py                     # CNN baseline
│     │  └─ tcn.py                     # Temporal ConvNet baseline
│     ├─ data_handler.py               # data/label tools + DataLoaders (Click CLI)
│     ├─ train.py                      # training CLI (Click)
│     ├─ eval.py                       # evaluation CLI (Click) + LLM explainability
│     └─ feature_visualisation.py      # LDA viz for saved features (Click)
│
├─ README.md
└─ requirements.txt
```

---

## Setup

```bash
# Python 3.10–3.11 recommended
python -m venv .venv
source .venv/bin/activate         # (Windows) .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

### (Optional) OpenAI API key for LLM explanations

Set one of:

* Environment variable:

  ```bash
  export OPENAI_API_KEY=sk-...          # (Windows PowerShell) $env:OPENAI_API_KEY="sk-..."
  ```
* Or put `OPENAI_API_KEY = "sk-..."` in `PHI-OTDR/src/config/config.py` or `OTDR/src/config/config.py`.

If no key is present, the LLM step is skipped gracefully.

---

## Data

### Standard OTDR (CSV/Parquet)

Place your cleaned file (default used by scripts):

```
OTDR/ src/ data/ OTDR_data.csv
```

### Φ-OTDR / DAS (`.mat` files)

Expected layout:

```
PHI-OTDR/src/data/das_data/
├─ train/
│  ├─ label.txt           # "relative/path/to/file.mat <int_label>"
│  └─ ... .mat
└─ test/
   ├─ label.txt
   └─ ... .mat
```

Don’t have `label.txt`? Build/validate from folder names:

```bash
# Rebuild labels by scanning a root (labels inferred from parent folders)
python PHI-OTDR/src/data_handler.py labels rebuild \
  --root PHI-OTDR/src/data/das_data/train \
  --out  PHI-OTDR/src/data/das_data/train/label.txt

python PHI-OTDR/src/data_handler.py labels rebuild \
  --root PHI-OTDR/src/data/das_data/test \
  --out  PHI-OTDR/src/data/das_data/test/label.txt

# Validate an existing label file against a root
python PHI-OTDR/src/data_handler.py labels validate \
  --root   PHI-OTDR/src/data/das_data/test \
  --labels PHI-OTDR/src/data/das_data/test/label.txt
```

> Supported class names: `background, digging, knocking, watering, shaking, walking` (IDs `0-5`).

---

## Train

### OTDR (standard; argparse)

```bash
# GRU-AE only
python OTDR/src/train.py --mode gru_ae

# TCN only
python OTDR/src/train.py --mode tcn

# TST only
python OTDR/src/train.py --mode tst

# TabNet only
python OTDR/src/train.py --mode tab

# All sequentially
python OTDR/src/train.py --mode all
```

### Φ-OTDR (Click CLI)

```bash
# CNN
python PHI-OTDR/src/train.py train --model cnn --epochs 30

# TCN (in_channels auto-inferred; override with --in-channels if needed)
python PHI-OTDR/src/train.py train --model tcn --epochs 30

# Common options
python PHI-OTDR/src/train.py train \
  --model cnn \
  --train-root PHI-OTDR/src/data/das_data/train \
  --train-list PHI-OTDR/src/data/das_data/train/label.txt \
  --test-root  PHI-OTDR/src/data/das_data/test  \
  --test-list  PHI-OTDR/src/data/das_data/test/label.txt \
  --out-dir PHI-OTDR/src/models \
  --batch-size 64 --lr 1e-3 --weight-decay 1e-5 --viz-samples 6
```

Artifacts:

* Weights → `.../models/{cnn,tcn}.pt`
* Quick test confusion matrix → `.../outputs/confusion_matrix_train_{cnn,tcn}.png`
* Raw sample heatmaps → `.../outputs/raw_samples/`

---

## Evaluate & Explain

### OTDR (standard; argparse + optional RAG)

```bash
# direct classifier or pipeline mode (see script help)
python OTDR/src/eval.py --mode direct --classifier tcn
```

### Φ-OTDR (Click CLI; CNN or TCN + LLM)

```bash
# Evaluate CNN
python PHI-OTDR/src/eval.py eval --model cnn --weights PHI-OTDR/src/models/cnn.pt

# Evaluate TCN (weights default to models/tcn.pt if not provided)
python PHI-OTDR/src/eval.py eval --model tcn

# Skip LLM step
python PHI-OTDR/src/eval.py eval --model cnn --skip-llm
```

Outputs:

* Confusion matrix → `PHI-OTDR/src/outputs/eval_outputs/confusion_matrix.png`
* Visual samples (raw/pred/LLM-sheet) → `PHI-OTDR/src/outputs/eval_outputs/{samples_raw, samples_pred, samples_llm}/`
* LLM explanation (if enabled) → `PHI-OTDR/src/outputs/llm_output/phi_otdr_{cnn|tcn}_llm_explanation*.txt`

---

## Feature visualization (Φ-OTDR)

If you save features to a CSV (features + last column = label), you can plot a 2-D LDA projection:

```bash
python PHI-OTDR/src/feature_visualisation.py lda \
  --features PHI-OTDR/src/outputs/cnn_features.csv \
  --out PHI-OTDR/src/outputs/lda_features.png \
  --components 2
```

---

## Tips & Troubleshooting

* **Weights loading**: scripts handle both old/new `torch.load` semantics (with/without `weights_only=True`).
* **Windows dataloaders**: `num_workers=0` by default for stability (shared counters).
* **Large traces**: eval down-samples time dimension in LLM sheets to keep images readable.
* **No API key**: LLM step is skipped with a clear message.
* **Missing/broken `.mat`**: silently filtered; final counts printed at the end of train/eval.

---

## Requirements

Install via `pip install -r requirements.txt`. Key packages: `torch`, `scikit-learn`, `numpy`, `scipy`, `matplotlib`, `click`, `openai`.

---

## Acknowledgements

* Φ-OTDR dataset structure adapted from the public “Phi-OTDR\_dataset\_and\_codes”.
* Standard OTDR pipeline includes GRU-AE anomaly detection and TCN/TST/TabNet classifiers, with optional RAG-assisted LLM explanations.
