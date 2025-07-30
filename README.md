# OTDR ML Pipeline 🚦📈

*A modular, end-to-end toolkit for anomaly detection, fault classification & localisation on optical-fibre OTDR traces.*

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## ✨ Key features

* **GRU Auto-Encoder** for unsupervised anomaly detection (`models/gruae.py`)
* **Dilated TCN** & **Time-Series Transformer** multitask models  
  – predict *fault class* **and** *fault position* in metres
* **Unified training script**
  ```bash
  python -m train --mode all        # GRU-AE ➜ TCN ➜ TST
  ```

* **Evaluation & visual analytics** (`evaluate.py`)
  Generates confusion matrices, random sample plots, and an **LLM explanation**.

  ```bash
  python -m evaluate --mode pipeline --classifier tst
  ```
* **Retrieval-Augmented Generation (RAG)**
  Ground explanations in ITU standards, or your own corpus via FAISS.
* Ready for a CLI (`otdr detect / classify / pipeline`) – stub in `otdr/cli.py`.

---

## 📂 Repository layout

```
corpus/          ← RAG snippets & FAISS index
    raw/         ← drop PDFs or docs here
    scripts/
        make_chunks.py  build_index.py  ← corpus helpers (run once after adding new docs)
data/            ← OTDR CSVs, preprocessed data
data_helper.py   ← load, split, scale OTDR CSVs
eval.py          ← metrics, plots, RAG explanation
rag.py           ← simple FAISS + OpenAI retrieval helper
train.py         ← end-to-end trainer
model_functions/
    gruae.py     ← GRU auto-encoder
    tcn.py       ← Dilated TCN multitask
    tst.py       ← Time-Series Transformer multitask
config/          ← sets OpenAI API key, must be present at .env

```

---


## 🧠 Using Retrieval-Augmented Generation

1. Drop PDFs or docs into `corpus/raw/`
2. ```bash
   python scripts/make_chunks.py
   python scripts/build_index.py
   ```
3. `eval.py` automatically retrieves relevant snippets and supplies them to GPT-4o.

*No API key?* The script gracefully skips the LLM step.

---


## 📝 License

MIT – see `LICENSE`.

