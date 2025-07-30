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
