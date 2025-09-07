import pathlib

from scipy.io import loadmat
import numpy as np, matplotlib.pyplot as plt

ROOT_DIR = pathlib.Path(__file__).parent
DATA_DIR = ROOT_DIR / "data/das_data/test/02_dig/220104_sys_dig_01_single_data_5.mat"
arr = loadmat(DATA_DIR)["data"].astype(np.float32)  # (T,C)

# per-sample minmax (same as data_handler)
arr_n = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)

print("shape:", arr_n.shape, "min/max:", float(arr_n.min()), float(arr_n.max()))
print("first 5 time steps of channels 0..2:\n", np.round(arr_n[:5, :3], 3))

# Heatmap like the report
plt.imshow(arr_n.T, aspect="auto", origin="lower")
plt.xlabel("Time index"); plt.ylabel("Channel"); plt.colorbar(label="norm. amplitude")
plt.show()

# Waveforms for a few channels
plt.plot(arr_n[:2000, 0], label="ch0")
plt.plot(arr_n[:2000, 1], label="ch1")
plt.plot(arr_n[:2000, 2], label="ch2")
plt.plot(arr_n[:2000, 3], label="ch3")
plt.legend(); plt.xlabel("Time"); plt.ylabel("norm. amplitude")
plt.show()
