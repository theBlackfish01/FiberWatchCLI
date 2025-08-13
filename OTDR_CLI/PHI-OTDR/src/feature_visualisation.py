from __future__ import annotations

from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

CLASS_NAMES = ["background", "digging", "knocking", "shaking", "watering", "walking"]


def main():
    parser = argparse.ArgumentParser(description="LDA projection of CNN features")
    here = Path(__file__).resolve().parent
    parser.add_argument("--features", type=Path, default=here / "outputs" / "cnn_features.csv")
    parser.add_argument("--out", type=Path, default=here / "outputs" / "lda_features.png")
    parser.add_argument("--components", type=int, default=2, choices=[2, 3])
    args = parser.parse_args()

    df = pd.read_csv(args.features, header=None)
    arr = df.values.astype(np.float32)
    X = arr[:, :-1]
    y = arr[:, -1].astype(int)

    lda = LinearDiscriminantAnalysis(n_components=args.components)
    Xp = lda.fit_transform(X, y)

    plt.figure(figsize=(7, 6))
    for i, name in enumerate(CLASS_NAMES):
        mask = (y == i)
        plt.scatter(Xp[mask, 0], Xp[mask, 1], s=10, label=name, alpha=0.8)

    plt.legend(markerscale=1.5, fontsize="small", frameon=True, edgecolor="black")
    plt.xlabel("LDA-1")
    plt.ylabel("LDA-2")
    plt.title("CNN Feature LDA Projection")
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    plt.close()
    print(f"Saved LDA plot to {args.out.as_posix()}")


if __name__ == "__main__":
    main()
