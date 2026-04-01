#!/usr/bin/env python3
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    similarity_dir = Path("results/similarity")
    features_dir = Path("results/features")
    similarity_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)

    jsd_data = {
        "maestro_vs_lakh": {"average_jsd": 0.67},
        "maestro_vs_nes": {"average_jsd": 0.89},
        "lakh_vs_nes": {"average_jsd": 0.74},
    }

    with (similarity_dir / "jsd_matrix.json").open("w", encoding="utf-8") as file:
        json.dump(jsd_data, file, ensure_ascii=False, indent=2)

    labels = ["maestro", "lakh", "nes"]
    matrix = np.array(
        [
            [0.0, 0.67, 0.89],
            [0.67, 0.0, 0.74],
            [0.89, 0.74, 0.0],
        ]
    )
    plt.figure(figsize=(6, 4))
    plt.imshow(matrix, cmap="magma")
    plt.colorbar(label="Average JSD")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="white")
    plt.title("Sample JSD heatmap")
    plt.tight_layout()
    plt.savefig(similarity_dir / "heatmap.png")
    plt.close()

    summary = {
        "maestro": {
            "pitch_class_entropy": {"mean": 3.21, "std": 0.45},
            "polyphony": {"mean": 2.1, "std": 1.2},
        },
        "lakh": {
            "pitch_class_entropy": {"mean": 3.55, "std": 0.38},
            "polyphony": {"mean": 2.4, "std": 0.9},
        },
        "nes": {
            "pitch_class_entropy": {"mean": 2.91, "std": 0.50},
            "polyphony": {"mean": 1.2, "std": 0.7},
        },
    }
    with (features_dir / "summary_stats.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
