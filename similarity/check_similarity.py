import itertools
import json
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from jsd import calc_jsd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DISTRIBUTION_PATH = ROOT_DIR / "results" / "distributions" / "distributions.json"
SIMILARITY_DIR = ROOT_DIR / "results" / "similarity"
FEATURES_PATH = ROOT_DIR / "results" / "features" / "features.json"
SUMMARY_PATH = ROOT_DIR / "results" / "features" / "summary_stats.json"


def _load_json(path: Path) -> dict:
    if not path.exists() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required file does not exist or is empty: {path}")

    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _save_jsd_heatmap(matrix: np.ndarray, labels: list[str], savepath: Path) -> None:
    plt.figure(figsize=(7, 5))
    plt.imshow(matrix, cmap="viridis")
    plt.colorbar(label="Average JSD")
    plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
    plt.yticks(range(len(labels)), labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white")
    plt.title("Jensen-Shannon Divergence (JSD) matrix")
    plt.tight_layout()
    plt.savefig(savepath)
    plt.close()


def _calculate_summary_stats(features_data: dict) -> dict:
    summary: dict[str, dict] = {}

    for dataset_name, rows in features_data.items():
        summary[dataset_name] = {}
        if not rows:
            continue
        keys = rows[0].keys()
        for key in keys:
            values = np.array([row[key] for row in rows], dtype=float)
            summary[dataset_name][key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
            }

    return summary


def main() -> None:
    datasets = _load_json(DISTRIBUTION_PATH)
    names = list(datasets.keys())

    SIMILARITY_DIR.mkdir(parents=True, exist_ok=True)

    pairwise_jsd: dict[str, dict[str, float]] = {}
    matrix = np.zeros((len(names), len(names)))

    for i, j in itertools.combinations(range(len(names)), 2):
        first_name, second_name = names[i], names[j]
        first_distributions = datasets[first_name]
        second_distributions = datasets[second_name]

        pitch_jsd = calc_jsd(
            first_name,
            second_name,
            first_distributions["pitch_class"],
            second_distributions["pitch_class"],
        )
        interval_jsd = calc_jsd(
            first_name,
            second_name,
            first_distributions["interval"],
            second_distributions["interval"],
        )

        average_jsd = float(np.mean([pitch_jsd, interval_jsd]))
        key = f"{first_name}_vs_{second_name}"
        pairwise_jsd[key] = {
            "pitch_class_jsd": float(pitch_jsd),
            "interval_jsd": float(interval_jsd),
            "average_jsd": average_jsd,
        }
        matrix[i, j] = average_jsd
        matrix[j, i] = average_jsd

    with (SIMILARITY_DIR / "jsd_matrix.json").open("w", encoding="utf-8") as file:
        json.dump(pairwise_jsd, file, ensure_ascii=False, indent=2)

    _save_jsd_heatmap(matrix, names, SIMILARITY_DIR / "heatmap.png")

    if FEATURES_PATH.exists() and FEATURES_PATH.stat().st_size > 0:
        summary = _calculate_summary_stats(_load_json(FEATURES_PATH))
        with SUMMARY_PATH.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    logging.info("Similarity artifacts saved under %s", SIMILARITY_DIR)


if __name__ == "__main__":
    main()
