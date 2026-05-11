import os
import json
from pathlib import Path
from itertools import combinations

from jsd import calc_jsd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _aggregate_histogram(dataset_data: dict, key: str) -> np.ndarray:
    values = dataset_data.get(key)
    if values is None:
        raise KeyError(f"Missing histogram '{key}' in dataset data")

    if isinstance(values, list):
        return np.asarray(values, dtype=float)

    if isinstance(values, dict):
        vectors = [np.asarray(v, dtype=float) for genre, v in values.items() if genre != "Unknown"]
        if not vectors:
            vectors = [np.asarray(v, dtype=float) for v in values.values()]
        if not vectors:
            raise ValueError(f"Histogram '{key}' has no values")
        lengths = {vec.size for vec in vectors}
        if len(lengths) != 1:
            raise ValueError(f"Histogram '{key}' has inconsistent bin counts: {lengths}")
        return np.sum(vectors, axis=0)

    raise TypeError(f"Unsupported histogram format for '{key}': {type(values)}")


def _save_jsd_heatmap(matrix: np.ndarray, names: list[str], dist_type: str, similarity_path: Path):
    if not names:
        return

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix)

    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)

    plt.title(f"JSD Heatmap - {dist_type}")
    plt.xticks(range(len(names)), names, rotation=90)
    plt.yticks(range(len(names)), names)
    plt.colorbar()
    plt.tight_layout()

    save_file = similarity_path / f"heatmap_{dist_type}.png"
    plt.savefig(save_file, dpi=600, bbox_inches="tight")
    plt.close()

    logging.info(f"[HEATMAP] saved: {save_file}")


def main():
    distribution_path = Path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            "distributions"
        )
    )

    path = distribution_path / "distributions.json"

    if not path.exists() or path.stat().st_size == 0:
        logging.error("[ERROR] distributions.json missing or empty")
        return

    with open(path, "r", encoding="utf-8") as f:
        datasets = json.load(f)

    logging.info("[DISTRIBUTIONS] loaded")

    similarity_path = Path(
        os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "results",
            "similarity"
        )
    )
    similarity_path.mkdir(parents=True, exist_ok=True)

    dataset_names = sorted(datasets.keys())
    jsd_results = {}

    for dist_type in ["pitch_class", "interval"]:
        histograms = {name: _aggregate_histogram(datasets[name], dist_type) for name in dataset_names}

        n = len(dataset_names)
        name_to_index = {name: i for i, name in enumerate(dataset_names)}
        matrix = np.zeros((n, n))
        dist_results = {}

        for left, right in combinations(dataset_names, 2):
            jsd = float(calc_jsd(histograms[left].tolist(), histograms[right].tolist()))
            pair_key = f"{left}_vs_{right}"
            dist_results[pair_key] = jsd
            i, j = name_to_index[left], name_to_index[right]
            matrix[i, j] = jsd
            matrix[j, i] = jsd

        jsd_results[dist_type] = dist_results
        _save_jsd_heatmap(matrix, dataset_names, dist_type, similarity_path)

    with (similarity_path / "jsd_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(jsd_results, f, ensure_ascii=False, indent=2)

    logging.info(f"[JSD] saved matrix: {similarity_path / 'jsd_matrix.json'}")


if __name__ == "__main__":
    main()
