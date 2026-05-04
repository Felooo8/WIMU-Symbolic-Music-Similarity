import os
import json
from pathlib import Path
from itertools import product

from jsd import calc_jsd
import matplotlib.pyplot as plt
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _distribution_items(dataset_name: str, dataset_data: dict, dist_type: str):
    values = dataset_data.get(dist_type, {})

    if isinstance(values, dict):
        items = [(genre, data) for genre, data in values.items() if genre != "Unknown"]
        if not items:
            items = list(values.items())
    else:
        items = [(dataset_name, values)]

    for genre, data in items:
        yield {
            "dataset": dataset_name,
            "label": f"{dataset_name}:{genre}",
            "type": dist_type,
            "data": data,
        }


def _save_jsd_heatmap(matrix: np.ndarray, names: list[str], dist_type: str, similarity_path: Path):
    if not names:
        return

    plt.figure(figsize=(12, 10))
    plt.imshow(matrix)

    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(
                j,
                i,
                f"{matrix[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
            )

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

    path = os.path.join(distribution_path, "distributions.json")

    if not os.path.exists(path) or os.path.getsize(path) == 0:
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

    jsd_results = {}

    for dist_type in ["pitch_class", "interval"]:
        entries = [
            entry
            for dataset_name, dataset_data in datasets.items()
            for entry in _distribution_items(dataset_name, dataset_data, dist_type)
        ]
        names = sorted({entry["label"] for entry in entries})
        name_to_index = {name: idx for idx, name in enumerate(names)}
        matrix = np.zeros((len(names), len(names)))
        dist_results = {}

        for a, b in product(entries, repeat=2):
            jsd = float(calc_jsd(a["data"], b["data"]))
            key = f'{a["label"]}_vs_{b["label"]}'
            dist_results[key] = jsd
            matrix[name_to_index[a["label"]], name_to_index[b["label"]]] = jsd

        jsd_results[dist_type] = dist_results
        _save_jsd_heatmap(matrix, names, dist_type, similarity_path)

    with (similarity_path / "jsd_matrix.json").open("w", encoding="utf-8") as f:
        json.dump(jsd_results, f, ensure_ascii=False, indent=2)

    logging.info(f"[JSD] saved matrix: {similarity_path / 'jsd_matrix.json'}")


if __name__ == "__main__":
    main()
