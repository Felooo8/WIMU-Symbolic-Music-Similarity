import os
import json
from pathlib import Path
from itertools import product
from jsd import calc_jsd
import matplotlib.pyplot as plt
import numpy as np
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


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

    all_entries = []

    for dataset_name, dataset_data in datasets.items():

        for dist_type in ["pitch_class", "interval", "length_note"]:
            genres = dataset_data.get(dist_type, {})

            for genre, values in genres.items():
                if genre == "Unknown":
                    continue

                all_entries.append({
                        "dataset": dataset_name,
                        "label": f"{dataset_name}:{genre}",
                        "type": dist_type,
                        "data": values
                    })


    jsd_results = {
        "pitch_class": {},
        "interval": {}
    }

    for a, b in product(all_entries, repeat=2):

        if a["type"] != b["type"] or (a["type"] == "length_note" or b["type"] == "length_note"):
            continue

        jsd = calc_jsd(a["data"], b["data"])

        jsd_results[a["type"]][(a["label"], b["label"])] = jsd

    for dist_type in ["pitch_class", "interval"]:

        names = set()
        for (n1, n2) in jsd_results[dist_type].keys():
            names.add(n1)
            names.add(n2)

        names = sorted(list(names))
        n = len(names)

        matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                matrix[i, j] = jsd_results[dist_type].get(
                    (names[i], names[j]),
                    0
                )

        plt.figure(figsize=(12, 10))
        plt.imshow(matrix)

        for i in range(n):
            for j in range(n):
                plt.text(
                    j, i,
                    f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8
                )

        plt.title(f"JSD Heatmap - {dist_type}")
        plt.xticks(range(n), names, rotation=90)
        plt.yticks(range(n), names)

        plt.colorbar()
        plt.tight_layout()

        similarity_path = Path(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "similarity"
            )
        )

        if not os.path.exists(similarity_path):
            similarity_path.mkdir(parents=True, exist_ok=True)
            logging.info(f"[HEATMAP] created similarity_path={similarity_path}")

        save_file = os.path.join(similarity_path, f"heatmap_{dist_type}.png")
        plt.savefig(save_file, dpi=600, bbox_inches="tight")
        plt.close()

        logging.info(f"[HEATMAP] saved: {save_file}")


if __name__ == "__main__":
    main()