import json
import logging
import os
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import wandb
from dotenv import load_dotenv
from scipy.stats import wasserstein_distance

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def _save_heatmap(pair_values: dict[str, float], names: list[str], title: str, save_path: Path) -> None:
    n = len(names)
    name_to_idx = {name: i for i, name in enumerate(names)}
    matrix = np.zeros((n, n))
    for pair_key, value in pair_values.items():
        if "_vs_" not in pair_key:
            continue
        left, right = pair_key.split("_vs_", 1)
        if left in name_to_idx and right in name_to_idx:
            i, j = name_to_idx[left], name_to_idx[right]
            matrix[i, j] = value
            matrix[j, i] = value

    fig_size = max(8, n * 1.2)
    plt.figure(figsize=(fig_size, fig_size * 0.85))
    plt.imshow(matrix, cmap="viridis")
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", fontsize=7)
    plt.title(title)
    plt.xticks(range(n), names, rotation=45, ha="right")
    plt.yticks(range(n), names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    logging.info(f"[HEATMAP] saved: {save_path}")


def normalize_histogram(hist: list[float]) -> np.ndarray:
    arr = np.asarray(hist, dtype=float)
    if arr.size == 0:
        raise ValueError("Histogram cannot be empty")
    if np.any(arr < 0):
        raise ValueError("Histogram contains negative values")
    total = arr.sum()
    if total == 0:
        return np.full(arr.shape, 1.0 / arr.size)
    return arr / total


def compute_wasserstein_pair(hist_a: list, hist_b: list, bins_a: np.ndarray, bins_b: np.ndarray) -> float:
    a = normalize_histogram(hist_a)
    b = normalize_histogram(hist_b)

    if len(a) != len(b):
        raise ValueError("Histograms must have equal length")

    return float(wasserstein_distance(bins_a, bins_b, u_weights=a, v_weights=b))


def _extract_histogram(dataset_data: dict, key: str) -> list[float]:
    values = dataset_data.get(key)
    if values is None:
        raise KeyError(f"Missing histogram '{key}' in dataset data")

    if isinstance(values, list):
        return values

    if isinstance(values, dict):
        vectors = [np.asarray(v, dtype=float) for genre, v in values.items() if genre != "Unknown"]
        if not vectors:
            vectors = [np.asarray(v, dtype=float) for v in values.values()]
        if not vectors:
            raise ValueError(f"Histogram '{key}' has no values")
        lengths = {vec.size for vec in vectors}
        if len(lengths) != 1:
            raise ValueError(f"Histogram '{key}' has inconsistent bin counts")
        return np.sum(vectors, axis=0).tolist()

    raise TypeError(f"Unsupported histogram format for '{key}': {type(values)}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    distribution_path = repo_root / "results" / "distributions" / "distributions.json"

    if not distribution_path.exists() or distribution_path.stat().st_size == 0:
        logging.error("[ERROR] distributions.json missing or empty")
        return

    with distribution_path.open("r", encoding="utf-8") as f:
        datasets = json.load(f)

    histogram_keys = ["pitch_class", "interval", "length_note"]
    aliases = {
        "pitch_class": "pitch_class_wasserstein",
        "interval": "interval_wasserstein",
        "length_note": "length_note_wasserstein",
    }

    dataset_names = sorted(datasets.keys())
    result = {"wasserstein_matrix": {}}

    for left, right in combinations(dataset_names, 2):
        pair_key = f"{left}_vs_{right}"
        pair_values = {}

        for hist_key in histogram_keys:
            left_hist = _extract_histogram(datasets[left], hist_key)
            right_hist = _extract_histogram(datasets[right], hist_key)

            if hist_key == "length_note":
                bins_left = np.array(datasets[left]["length_note_bins"])
                bins_right = np.array(datasets[right]["length_note_bins"])
            else:
                bins_left = np.arange(len(left_hist), dtype=float)
                bins_right = np.arange(len(right_hist), dtype=float)

            pair_values[aliases[hist_key]] = compute_wasserstein_pair(
                left_hist, right_hist, bins_left, bins_right
            )

        pair_values["average_wasserstein"] = float(np.mean(list(pair_values.values())))
        result["wasserstein_matrix"][pair_key] = pair_values

    similarity_dir = repo_root / "results" / "similarity"
    similarity_dir.mkdir(parents=True, exist_ok=True)
    output_path = similarity_dir / "wasserstein_matrix.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    logging.info(f"[WASSERSTEIN] saved matrix: {output_path}")

    wass_data = result["wasserstein_matrix"]
    for metric in ["pitch_class_wasserstein", "interval_wasserstein", "length_note_wasserstein", "average_wasserstein"]:
        pair_values = {pair: vals[metric] for pair, vals in wass_data.items() if metric in vals}
        _save_heatmap(
            pair_values,
            dataset_names,
            f"Wasserstein Heatmap - {metric}",
            similarity_dir / f"heatmap_{metric}.png",
        )

    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logging.warning("WANDB_API_KEY not found; skipping W&B logging")
        return

    wandb.login(key=api_key)
    run = wandb.init(
        project="symbolic-music-similarity",
        entity="wimu-team-6-proj-3",
        job_type="wasserstein-similarity",
        name="wasserstein-similarity",
    )
    flat_payload = {}
    for pair, metrics in result["wasserstein_matrix"].items():
        for metric_name, metric_value in metrics.items():
            flat_payload[f"wasserstein/{pair}/{metric_name}"] = metric_value
    wandb.log(flat_payload)
    run.finish()


if __name__ == "__main__":
    main()
