import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

SCALAR_FEATURES = [
    "pitch_class_entropy",
    "pitch_entropy",
    "pitch_range",
    "scale_consistency",
    "polyphony",
    "empty_beat_rate",
    "groove_consistency",
]


def load_features(features_path: Path) -> dict[str, list[dict]]:
    if not features_path.exists() or features_path.stat().st_size == 0:
        raise FileNotFoundError(f"features.json missing or empty: {features_path}")
    with features_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_vectors(dataset_features: list[dict]) -> np.ndarray:
    vectors = []
    for entry in dataset_features:
        row = [entry.get(f) for f in SCALAR_FEATURES]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in row):
            continue
        vectors.append(row)
    if not vectors:
        raise ValueError("No valid feature vectors found")
    return np.array(vectors, dtype=float)


def compute_global_stats(
    features: dict[str, list[dict]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (global_mean, global_std, covariance_of_normalized) from all individual songs."""
    all_vectors = np.vstack([extract_vectors(entries) for entries in features.values()])
    mean = all_vectors.mean(axis=0)
    std = all_vectors.std(axis=0)
    std[std == 0] = 1.0
    normalized = (all_vectors - mean) / std
    cov = np.cov(normalized.T)
    return mean, std, cov


def dataset_mean(
    dataset_features: list[dict],
    global_mean: np.ndarray,
    global_std: np.ndarray,
) -> np.ndarray:
    """Z-normalize each song individually using global stats, then average."""
    vectors = extract_vectors(dataset_features)
    normalized = (vectors - global_mean) / global_std
    return normalized.mean(axis=0)


def save_heatmap(
    pair_values: dict[str, float],
    names: list[str],
    title: str,
    save_path: Path,
    fmt: str = ".3f",
) -> None:
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
            plt.text(j, i, format(matrix[i, j], fmt), ha="center", va="center", fontsize=7)
    plt.title(title)
    plt.xticks(range(n), names, rotation=45, ha="right")
    plt.yticks(range(n), names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    logging.info(f"[HEATMAP] saved: {save_path}")
