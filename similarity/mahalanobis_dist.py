import json
import logging
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import wandb
from dotenv import load_dotenv
from scipy.spatial.distance import mahalanobis

from feature_utils import load_features, compute_global_stats, dataset_mean, save_heatmap

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

WANDB_PROJECT = "symbolic-music-similarity"
WANDB_ENTITY = "wimu-team-6-proj-3"


def _compute_matrix(dataset_means: dict, cov_inv: np.ndarray) -> dict:
    matrix = {}
    for left, right in combinations(sorted(dataset_means), 2):
        pair_key = f"{left}_vs_{right}"
        dist = float(mahalanobis(dataset_means[left], dataset_means[right], cov_inv))
        matrix[pair_key] = {"mahalanobis": dist}
        logging.info(f"[MAHALANOBIS] {pair_key}: {dist:.4f}")
    return matrix


def _save(matrix: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "mahalanobis_matrix.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"mahalanobis_matrix": matrix}, f, indent=2, ensure_ascii=False)
    logging.info(f"[MAHALANOBIS] Saved matrix: {output_path}")
    return output_path


def _log_to_wandb(matrix: dict) -> None:
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logging.warning("[MAHALANOBIS] WANDB_API_KEY not found; skipping W&B logging")
        return

    wandb.login(key=api_key)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="mahalanobis-similarity",
        name="mahalanobis-similarity",
    )
    wandb.log({
        f"mahalanobis/{pair}/{metric}": value
        for pair, metrics in matrix.items()
        for metric, value in metrics.items()
    })
    run.finish()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    features_path = repo_root / "results" / "features" / "features.json"

    features = load_features(features_path)
    logging.info(f"[MAHALANOBIS] Total songs: {sum(len(v) for v in features.values())}")

    global_mean, global_std, cov = compute_global_stats(features)
    cov_inv = np.linalg.pinv(cov)

    dataset_means = {
        name: dataset_mean(entries, global_mean, global_std)
        for name, entries in features.items()
    }
    logging.info(f"[MAHALANOBIS] Datasets: {list(dataset_means.keys())}")

    matrix = _compute_matrix(dataset_means, cov_inv)

    similarity_dir = repo_root / "results" / "similarity"
    _save(matrix, similarity_dir)

    pair_values = {pair: vals["mahalanobis"] for pair, vals in matrix.items()}
    save_heatmap(
        pair_values,
        sorted(dataset_means.keys()),
        "Mahalanobis Distance Heatmap",
        similarity_dir / "heatmap_mahalanobis.png",
    )

    _log_to_wandb(matrix)


if __name__ == "__main__":
    main()
