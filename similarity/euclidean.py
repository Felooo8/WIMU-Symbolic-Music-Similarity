import json
import logging
import os
from itertools import combinations
from pathlib import Path

import wandb
from dotenv import load_dotenv
from scipy.spatial.distance import euclidean

from feature_utils import load_features, compute_global_stats, dataset_mean, save_heatmap, SCALAR_FEATURES

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

WANDB_PROJECT = "symbolic-music-similarity"
WANDB_ENTITY = "wimu-team-6-proj-3"


def _compute_matrix(dataset_means: dict) -> dict:
    matrix = {}
    for left, right in combinations(sorted(dataset_means), 2):
        pair_key = f"{left}_vs_{right}"
        a, b = dataset_means[left], dataset_means[right]
        row: dict = {"euclidean": float(euclidean(a, b))}
        for i, feature in enumerate(SCALAR_FEATURES):
            row[f"euclidean_{feature}"] = float(abs(a[i] - b[i]))
        matrix[pair_key] = row
        logging.info(f"[EUCLIDEAN] {pair_key}: combined={row['euclidean']:.4f}")
    return matrix


def _save(matrix: dict, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "euclidean_matrix.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({"euclidean_matrix": matrix}, f, indent=2, ensure_ascii=False)
    logging.info(f"[EUCLIDEAN] Saved matrix: {output_path}")
    return output_path


def _log_to_wandb(matrix: dict) -> None:
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logging.warning("[EUCLIDEAN] WANDB_API_KEY not found; skipping W&B logging")
        return

    wandb.login(key=api_key)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="euclidean-similarity",
        name="euclidean-similarity",
    )
    wandb.log({
        f"euclidean/{pair}/{metric}": value
        for pair, metrics in matrix.items()
        for metric, value in metrics.items()
    })
    run.finish()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    features_path = repo_root / "results" / "features" / "features.json"

    features = load_features(features_path)
    logging.info(f"[EUCLIDEAN] Total songs: {sum(len(v) for v in features.values())}")

    global_mean, global_std, _ = compute_global_stats(features)
    dataset_means = {
        name: dataset_mean(entries, global_mean, global_std)
        for name, entries in features.items()
    }
    logging.info(f"[EUCLIDEAN] Datasets: {list(dataset_means.keys())}")

    matrix = _compute_matrix(dataset_means)

    similarity_dir = repo_root / "results" / "similarity"
    _save(matrix, similarity_dir)

    names = sorted(dataset_means.keys())
    all_metrics = ["euclidean"] + [f"euclidean_{f}" for f in SCALAR_FEATURES]
    for metric in all_metrics:
        pair_values = {pair: vals[metric] for pair, vals in matrix.items()}
        save_heatmap(
            pair_values,
            names,
            f"{metric} Heatmap",
            similarity_dir / f"heatmap_{metric}.png",
        )

    _log_to_wandb(matrix)


if __name__ == "__main__":
    main()
