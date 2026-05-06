import json
import logging
import os
from itertools import combinations
from pathlib import Path

import numpy as np
import wandb
from dotenv import load_dotenv
from scipy.spatial.distance import euclidean

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

WANDB_PROJECT = "symbolic-music-similarity"
WANDB_ENTITY = "wimu-team-6-proj-3"

SCALAR_FEATURES = [
    "pitch_class_entropy",
    "pitch_entropy",
    "pitch_range",
    "scale_consistency",
    "polyphony",
    "empty_beat_rate",
    "groove_consistency",
]


def _load_features(features_path: Path) -> dict[str, list[dict]]:
    if not features_path.exists() or features_path.stat().st_size == 0:
        raise FileNotFoundError(f"features.json missing or empty: {features_path}")
    with features_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _mean_vector(dataset_features: list[dict]) -> np.ndarray:
    vectors = []
    for entry in dataset_features:
        row = [entry.get(f) for f in SCALAR_FEATURES]
        if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in row):
            continue
        vectors.append(row)
    if not vectors:
        raise ValueError("No valid feature vectors found")
    return np.mean(vectors, axis=0)


def _normalize(dataset_vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    matrix = np.stack(list(dataset_vectors.values()))
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std[std == 0] = 1.0
    return {
        name: (vec - mean) / std
        for name, vec in dataset_vectors.items()
    }


def _compute_matrix(normalized: dict[str, np.ndarray]) -> dict:
    matrix = {}
    for left, right in combinations(sorted(normalized), 2):
        pair_key = f"{left}_vs_{right}"
        dist = float(euclidean(normalized[left], normalized[right]))
        matrix[pair_key] = {"euclidean": dist}
        logging.info(f"[EUCLIDEAN] {pair_key}: {dist:.4f}")
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

    features = _load_features(features_path)

    dataset_vectors = {
        name: _mean_vector(entries)
        for name, entries in features.items()
    }
    logging.info(f"[EUCLIDEAN] Datasets: {list(dataset_vectors.keys())}")

    normalized = _normalize(dataset_vectors)
    matrix = _compute_matrix(normalized)

    _save(matrix, repo_root / "results" / "similarity")
    _log_to_wandb(matrix)


if __name__ == "__main__":
    main()
