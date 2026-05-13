import json
import logging
import os
import tempfile
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import muspy
import muspy.outputs.midi as _muspy_midi_out
import numpy as np
import wandb

_muspy_midi_out.PITCH_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
from dotenv import load_dotenv
from frechet_music_distance import FrechetMusicDistance

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

WANDB_PROJECT = "symbolic-music-similarity"
WANDB_ENTITY = "wimu-team-6-proj-3"


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
            plt.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", fontsize=7)
    plt.title(title)
    plt.xticks(range(n), names, rotation=45, ha="right")
    plt.yticks(range(n), names)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()
    logging.info(f"[HEATMAP] saved: {save_path}")


def _remove_invalid_midi(midi_dir: Path) -> int:
    import pretty_midi
    valid = 0
    for midi_file in list(midi_dir.glob("*.mid")):
        try:
            pretty_midi.PrettyMIDI(str(midi_file))
            valid += 1
        except Exception:
            logging.warning(f"[FMD] Removing invalid MIDI: {midi_file.name}")
            midi_file.unlink()
    return valid


class FMDPipeline:
    def __init__(self, processed_dir: Path, output_dir: Path, sample: int | None = None):
        self.processed_dir = processed_dir
        self.output_dir = output_dir
        self.sample = sample
        self._metric = FrechetMusicDistance(
            feature_extractor="clamp2", gaussian_estimator="mle", verbose=True
        )

    def run(self) -> dict:
        dataset_dirs = self._discover_datasets()
        with tempfile.TemporaryDirectory() as tmp:
            midi_dirs = {d.name: self._convert_to_midi(d, Path(tmp)) for d in dataset_dirs}
            matrix = self._compute_matrix(midi_dirs)
        self._save(matrix)
        return matrix

    def _discover_datasets(self) -> list[Path]:
        if not self.processed_dir.exists():
            raise FileNotFoundError(
                f"Processed data not found: {self.processed_dir} — run dataset ingestion first"
            )
        dirs = sorted(d for d in self.processed_dir.iterdir() if d.is_dir())
        if len(dirs) < 2:
            raise ValueError(f"Need at least 2 datasets, found {len(dirs)}")
        logging.info(f"[FMD] Found datasets: {[d.name for d in dirs]}")
        return dirs

    def _convert_to_midi(self, dataset_dir: Path, tmp: Path) -> Path:
        midi_dir = tmp / dataset_dir.name
        midi_dir.mkdir(parents=True, exist_ok=True)

        json_files = list(dataset_dir.glob("*.json"))
        if self.sample is not None:
            json_files = json_files[: self.sample]
        converted = 0

        for json_file in json_files:
            try:
                music = muspy.load_json(json_file)
                muspy.write_midi(str(midi_dir / (json_file.stem + ".mid")), music)
                converted += 1
            except Exception as e:
                logging.warning(f"[FMD] Skipping {json_file.name}: {e}")

        valid = _remove_invalid_midi(midi_dir)
        if valid == 0:
            raise RuntimeError(f"No valid MIDI files for dataset '{dataset_dir.name}'")
        logging.info(f"[FMD] {dataset_dir.name}: {valid} valid files (converted {converted}/{len(json_files)})")
        return midi_dir

    def _compute_matrix(self, midi_dirs: dict[str, Path]) -> dict:
        matrix = {}
        for left, right in combinations(sorted(midi_dirs), 2):
            pair_key = f"{left}_vs_{right}"
            logging.info(f"[FMD] Computing {pair_key}...")
            score = self._metric.score(
                reference_path=str(midi_dirs[left]),
                test_path=str(midi_dirs[right]),
            )
            matrix[pair_key] = {"fmd": float(score)}
            logging.info(f"[FMD] {pair_key}: {score:.4f}")
        return matrix

    def _save(self, matrix: dict) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.output_dir / "fmd_matrix.json"
        with output_path.open("w", encoding="utf-8") as f:
            json.dump({"fmd_matrix": matrix}, f, indent=2, ensure_ascii=False)
        logging.info(f"[FMD] Saved matrix: {output_path}")

        pair_values = {pair: vals["fmd"] for pair, vals in matrix.items()}
        names = sorted({name for pair in matrix for name in pair.split("_vs_", 1)})
        _save_heatmap(pair_values, names, "FMD Heatmap", self.output_dir / "heatmap_fmd.png")

        return output_path


def _log_to_wandb(matrix: dict) -> None:
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        logging.warning("[FMD] WANDB_API_KEY not found; skipping W&B logging")
        return

    wandb.login(key=api_key)
    run = wandb.init(
        project=WANDB_PROJECT,
        entity=WANDB_ENTITY,
        job_type="fmd-similarity",
        name="fmd-similarity",
    )
    wandb.log({
        f"fmd/{pair}/{metric_name}": value
        for pair, metrics in matrix.items()
        for metric_name, value in metrics.items()
    })
    run.finish()


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Limit files per dataset (e.g. 2 for quick test)")
    args = parser.parse_args()

    pipeline = FMDPipeline(
        processed_dir=repo_root / "data" / "processed",
        output_dir=repo_root / "results" / "similarity",
        sample=args.sample,
    )
    matrix = pipeline.run()
    _log_to_wandb(matrix)


if __name__ == "__main__":
    main()
