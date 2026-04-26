import yaml
import muspy
import wandb
import os
import json
import logging
from dotenv import load_dotenv

from data_providers import DatasetFactory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_genre_map(genre_file_path: str) -> dict:
    genre_map = {}
    with open(genre_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                genre_map[parts[0]] = parts[1]
    return genre_map


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def inject_genre_into_json(
    json_path: str,
    genre: str | None,
    track_id: str | None,
    dataset: str
):
    with open(json_path, "r") as f:
        data = json.load(f)

    if "metadata" not in data or data["metadata"] is None:
        data["metadata"] = {}

    data["metadata"]["genre"] = genre
    data["metadata"]["msd_track_id"] = track_id
    data["metadata"]["dataset"] = dataset


    with open(json_path, "w") as f:
        json.dump(data, f)


def main():
    cfg = load_config()
    load_dotenv()

    api_key = os.getenv("WANDB_API_KEY")
    if not api_key:
        raise ValueError("Missing WANDB_API_KEY or .env file!")

    wandb.login(key=api_key)

    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        config=cfg,
        name="full-dataset-ingestion"
    )

    genre_map = {}
    genre_file = cfg["datasets"]["lakh_midi"].get("genre_file")
    if genre_file and os.path.exists(genre_file):
        genre_map = load_genre_map(genre_file)
        logging.info(f"Loaded {len(genre_map)} genre mappings.")

    dataset_artifact = wandb.Artifact(
        name="sampled-symbolic-datasets",
        type="dataset"
    )

    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    for dataset_key, dataset_cfg in cfg["datasets"].items():
        provider = DatasetFactory.create(dataset_cfg)

        parsed_scores, error_count = provider.process()

        wandb.log({
            f"{dataset_key}/successfully_parsed": len(parsed_scores),
            f"{dataset_key}/corrupted_skipped": error_count,
        })

        dataset_out_dir = os.path.join(processed_dir, dataset_key)
        os.makedirs(dataset_out_dir, exist_ok=True)

        for i, (score, track_id) in enumerate(parsed_scores):
            out_path = os.path.join(dataset_out_dir, f"score_{i:04d}.json")
            muspy.save(out_path, score)

            if dataset_key.lower() == "lakh_midi":
                genre = genre_map.get(track_id)
                if genre is None or genre.strip() == "":
                    genre = "Unknown"
            elif dataset_key.lower() == "maestro_v3":
                genre = "Classical music (piano)"
            elif dataset_key.lower() == "nes_mdb":
                genre = "Chiptune (8-bit music)"
            else:
                genre = "Unknown"

            inject_genre_into_json(
                out_path,
                genre,
                track_id,
                dataset_key
            )

            dataset_artifact.add_file(out_path, name=f"{dataset_key}/score_{i:04d}.json")

    wandb.log_artifact(dataset_artifact)
    wandb.finish()
    logging.info("Done.")


if __name__ == "__main__":
    main()