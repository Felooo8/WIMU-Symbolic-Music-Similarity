import yaml
import muspy
import wandb
import os
import logging
from dotenv import load_dotenv

from data_providers import DatasetFactory

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


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

    dataset_artifact = wandb.Artifact(
        name="sampled-symbolic-datasets",
        type="dataset",
        description="Prefiltered MusPy Json files."
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

        for i, score in enumerate(parsed_scores):
            out_path = os.path.join(dataset_out_dir, f"score_{i:04d}.json")
            muspy.save(out_path, score)
            dataset_artifact.add_file(out_path, name=f"{dataset_key}/score_{i:04d}.json")

    logging.info("Sending W&B artifact...")
    wandb.log_artifact(dataset_artifact)
    wandb.finish()
    logging.info("Run W&B done. Check the artifact on the W&B!")


if __name__ == "__main__":
    main()