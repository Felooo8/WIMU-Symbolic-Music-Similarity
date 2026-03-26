import yaml
import muspy
import wandb
import os
from dotenv import load_dotenv


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


mapping = {
    "maestro_v3": muspy.MAESTRODatasetV3,
    "lakh_midi": muspy.LakhMIDIAlignedDataset,
    "jsb_chorales": muspy.JSBChoralesDataset,
    "nes_mdb": muspy.NESMusicDatabase
}


def get_muspy_dataset_class(dataset_name):
    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available options are: {list(mapping.keys())}")

    return mapping[dataset_name]


def load_and_parse_dataset(dataset_cfg):
    path = dataset_cfg["path"]
    name = dataset_cfg["name"]
    sample_size = dataset_cfg.get("sample_size", float('inf'))

    print(f"\n[{name}] Starting processing (Limit: {sample_size} files)...")

    DatasetClass = get_muspy_dataset_class(name)
    if name == "lakh_midi":
        dataset = DatasetClass(path, download_and_extract=True)
    else:
        dataset = DatasetClass(path, download_and_extract=False)

    valid_scores = []
    errors = 0

    for i, score in enumerate(dataset):
        if i >= sample_size:
            print(f"[{name}] Sample limit reached ({sample_size}). Stopping ingestion.")
            break

        if score is not None and len(score.tracks) > 0:
            valid_scores.append(score)
        else:
            errors += 1

    print(f"[{name}] Finished. Valid: {len(valid_scores)}, Errors: {errors}")
    return valid_scores, errors


def main():
    cfg = load_config()
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")

    if api_key:
        wandb.login(key=api_key)
    else:
        raise ValueError("Missing WANDB_API_KEY or .env file!")

    run = wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        config=cfg,
        name="full-dataset-ingestion"
    )

    all_datasets_data = {}

    dataset_artifact = wandb.Artifact(
        name="sampled-symbolic-datasets",
        type="dataset",
        description="Prefiltered MusPy Json files."
    )

    processed_dir = "data/processed"
    os.makedirs(processed_dir, exist_ok=True)

    for dataset_key, dataset_cfg in cfg["datasets"].items():
        parsed_scores, error_count = load_and_parse_dataset(dataset_cfg)
        all_datasets_data[dataset_key] = parsed_scores

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

    print("\nSending W&B artifact")
    run.log_artifact(dataset_artifact)

    wandb.finish()
    print("Run W&B done. Check the artifact on the W&B!")


if __name__ == "__main__":
    main()
