import yaml
import muspy
import wandb


def load_config(config_path="configs/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_muspy_dataset_class(dataset_name):
    mapping = {
        "maestro_v3": muspy.MAESTRODatasetV3,
        "lakh_midi": muspy.LakhMIDIMatchedDataset,
        "jsb_chorales": muspy.JSBChoralesDataset,
        "nes_mdb": muspy.NESMusicDatabase
    }

    if dataset_name not in mapping:
        raise ValueError(f"Unsupported dataset: {dataset_name}. Available options are: {list(mapping.keys())}")

    return mapping[dataset_name]


def load_and_parse_dataset(dataset_cfg):
    path = dataset_cfg["path"]
    name = dataset_cfg["name"]
    sample_size = dataset_cfg.get("sample_size", float('inf'))

    print(f"\n[{name}] Starting processing (Limit: {sample_size} files)...")

    DatasetClass = get_muspy_dataset_class(name)
    dataset = DatasetClass(path, download_and_extract=True)

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

    # wandb.init(
    #     project=cfg["wandb"]["project"],
    #     entity=cfg["wandb"]["entity"],
    #     config=cfg,
    #     name="full-dataset-ingestion"
    # )

    all_datasets_data = {}

    for dataset_key, dataset_cfg in cfg["datasets"].items():
        parsed_scores, error_count = load_and_parse_dataset(dataset_cfg)

        all_datasets_data[dataset_key] = parsed_scores

        # wandb.log({
        #     f"{dataset_key}/successfully_parsed": len(parsed_scores),
        #     f"{dataset_key}/corrupted_skipped": error_count,
        #     f"{dataset_key}/target_sample_size": dataset_cfg.get("sample_size", "ALL")
        # })

    print("All datasets have been successfully loaded!")
    for key, data in all_datasets_data.items():
        print(f" - {key.upper()}: {len(data)} scores ready for feature extraction.")

    wandb.finish()


if __name__ == "__main__":
    main()