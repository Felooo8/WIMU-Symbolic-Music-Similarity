import os
import wandb
import glob
import json
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

from music_features import MusicFeatures
from aggregation import Aggregator

def process_single_file(file_path: str):
    try:
        filename = os.path.basename(file_path)
        dataset_name = os.path.basename(os.path.dirname(file_path))

        music_features = MusicFeatures(measure_resolution=1)
        music_features.calc(file_path)

        print(f"[{dataset_name.upper()}] File: {filename}, features:\n {json.dumps(music_features.to_json(), ensure_ascii=False, indent=5)}")

        return dataset_name, music_features
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None


def main():
    load_dotenv()
    api_key = os.getenv("WANDB_API_KEY")
    if api_key:
        wandb.login(key=api_key)
    else:
        raise ValueError("Missing WANDB_API_KEY or .env file!")

    run = wandb.init(
        project="symbolic-music-similarity",
        entity="wimu-team-6-proj-3",
        job_type="feature-extraction",
        name="parallel-feature-extraction"
    )

    print("\nDownloading artifact from W&B...")
    processed_dir = "../data/processed"
    artifact = run.use_artifact('sampled-symbolic-datasets:latest', type='dataset')
    artifact_dir = artifact.download(root=processed_dir)

    search_pattern = os.path.join(artifact_dir, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        print("Json files not found in W&B.")
        wandb.finish()
        return

    print(f"\nFound {len(json_files)} files. Starting analysis of features...\n")
    print("-" * 60)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, f) for f in json_files]

        #for future in as_completed(futures):
        #    print(future.result())

    print("-" * 60)
    wandb.finish()

    data = [item.result() for item in futures if item.result() is not None]
    if len(data) > 0:
        print("\nParallel extraction successfully completed!")
        aggregator = Aggregator(data)
        aggregator.save_features()
        aggregator.create_histograms()


if __name__ == "__main__":
    main()