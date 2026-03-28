import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import wandb
import glob
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor, as_completed

from features.scalars import MusicScalars

def process_single_file(file_path):
    try:
        filename = os.path.basename(file_path)
        dataset_name = os.path.basename(os.path.dirname(file_path))

        music_scalars = MusicScalars(measure_resolution=1)
        results = music_scalars.calc(file_path)

        return f"[{dataset_name.upper()}] File: {filename}, {music_scalars.get_as_txt()}"
    except Exception as e:
        return f"Error in {file_path}: {e}"


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

    print(f"\nFound {len(json_files)} files. Starting analysis of 'pitch_range'...\n")
    print("-" * 60)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, f) for f in json_files]

        for future in as_completed(futures):
            print(future.result())

    print("-" * 60)
    wandb.finish()
    print("\nParallel extraction successfully completed!")


if __name__ == "__main__":
    main()