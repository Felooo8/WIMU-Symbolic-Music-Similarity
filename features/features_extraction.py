import os
import wandb
import glob
import json
from dotenv import load_dotenv
from concurrent.futures import ProcessPoolExecutor
import logging

from music_features import MusicFeatures
from aggregation import Aggregator
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def process_single_file(file_path: str):
    try:
        filename = os.path.basename(file_path)
        dataset_name = os.path.basename(os.path.dirname(file_path))

        music_features = MusicFeatures(measure_resolution=1)
        music_features.calc(file_path)

        logging.info(f"[{dataset_name.upper()}] File: {filename}, genre:{music_features.genre} \features:\n {json.dumps(music_features.to_json(), ensure_ascii=False, indent=5)}")

        return dataset_name, music_features
    except Exception as e:
        logging.error(f"Error in {file_path}: {e}")
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

    logging.info("\nDownloading artifact from W&B...")
    processed_dir = "../data/processed"
    artifact = run.use_artifact('sampled-symbolic-datasets:latest', type='dataset')
    artifact_dir = artifact.download(root=processed_dir)

    search_pattern = os.path.join(artifact_dir, "**", "*.json")
    json_files = glob.glob(search_pattern, recursive=True)

    if not json_files:
        logging.error("Json files not found in W&B.")
        wandb.finish()
        return

    logging.error(f"\nFound {len(json_files)} files. Starting analysis of features...\n")
    logging.info("-" * 60)

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_single_file, f) for f in json_files]

        #for future in as_completed(futures):
        #    print(future.result())

    logging.info("-" * 60)
    wandb.finish()

    data = [item.result() for item in futures if item.result() is not None]
    if len(data) > 0:
        logging.info("\nParallel extraction successfully completed!")
        aggregator = Aggregator(data)
        aggregator.save_features()
        aggregator.create_histograms()
    
    if len(data) > 0:
        print("\nLogging to WandB...")
        
        total_processed = len(data)
        error_rate = (len(json_files) - total_processed) / len(json_files)
        
        wandb.log({
            "total_files": len(json_files),
            "files_processed": total_processed,
            "error_rate": error_rate,
            "datasets": {"maestro_v3": sum(1 for d, _ in data if d == "maestro_v3"),
                        "lakh_midi": sum(1 for d, _ in data if d == "lakh_midi"),
                        "nes_mdb": sum(1 for d, _ in data if d == "nes_mdb")}
        })
        
        sample_features = [{"dataset": d, **mf.to_json()} for d, mf in data[:10]]
        if sample_features:
            columns = list(sample_features[0].keys())
            rows = [[item[column] for column in columns] for item in sample_features]
            wandb.log({"sample_features": wandb.Table(columns=columns, data=rows)})
        
        aggregator = Aggregator(data)
        summary_stats = aggregator.summary_stats
        wandb.log(summary_stats)
        
        print("✅ Logged to WandB!")

    wandb.finish()


if __name__ == "__main__":
    main()
