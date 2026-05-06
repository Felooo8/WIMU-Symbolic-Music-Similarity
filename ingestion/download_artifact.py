import os
import yaml
import wandb
from pathlib import Path
from dotenv import load_dotenv


def _load_config(repo_root: Path) -> dict:
    with (repo_root / "configs" / "config.yaml").open() as f:
        return yaml.safe_load(f)


def _download(config: dict, output_dir: Path) -> None:
    wandb.login()
    run = wandb.init(
        project=config["wandb"]["project"],
        entity=config["wandb"]["entity"],
        job_type="download-artifact",
    )
    artifact = run.use_artifact("sampled-symbolic-datasets:latest", type="dataset")
    artifact.download(root=str(output_dir))
    run.finish()


def main() -> None:
    load_dotenv()
    if not os.getenv("WANDB_API_KEY"):
        raise EnvironmentError("Missing WANDB_API_KEY — add it to .env file")

    repo_root = Path(__file__).resolve().parents[1]
    config = _load_config(repo_root)
    output_dir = repo_root / "data" / "processed"

    output_dir.mkdir(parents=True, exist_ok=True)
    _download(config, output_dir)


if __name__ == "__main__":
    main()
