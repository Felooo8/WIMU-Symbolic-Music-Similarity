#!/usr/bin/env python3
import json
import os
import subprocess
from pathlib import Path

from dotenv import load_dotenv


def _print_matrix(result_path: Path) -> int:
    if not result_path.exists():
        print("No jsd_matrix.json file found.")
        return 1

    with result_path.open("r", encoding="utf-8") as file:
        matrix = json.load(file)

    print(json.dumps(matrix, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    load_dotenv()
    print("WIMU Symbolic Music Similarity Demo")
    result_path = Path("results/similarity/jsd_matrix.json")

    if os.getenv("WANDB_API_KEY"):
        command = ["make", "all"]
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            print(f"Pipeline command failed: {' '.join(command)}")
            return completed.returncode
        print("Pipeline completed. JSD matrix:")
        return _print_matrix(result_path)

    print("WANDB_API_KEY not set. Generating local sample artifacts instead.")
    completed = subprocess.run(
        ["python3", "generate_sample_results.py"],
        check=False,
    )
    if completed.returncode != 0:
        print("Failed to generate sample artifacts.")
        return completed.returncode

    print("Sample artifacts generated. JSD matrix:")
    return _print_matrix(result_path)


if __name__ == "__main__":
    raise SystemExit(main())
