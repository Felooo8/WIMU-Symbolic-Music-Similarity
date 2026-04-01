#!/usr/bin/env python3
import logging
import os
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _run_check(command: list[str]) -> None:
    logging.info("Running check: %s", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        raise AssertionError(f"Command failed: {' '.join(command)}")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def check_prototype_readiness() -> None:
    required_files = [
        Path("results/similarity/jsd_matrix.json"),
        Path("results/features/summary_stats.json"),
    ]
    for path in required_files:
        _remove_if_exists(path)

    _run_check([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short"])
    #_run_check(["python", "-m", "pytest", "tests/", "-v", "--tb=short"])

    if os.getenv("WANDB_API_KEY"):
        _run_check(["make", "download-data"])
        _run_check(["make", "run-extraction"])
        _run_check(["make", "run-similarity"])
    else:
        logging.warning("WANDB_API_KEY not set; using local sample-artifact generation.")
        _run_check(["python3", "generate_sample_results.py"])

    for path in required_files:
        if not path.is_file():
            raise AssertionError(f"Missing required artifact: {path}")

    logging.info("Verification completed successfully.")


if __name__ == "__main__":
    check_prototype_readiness()
