import os
import re
import json
import subprocess
import shutil
import numpy as np
from collections import defaultdict
from pathlib import Path

SIZES = [10, 50, 100, 400, 1000]
RUNS = 1
CONFIG_PATH = "configs/config.yaml"
RESULTS_ROOT = Path("results")
FINAL_RESULTS_PATH = RESULTS_ROOT / "final_averaged_results.json"

FOLDERS_TO_ARCHIVE = [
    "analysis",
    "features",
    "fmd",
    "histograms",
    "similarity",
    "distributions"
]

CRITICAL_COMMANDS = {
    "ingestion/dataset-ingestion.py": True,
    "features/features_extraction.py": True,
    "similarity/check_similarity.py": True,
    "similarity/wasserstein.py": True,
    "similarity/euclidean.py": True,
    "similarity/mahalanobis_dist.py": True,
    "fmd/compute_fmd.py": False,
    "analysis/correlation.py": True,
    "analysis/baseline_classifier.py": False
}

COMMANDS = [
    ["poetry", "run", "python", "ingestion/dataset-ingestion.py"],
    ["poetry", "run", "python", "features/features_extraction.py"],
    ["poetry", "run", "python", "similarity/check_similarity.py"],
    ["poetry", "run", "python", "similarity/wasserstein.py"],
    ["poetry", "run", "python", "similarity/euclidean.py"],
    ["poetry", "run", "python", "similarity/mahalanobis_dist.py"],
    ["poetry", "run", "python", "fmd/compute_fmd.py"],
    ["poetry", "run", "python", "analysis/correlation.py"],
    ["poetry", "run", "python", "analysis/baseline_classifier.py"]
]


def set_sample_size(size: int):
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = re.sub(r"sample_size:\s*\d+", f"sample_size: {size}", content)
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        f.write(new_content)


def clean_workspace():
    if os.path.exists("data/processed"):
        shutil.rmtree("data/processed")
    for folder_name in FOLDERS_TO_ARCHIVE:
        target = RESULTS_ROOT / folder_name
        if target.exists():
            shutil.rmtree(target)


def archive_run_results(size: int, run_idx: int, current_run_stats: dict):
    archive_dir = RESULTS_ROOT / f"N_{size}" / f"run_{run_idx + 1}"
    archive_dir.mkdir(parents=True, exist_ok=True)

    with open(archive_dir / "run_results.json", "w", encoding="utf-8") as f:
        json.dump(current_run_stats, f, indent=4)

    for folder_name in FOLDERS_TO_ARCHIVE:
        source = RESULTS_ROOT / folder_name
        if source.exists():
            shutil.move(str(source), str(archive_dir / folder_name))


def run_pipeline():
    clean_workspace()
    for cmd in COMMANDS:
        script_path = cmd[3]
        result = subprocess.run(cmd)
        if result.returncode != 0:
            if CRITICAL_COMMANDS.get(script_path, True):
                print(f"  [!] Przerwano potok na skrypcie {script_path}.")
                return False
    return True


def main():
    stats = {size: defaultdict(list) for size in SIZES}
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    for size in SIZES:
        print(f"\n=======================================================")
        print(f"ROZPOCZYNAMY TESTY DLA PRÓBKI: N = {size}")
        print(f"=======================================================")
        set_sample_size(size)

        for run_idx in range(RUNS):
            print(f"\n--- Przebieg {run_idx + 1}/{RUNS} dla N={size} ---")
            run_pipeline()

            corr_path = RESULTS_ROOT / "analysis" / "correlation.json"
            current_run_stats = {}

            if corr_path.exists():
                try:
                    with open(corr_path, "r", encoding="utf-8") as f:
                        corr_data = json.load(f)

                    metrics_data = corr_data.get("vs_fmd", {})

                    added_count = 0
                    for metric_name, values in metrics_data.items():
                        if isinstance(values, dict) and "rho" in values:
                            val = values["rho"]
                            if val is not None and not np.isnan(val):
                                stats[size][metric_name].append(val)
                                current_run_stats[metric_name] = val
                                added_count += 1

                    print(f"  [SUKCES] Odczytano {added_count} metryk z pliku korelacji!")
                except Exception as e:
                    print(f"  [BŁĄD ODCZYTU] Plik jest uszkodzony: {e}")
            else:
                print(f"  [BŁĄD] Plik {corr_path} NIE ISTNIEJE.")

            archive_run_results(size, run_idx, current_run_stats)

    final_report = {}
    for size in SIZES:
        final_report[size] = {}
        for metric in sorted(stats[size].keys()):
            rhos = stats[size][metric]
            if len(rhos) > 0:
                mean_rho = float(np.mean(rhos))
                std_rho = float(np.std(rhos)) if len(rhos) > 1 else 0.0
                final_report[size][metric] = {
                    "mean_rho": mean_rho,
                    "std_rho": std_rho,
                    "raw_runs": rhos
                }

    with open(FINAL_RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=4)
    print(f"\n[ORCHESTRATOR] Gotowe! Zapisano uśredniony raport w: {FINAL_RESULTS_PATH}")


if __name__ == "__main__":
    main()