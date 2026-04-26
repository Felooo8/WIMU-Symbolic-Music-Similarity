import os
import json
from pathlib import Path
import numpy as np
import logging

from music_features import MusicFeatures
from histograms import Histogram

class  Aggregator:
    def __init__(self, unsorted_datasets: list[tuple[str, MusicFeatures]]):
        self.sorted_datasets: dict[str, list[MusicFeatures]] = {}

        for set in unsorted_datasets:
            self.sorted_datasets.setdefault(set[0], [])
            self.sorted_datasets[set[0]].append(set[1])

        logging.info("\n[AGGREGATOR] datasets sorted by its name.")
        self.summary_stats = self._calculate_summary_stats()

    def _calculate_summary_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        summary: dict[str, dict[str, dict[str, float]]] = {}

        for dataset_name, dataset_features in self.sorted_datasets.items():
            serialized_features = [feature.to_json() for feature in dataset_features]
            summary[dataset_name] = {}

            if not serialized_features:
                continue

            for key in serialized_features[0]:
                values = np.array([row[key] for row in serialized_features], dtype=float)
                summary[dataset_name][key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                }

        return summary

    def save_features(self):
        savepath = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "features"))
        
        if not os.path.exists(savepath):
            savepath.mkdir(parents=True, exist_ok=True)
            print(f"\n[FEATURES] created savepath={savepath}")

        path = os.path.join(savepath, "features.json")

        data = {}
        for name, dataset_features in self.sorted_datasets.items():
            data.setdefault(name, [])
            for feature in dataset_features:
                data[name].append(feature.to_json())

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=5)

        logging.info(f"\n[FEATURES] saved in path={path}")

        summary_path = os.path.join(savepath, "summary_stats.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_stats, f, ensure_ascii=False, indent=5)

        logging.info(f"\n[FEATURES] summary stats saved in path={summary_path}")
        

    def create_histograms(self):
        resultpath = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "histograms"))
        if not os.path.exists(resultpath):
            resultpath.mkdir(parents=True, exist_ok=True)
            logging.info(f"\n[HISTOGRAM] created resultpath={resultpath}")

        his = Histogram()

        for name, dataset_features in self.sorted_datasets.items():
            his.create_by_pitch_class(name, dataset_features, 12, resultpath)
            his.create_by_intervals(name, dataset_features, 49, resultpath)
            his.create_by_notes_length(name, dataset_features, 16, resultpath)

        his.save_to_json()