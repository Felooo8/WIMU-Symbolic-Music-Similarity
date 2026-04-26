#import sys
import os

#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
from pathlib import Path
import numpy as np

from music_features import MusicFeatures
from histograms import Histogram

class  Aggregator:
    def __init__(self, unsorted_datasets: list[tuple[str, MusicFeatures]]):
        self.sorted_datasets: dict[str, list[MusicFeatures]] = {}

        for set in unsorted_datasets:
            self.sorted_datasets.setdefault(set[0], [])
            self.sorted_datasets[set[0]].append(set[1])

        print("\n[AGGREGATOR] datasets sorted by its name.")
        self.summary_stats = self._calculate_summary_stats()

    def _calculate_summary_stats(self) -> dict[str, dict[str, dict[str, float]]]:
        summary: dict[str, dict[str, dict[str, float]]] = {}

        all_features = []
        for dataset_features in self.sorted_datasets.values():
            all_features.extend([f.to_json() for f in dataset_features])

        if not all_features:
            return summary

        genres = list(set(row['genre'] for row in all_features))

        for genre in genres:
            genre_features = [row for row in all_features if row['genre'] == genre]
            summary[genre] = {}

            for key in all_features[0].keys():
                if key == "genre":
                    continue

                values = np.array(
                    [row[key] for row in genre_features if not isinstance(row[key], str)
                    and not np.isnan(row[key])],
                    dtype=float
                )

                if len(values) == 0:
                    continue

                summary[genre][key] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values)),
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

        print(f"\n[FEATURES] saved in path={path}")

        summary_path = os.path.join(savepath, "summary_stats.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary_stats, f, ensure_ascii=False, indent=5)

        print(f"\n[FEATURES] summary stats saved in path={summary_path}")
        

    def create_histograms(self):
        resultpath = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "histograms"))
        if not os.path.exists(resultpath):
            resultpath.mkdir(parents=True, exist_ok=True)
            print(f"\n[HISTOGRAM] created resultpath={resultpath}")

        his = Histogram()

        for name, dataset_features in self.sorted_datasets.items():
            his.create_by_pitch_class(name, dataset_features, 12, resultpath)
            his.create_by_intervals(name, dataset_features, 49, resultpath)
            his.create_by_notes_length(name, dataset_features, 16, resultpath)

        his.save_to_json()
    
