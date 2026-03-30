import numpy as np
import math
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path

from music_features import MusicFeatures

class Histogram:
    def __init__(self):
        self.serialized = {}

    def _add_serializable_data(self, dataset_name: str, dis_name: str, distribution: np.ndarray):
        self.serialized.setdefault(dataset_name, {"pitch_class": None, "interval": None, "length_note": None})
        self.serialized[dataset_name][dis_name] = distribution.tolist()

    def save_to_json(self):
        savepath = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "distributions"))
        
        if not os.path.exists(savepath):
            savepath.mkdir(parents=True, exist_ok=True)
            print(f"\n[DISTRIBUTIONS] the savepath created (savepath={savepath})")

        path = os.path.join(savepath, "distributions.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.serialized, f, ensure_ascii=False, indent=5)

        print(f"\n[DISTRIBUTIONS] saved in filepath={path}")

    @staticmethod
    def save_plot(data: np.ndarray, bins: list, title: str, xlabel: str, ylabel: str, savepath: str):
        plt.figure()
        plt.bar(bins, data)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if len(bins) < 20:
            plt.xticks(bins, bins)
        plt.savefig(savepath)

    def create_by_pitch_class(self, dataset_name: str, dataset_features: list[MusicFeatures], bin_number: int, resultpath: str):
        distribution = np.zeros(bin_number)
        for features in dataset_features:
            distribution += features.pitch_class

        print(f"[HISTOGRAM] histogram pitch class created for dataset named {dataset_name}: {distribution}")

        filename = f"his_pitch_class_dataset_{dataset_name}.png"

        Histogram.save_plot(data=distribution,  
                            bins=list(range(bin_number)),
                            title=f"Histogram pitch class datasetu\n{dataset_name}",
                            xlabel="Pitch Class",
                            ylabel="Liczba",
                            savepath=os.path.join(resultpath, filename))
        
        
        self._add_serializable_data(dataset_name, "pitch_class", distribution)
        print(f"[HISTOGRAM] histogram saved as {filename}.")

    def create_by_intervals(self, dataset_name: str, dataset_features: list[MusicFeatures], bin_number: int, resultpath: str):
        distribution = np.zeros(bin_number)
        for features in dataset_features:
            distribution += features.intervals

        print(f"[HISTOGRAM] interval histogram created for dataset named {dataset_name}: {distribution}")

        filename = f"interval_his_dataset_{dataset_name}.png"

        Histogram.save_plot(data=distribution,  
                            bins=list(range(bin_number)),
                            title=f"Histogram interwałów (±24 półtony)\ndatasetu {dataset_name}",
                            xlabel="Interwał",
                            ylabel="Liczba",
                            savepath=os.path.join(resultpath, filename))
        
        self._add_serializable_data(dataset_name, "interval", distribution)
        print(f"[HISTOGRAM] histogram saved as {filename}.")

    def create_by_notes_length(self, dataset_name: str, dataset_features: list[MusicFeatures], bin_number: int, resultpath: str):
        dataset_durations = np.array([])
        for features in dataset_features:
            dataset_durations = np.append(dataset_durations, features.durations)

        note_duration_min = min(dataset_durations)
        note_duration_max = max(dataset_durations)

        bins = np.logspace(math.log10(note_duration_min), math.log10(note_duration_max), bin_number + 1)
        distribution, _ = np.histogram(dataset_durations, bins=bins) 

        print(f"[HISTOGRAM] notes length histogram created for dataset named {dataset_name}: {distribution}")

        filename = f"notes_length_his_dataset_{dataset_name}.png"

        Histogram.save_plot(data=distribution,  
                            bins=list(range(bin_number)),
                            title=f"Histogram długości nut\ndatasetu {dataset_name}",
                            xlabel="Długść Nut",
                            ylabel="Liczba",
                            savepath=os.path.join(resultpath, filename))
        
        self._add_serializable_data(dataset_name, "length_note", distribution)
        print(f"[HISTOGRAM] histogram saved as {filename}.")
