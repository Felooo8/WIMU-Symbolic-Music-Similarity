import numpy as np
import math
import os
import json
import matplotlib.pyplot as plt
from pathlib import Path
from numpy.typing import ArrayLike
import logging

from music_features import MusicFeatures


class Histogram:
    def __init__(self):
        self.serialized = {}

    def _add_serializable_data(
        self,
        dataset_name: str,
        dis_name: str,
        genres: list[str],
        distribution: np.ndarray
    ):


        self.serialized.setdefault(dataset_name, {
            "pitch_class": {},
            "interval": {},
            "length_note": {}
        })

        target = self.serialized[dataset_name][dis_name]
        for i, genre in enumerate(genres):
            target[genre] = distribution[i].tolist()

    def save_to_json(self):
        savepath = Path(
            os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "results",
                "distributions"
            )
        )

        savepath.mkdir(parents=True, exist_ok=True)

        path = os.path.join(savepath, "distributions.json")

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.serialized, f, ensure_ascii=False, indent=5)

        logging.info(f"\n[DISTRIBUTIONS] saved in filepath={path}")


    @staticmethod
    def save_plot(
        data: np.ndarray,
        bins: ArrayLike,
        genres: list[str],
        title: str,
        xlabel: str,
        ylabel: str,
        savepath: str,
        bin_edges: list = None,
        scale_type: str = "linear"
    ):
        genres = list(genres)

        if len(genres) > 1:
            n = len(genres)
            cols = 2
            rows = (n + cols - 1) // cols

            fig, ax = plt.subplots(rows, cols, figsize=(10, 5 * rows))
            fig.suptitle(title)

            ax = np.atleast_2d(ax)

            for i in range(n):
                r, c = divmod(i, cols)

                ax[r, c].bar(bins, data[i], edgecolor="black")
                ax[r, c].set_title(f"dla \"{genres[i]}\" (liczba utworów: {len(data[i])})")
                ax[r, c].set_xlabel(xlabel)
                ax[r, c].set_ylabel(ylabel)
                ax[r, c].set_xscale(scale_type)

            for i in range(n, rows * cols):
                r, c = divmod(i, cols)
                ax[r, c].axis("off")

        else:
            plt.title(title + f" dla gatunku {genres[0]} (liczba utworów: {len(data[0])})")

            plt.bar(bins, data[0], edgecolor="black")
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.xscale(scale_type)

        plt.subplots_adjust(wspace=0.5)
        plt.savefig(savepath, dpi=600, bbox_inches="tight")
        plt.close()

    def create_by_pitch_class(
        self,
        dataset_name: str,
        dataset_features: list[MusicFeatures],
        bin_number: int,
        resultpath: str
    ):
        unique_genres = list(set(f.genre for f in dataset_features))

        distributions = np.zeros((len(unique_genres), bin_number))

        for features in dataset_features:
            genre_index = unique_genres.index(features.genre)
            distributions[genre_index] += features.pitch_class

        filename = f"his_pitch_class_dataset_{dataset_name}.png"

        Histogram.save_plot(
            data=distributions,
            genres=unique_genres,
            bins=list(range(bin_number)),
            title=f"Histogram pitch class datasetu\n{dataset_name}",
            xlabel="Pitch Class",
            ylabel="Liczba",
            savepath=os.path.join(resultpath, filename)
        )
        logging.info(f"[HISTOGRAM] pitch class histogram created")

        self._add_serializable_data(
            dataset_name,
            "pitch_class",
            unique_genres,
            distributions
        )

    def create_by_intervals(
        self,
        dataset_name: str,
        dataset_features: list[MusicFeatures],
        bin_number: int,
        resultpath: str
    ):
        unique_genres = list(set(f.genre for f in dataset_features))

        distributions = np.zeros((len(unique_genres), bin_number))

        for features in dataset_features:
            genre_index = unique_genres.index(features.genre)
            distributions[genre_index] += features.intervals

        logging.info(f"[HISTOGRAM] interval histogram created")

        filename = f"interval_his_dataset_{dataset_name}.png"


        Histogram.save_plot(
            data=distributions,
            genres=unique_genres,
            bins=np.arange(-24, 25),
            title=f"Histogram interwałów (±24 półtony)\n{dataset_name}",
            xlabel="Interwał",
            ylabel="Liczba",
            savepath=os.path.join(resultpath, filename)
        )

        self._add_serializable_data(
            dataset_name,
            "interval",
            unique_genres,
            distributions
        )

    def create_by_notes_length(
        self,
        dataset_name: str,
        dataset_features: list[MusicFeatures],
        bin_number: int,
        resultpath: str
    ):
        unique_genres = list(set(f.genre for f in dataset_features))

        distributions = np.zeros((len(unique_genres), bin_number))

        genre_durations = {genre: [] for genre in unique_genres}

        for features in dataset_features:
            genre_durations[features.genre].extend(features.durations)


        bin_edges = None

        for i, genre in enumerate(unique_genres):
            data = np.array(genre_durations[genre])

            if len(data) == 0:
                continue

            mn, mx = data.min(), data.max()

            if mn == mx:
                mx += 1e-6

            bin_edges = np.logspace(
                math.log10(mn),
                math.log10(mx),
                bin_number + 1
            )

            distributions[i], _ = np.histogram(data, bins=bin_edges)

        logging.info(f"[HISTOGRAM] notes length histogram created")

        filename = f"notes_length_his_dataset_{dataset_name}.png"

        bin_range = (bin_edges[:-1] + bin_edges[1:]) / 2

        Histogram.save_plot(
            data=distributions,
            genres=unique_genres,
            bins=bin_range,
            title=f"Histogram długości nut\n{dataset_name}",
            xlabel="Długość nut",
            ylabel="Liczba",
            savepath=os.path.join(resultpath, filename),
            bin_edges=bin_edges,
            scale_type="log"
        )

        self._add_serializable_data(
            dataset_name,
            "length_note",
            unique_genres,
            distributions
        )
