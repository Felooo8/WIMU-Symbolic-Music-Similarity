import os
import random
import logging
import tarfile
from pathlib import Path
import gdown
import muspy
from abc import ABC, abstractmethod


class BaseDatasetProvider(ABC):

    def __init__(self, cfg):
        self.name = cfg["name"]
        self.path = cfg["path"]
        self.sample_size = cfg.get("sample_size", float('inf'))
        self.cfg = cfg

    @abstractmethod
    def prepare_and_get_dataset(self):
        pass

    def get_track_id(self, index, dataset) -> str | None:
        return None

    def process(self):
        logging.info(f"\n[{self.name}] Starting processing (Limit: {self.sample_size} files)...")

        dataset = self.prepare_and_get_dataset()

        valid_scores = []
        errors = 0
        indices = list(range(len(dataset)))
        random.shuffle(indices)

        for i in indices:
            if len(valid_scores) >= self.sample_size:
                break
            try:
                score = dataset[i]
                if score is not None and len(score.tracks) > 0:
                    track_id = self.get_track_id(i, dataset)
                    valid_scores.append((score, track_id))
                else:
                    errors += 1
            except Exception as e:
                logging.warning(f"[{self.name}] Ignored file at index {i}. Reason: {e}")
                errors += 1

        logging.info(f"[{self.name}] Finished. Valid: {len(valid_scores)}, Errors: {errors}")
        return valid_scores, errors


class NativeMuspyProvider(BaseDatasetProvider):

    _MAPPING = {
        "maestro_v3": muspy.MAESTRODatasetV3,
        "lakh_midi": muspy.LakhMIDIAlignedDataset,
        "music_net": muspy.MusicNetDataset,
    }

    def prepare_and_get_dataset(self):
        DatasetClass = self._MAPPING[self.name]
        os.makedirs(os.path.join("data", "raw"), exist_ok=True)
        dataset = DatasetClass(self.path, download_and_extract=True)
        self._dataset = dataset
        return dataset

    def get_maestro_metadata(self, score):
        if self.name != "maestro_v3":
            return {}

        filename = os.path.basename(score.metadata.source_filename)  

        return self.maestro_map.get(filename, {})

    def get_track_id(self, index, dataset) -> str | None:
        if self.name == "lakh_midi":
            try:
                file_path = Path(dataset._filenames[index])
                track_id = file_path.parts[-2]
                if track_id.startswith("TR"):
                    return track_id
            except Exception as e:
                logging.warning(f"[lakh_midi] Track ID error: {e}")
        return None


class LakhMidiGenreProvider(BaseDatasetProvider):

    def prepare_and_get_dataset(self):
        genre_file = self.cfg.get("genre_file")
        if not genre_file or not os.path.exists(genre_file):
            raise FileNotFoundError(f"[lakh_midi] Genre file not found: {genre_file}")

        self._target_genre = self.cfg["genre"]
        genre_map: dict[str, str] = {}
        with open(genre_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    genre_map[parts[0]] = parts[1]

        available_genres = set(genre_map.values())
        if self._target_genre not in available_genres:
            raise ValueError(
                f"[lakh_midi] Genre '{self._target_genre}' not found in {genre_file}. "
                f"Available genres: {sorted(available_genres)}"
            )

        self._dataset = muspy.LakhMIDIAlignedDataset(self.path, download_and_extract=True)
        self._filtered_indices = [
            i for i, filename in enumerate(self._dataset._filenames)
            if genre_map.get(Path(filename).parts[-2]) == self._target_genre
        ]

        if not self._filtered_indices:
            raise ValueError(f"[lakh_midi] No tracks found for genre '{self._target_genre}'")

        logging.info(
            f"[lakh_midi] genre='{self._target_genre}': {len(self._filtered_indices)} tracks found"
        )
        return self

    def __len__(self):
        return len(self._filtered_indices)

    def __getitem__(self, index):
        return self._dataset[self._filtered_indices[index]]

    def get_track_id(self, index, dataset) -> str | None:
        real_index = self._filtered_indices[index]
        try:
            file_path = Path(self._dataset._filenames[real_index])
            track_id = file_path.parts[-2]
            if track_id.startswith("TR"):
                return track_id
        except Exception as e:
            logging.warning(f"[lakh_midi] Track ID error: {e}")
        return None


class Music21JsbProvider(BaseDatasetProvider):

    def prepare_and_get_dataset(self):
        from music21 import corpus
        all_paths = corpus.getComposer("bach")
        self._paths = [p for p in all_paths if "bwv" in str(p).lower()]
        logging.info(f"[jsb_chorales] Found {len(self._paths)} chorales in music21 corpus")
        return self

    def __len__(self):
        return len(self._paths)

    def __getitem__(self, index):
        from music21 import corpus
        score = corpus.parse(self._paths[index])
        return muspy.from_music21(score)


class NesMdbProvider(BaseDatasetProvider):

    def prepare_and_get_dataset(self):
        os.makedirs(self.path, exist_ok=True)
        success_file = os.path.join(self.path, ".muspy.success")

        if not os.path.exists(success_file):
            gdrive_url = self.cfg.get("url")
            archive_path = os.path.join(self.path, "nesmdb_midi.tar.gz")

            gdown.download(url=gdrive_url, output=archive_path, quiet=False, fuzzy=True)

            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self.path)

            open(success_file, "w").close()
            os.remove(archive_path)

        return muspy.NESMusicDatabase(self.path, download_and_extract=False)


class DatasetFactory:
    @staticmethod
    def create(dataset_cfg):
        name = dataset_cfg["name"]

        if dataset_cfg.get("genre"):
            return LakhMidiGenreProvider(dataset_cfg)
        elif name in NativeMuspyProvider._MAPPING:
            return NativeMuspyProvider(dataset_cfg)
        elif name == "nes_mdb":
            return NesMdbProvider(dataset_cfg)
        elif name == "jsb_chorales":
            return Music21JsbProvider(dataset_cfg)
        else:
            raise ValueError(f"Unsupported dataset configuration: {name}")
