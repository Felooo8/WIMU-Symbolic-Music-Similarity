import os
import logging
import tarfile
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

    def process(self):
        logging.info(f"\n[{self.name}] Starting processing (Limit: {self.sample_size} files)...")

        dataset = self.prepare_and_get_dataset()

        valid_scores = []
        errors = 0

        for i in range(len(dataset)):
            if i >= self.sample_size:
                logging.info(f"[{self.name}] Sample limit reached ({self.sample_size}).")
                break

            try:
                score = dataset[i]
                if score is not None and len(score.tracks) > 0:
                    valid_scores.append(score)
                else:
                    logging.warning(f"[{self.name}] Ignored file at index {i}. Reason: Empty/No tracks.")
                    errors += 1
            except Exception as e:
                logging.warning(f"[{self.name}] Ignored file at index {i}. Reason: {type(e).__name__}: {e}")
                errors += 1

        logging.info(f"[{self.name}] Finished. Valid: {len(valid_scores)}, Errors: {errors}")
        return valid_scores, errors


class NativeMuspyProvider(BaseDatasetProvider):
    _MAPPING = {
        "maestro_v3": muspy.MAESTRODatasetV3,
        "lakh_midi": muspy.LakhMIDIAlignedDataset,
    }

    def prepare_and_get_dataset(self):
        DatasetClass = self._MAPPING[self.name]
        return DatasetClass(self.path, download_and_extract=True)


class NesMdbProvider(BaseDatasetProvider):

    def prepare_and_get_dataset(self):
        os.makedirs(self.path, exist_ok=True)
        success_file = os.path.join(self.path, ".muspy.success")

        if not os.path.exists(success_file):
            gdrive_url = self.cfg.get("url")
            if not gdrive_url:
                raise ValueError("Missing 'url' in config for NES-MDB!")

            logging.info(f"[{self.name}] Downloading via GDrive bypass...")
            archive_path = os.path.join(self.path, "nesmdb_midi.tar.gz")
            gdown.download(url=gdrive_url, output=archive_path, quiet=False, fuzzy=True)

            logging.info(f"[{self.name}] Extracting...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self.path)

            with open(success_file, "w") as f:
                f.write("")
            os.remove(archive_path)

        return muspy.NESMusicDatabase(self.path, download_and_extract=False)


class DatasetFactory:
    @staticmethod
    def create(dataset_cfg):
        name = dataset_cfg["name"]
        if name in NativeMuspyProvider._MAPPING:
            return NativeMuspyProvider(dataset_cfg)
        elif name == "nes_mdb":
            return NesMdbProvider(dataset_cfg)
        else:
            raise ValueError(f"Unsupported dataset configuration for: {name}")