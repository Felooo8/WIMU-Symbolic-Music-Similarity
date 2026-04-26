import json
from pathlib import Path

import numpy as np

from histograms import Histogram
from music_features import MusicFeatures


def _feature(pitch_idx: int, interval_idx: int, durations: list[float]) -> MusicFeatures:
    features = MusicFeatures()
    features.pitch_class[pitch_idx] = 2
    features.intervals[interval_idx] = 1
    features.durations = np.array(durations)
    return features


def test_histogram_creation_serialization(tmp_path):
    result_dir = tmp_path / "hist"
    result_dir.mkdir()

    dataset_features = [
        _feature(0, 24, [1.0, 2.0]),
        _feature(7, 25, [2.0, 4.0]),
    ]

    histogram = Histogram()
    histogram.create_by_pitch_class("demo", dataset_features, 12, str(result_dir))
    histogram.create_by_intervals("demo", dataset_features, 49, str(result_dir))
    histogram.create_by_notes_length("demo", dataset_features, 4, str(result_dir))

    assert (result_dir / "his_pitch_class_dataset_demo.png").exists()
    assert (result_dir / "interval_his_dataset_demo.png").exists()
    assert (result_dir / "notes_length_his_dataset_demo.png").exists()

    assert histogram.serialized["demo"]["pitch_class"][0] == 2.0
    assert histogram.serialized["demo"]["pitch_class"][7] == 2.0
    assert histogram.serialized["demo"]["interval"][24] == 1.0
    assert histogram.serialized["demo"]["interval"][25] == 1.0

    histogram.save_to_json()
    output = Path("results/distributions/distributions.json")
    assert output.exists()

    with output.open("r", encoding="utf-8") as file:
        data = json.load(file)
    assert "demo" in data
    assert data["demo"]["length_note"] is not None
