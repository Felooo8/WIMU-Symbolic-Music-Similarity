import json
from pathlib import Path

import numpy as np

from aggregation import Aggregator
from music_features import MusicFeatures
import check_similarity


def _feature(seed: int) -> MusicFeatures:
    feature = MusicFeatures()
    feature.pitch_class[seed % 12] = 4
    feature.pitch_class[(seed + 5) % 12] = 2
    feature.intervals[24] = 5 + seed
    feature.intervals[25] = 1
    feature.durations = np.array([1.0 + seed, 2.0 + seed])
    feature.pitch_class_entropy = 3.0 + seed
    feature.pitch_entropy = 4.0 + seed
    feature.pitch_range = 10 + seed
    feature.scale_consistency = 0.8
    feature.polyphony = 2.0
    feature.empty_beat_rate = 0.1
    feature.groove_consistency = 0.5
    return feature


def test_pipeline_extraction_to_similarity_outputs():
    unsorted = [
        ("maestro", _feature(1)),
        ("lakh", _feature(2)),
        ("nes", _feature(3)),
    ]
    aggregator = Aggregator(unsorted)
    aggregator.save_features()
    aggregator.create_histograms()

    check_similarity.main()

    matrix_path = Path("results/similarity/jsd_matrix.json")
    summary_path = Path("results/features/summary_stats.json")

    assert matrix_path.exists()
    assert summary_path.exists()

    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    assert "maestro_vs_lakh" in matrix
    assert "maestro_vs_nes" in matrix

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "maestro" in summary
