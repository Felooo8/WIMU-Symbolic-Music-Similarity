import muspy
import numpy as np

from music_features import MusicFeatures


def _build_music() -> muspy.Music:
    track = muspy.Track(program=0)
    track.notes = [
        muspy.Note(time=0, pitch=60, duration=4, velocity=80),
        muspy.Note(time=4, pitch=64, duration=4, velocity=80),
        muspy.Note(time=8, pitch=67, duration=8, velocity=80),
    ]
    return muspy.Music(resolution=24, tracks=[track])


def test_music_features_calc_on_sample_json(tmp_path):
    music = _build_music()
    sample_path = tmp_path / "sample_score.json"
    muspy.save(sample_path, music)

    features = MusicFeatures(measure_resolution=1)
    returned = features.calc(str(sample_path))

    assert len(returned) == 8
    assert features.pitch_class.shape == (12,)
    assert int(features.pitch_class.sum()) == 3
    assert features.pitch_range == 7
    assert features.intervals.shape == (49,)
    assert np.isclose(features.intervals.sum(), 2)
    assert features.durations.tolist() == [4.0, 4.0, 8.0]
    assert 0.0 <= features.scale_consistency <= 1.0
