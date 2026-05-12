import numpy as np
import pytest

from feature_utils import SCALAR_FEATURES, compute_global_stats, dataset_mean, extract_vectors


def _entry(**overrides) -> dict:
    base = {f: 1.0 for f in SCALAR_FEATURES}
    base.update(overrides)
    return base


def test_extract_vectors_shape():
    entries = [_entry(), _entry(pitch_range=5.0)]
    vecs = extract_vectors(entries)
    assert vecs.shape == (2, len(SCALAR_FEATURES))


def test_extract_vectors_filters_nan_rows():
    entries = [_entry(), _entry(pitch_range=float("nan")), _entry()]
    vecs = extract_vectors(entries)
    assert vecs.shape[0] == 2


def test_extract_vectors_raises_when_all_invalid():
    with pytest.raises(ValueError):
        extract_vectors([_entry(pitch_range=float("nan"))])


def test_compute_global_stats_shapes():
    features = {
        "a": [_entry(pitch_range=1.0), _entry(pitch_range=2.0)],
        "b": [_entry(pitch_range=3.0), _entry(pitch_range=4.0)],
    }
    mean, std, cov = compute_global_stats(features)
    n = len(SCALAR_FEATURES)
    assert mean.shape == (n,)
    assert std.shape == (n,)
    assert cov.shape == (n, n)


def test_compute_global_stats_zero_std_becomes_one():
    features = {"a": [_entry(), _entry()]}
    _, std, _ = compute_global_stats(features)
    assert np.all(std == 1.0)


def test_dataset_mean_low_vs_high_pitch_range():
    features = {
        "low": [_entry(pitch_range=1.0)] * 3,
        "high": [_entry(pitch_range=100.0)] * 3,
    }
    mean, std, _ = compute_global_stats(features)
    idx = SCALAR_FEATURES.index("pitch_range")
    assert dataset_mean(features["low"], mean, std)[idx] < dataset_mean(features["high"], mean, std)[idx]


def test_dataset_mean_deterministic():
    features = {
        "a": [_entry(pitch_range=2.0)],
        "b": [_entry(pitch_range=4.0)],
    }
    mean, std, _ = compute_global_stats(features)
    r1 = dataset_mean(features["a"], mean, std)
    r2 = dataset_mean(features["a"], mean, std)
    np.testing.assert_array_equal(r1, r2)


def test_dataset_mean_uses_global_normalization():
    features = {
        "low": [_entry(pitch_range=1.0), _entry(pitch_range=1.0)],
        "high": [_entry(pitch_range=10.0), _entry(pitch_range=10.0)],
    }
    mean, std, _ = compute_global_stats(features)
    idx = SCALAR_FEATURES.index("pitch_range")
    low_z = dataset_mean(features["low"], mean, std)[idx]
    high_z = dataset_mean(features["high"], mean, std)[idx]
    assert low_z < 0
    assert high_z > 0
