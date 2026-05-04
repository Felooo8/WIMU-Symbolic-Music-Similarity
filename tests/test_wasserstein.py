import json
from pathlib import Path

import numpy as np

from jsd import calc_jsd
from wasserstein import compute_wasserstein_pair, main


def test_identical_histograms_zero_distance():
    hist = [0, 1, 2, 3]
    assert compute_wasserstein_pair(hist, hist) == 0.0


def test_shifted_histograms_positive_distance():
    base = [1, 0, 0, 0]
    near = [0, 1, 0, 0]
    far = [0, 0, 0, 1]

    distance_near = compute_wasserstein_pair(base, near)
    distance_far = compute_wasserstein_pair(base, far)

    assert distance_near > 0
    assert distance_far > distance_near


def test_jsd_does_not_respect_order():
    base = np.array([1, 0, 0, 0], dtype=float)
    near = np.array([0, 1, 0, 0], dtype=float)
    far = np.array([0, 0, 0, 1], dtype=float)

    jsd_near = calc_jsd(base, near)
    jsd_far = calc_jsd(base, far)

    assert np.isclose(jsd_near, jsd_far)


def test_output_file_exists(tmp_path, monkeypatch):
    repo = tmp_path
    distributions_dir = repo / "results" / "distributions"
    distributions_dir.mkdir(parents=True)

    payload = {
        "nes_mdb": {
            "pitch_class": [1, 2, 3, 4],
            "interval": [1, 0, 0, 1],
            "notes_length": [4, 2, 1, 0],
        },
        "lakh_midi": {
            "pitch_class": [1, 1, 1, 1],
            "interval": [0, 1, 1, 0],
            "notes_length": [2, 2, 2, 2],
        },
        "maestro_v3": {
            "pitch_class": [4, 3, 2, 1],
            "interval": [0, 0, 1, 1],
            "notes_length": [0, 1, 2, 4],
        },
    }
    (distributions_dir / "distributions.json").write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.chdir(repo / "similarity") if (repo / "similarity").exists() else None
    monkeypatch.setattr("wasserstein.Path", Path)

    import wasserstein

    monkeypatch.setattr(wasserstein.Path, "resolve", lambda self: repo / "similarity" / "wasserstein.py")
    main()

    output_path = repo / "results" / "similarity" / "wasserstein_matrix.json"
    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))

    matrix = data["wasserstein_matrix"]
    expected = {"lakh_midi_vs_nes_mdb", "lakh_midi_vs_maestro_v3", "maestro_v3_vs_nes_mdb"}
    assert expected.issubset(set(matrix.keys()))


def test_values_in_reasonable_range(tmp_path, monkeypatch):
    repo = tmp_path
    distributions_dir = repo / "results" / "distributions"
    distributions_dir.mkdir(parents=True)

    payload = {
        "nes_mdb": {
            "pitch_class": [8, 4, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "interval": [1] * 49,
            "notes_length": [6, 3, 2, 1, 1, 1],
        },
        "lakh_midi": {
            "pitch_class": [7, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "interval": [0, 1] * 24 + [1],
            "notes_length": [4, 4, 2, 2, 1, 1],
        },
        "maestro_v3": {
            "pitch_class": [2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0],
            "interval": [0] * 20 + [1] * 29,
            "notes_length": [1, 1, 2, 3, 4, 5],
        },
    }
    (distributions_dir / "distributions.json").write_text(json.dumps(payload), encoding="utf-8")

    import wasserstein

    monkeypatch.setattr(wasserstein.Path, "resolve", lambda self: repo / "similarity" / "wasserstein.py")
    main()

    output_path = repo / "results" / "similarity" / "wasserstein_matrix.json"
    data = json.loads(output_path.read_text(encoding="utf-8"))

    for metrics in data["wasserstein_matrix"].values():
        assert metrics["pitch_class_wasserstein"] >= 0
        assert metrics["interval_wasserstein"] >= 0
        assert metrics["notes_length_wasserstein"] >= 0
        assert metrics["average_wasserstein"] >= 0
        assert metrics["pitch_class_wasserstein"] < 10
