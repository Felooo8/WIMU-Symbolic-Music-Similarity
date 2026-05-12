import numpy as np
import pytest

from mahalanobis_dist import _compute_matrix


def _eye(n: int) -> np.ndarray:
    return np.eye(n)


def test_compute_matrix_pair_keys():
    means = {
        "lakh_midi_rock": np.zeros(3),
        "maestro_v3": np.ones(3),
        "nes_mdb": np.full(3, 2.0),
    }
    matrix = _compute_matrix(means, _eye(3))
    assert "lakh_midi_rock_vs_maestro_v3" in matrix
    assert "lakh_midi_rock_vs_nes_mdb" in matrix
    assert "maestro_v3_vs_nes_mdb" in matrix


def test_compute_matrix_no_self_pairs():
    means = {"a": np.zeros(2), "b": np.ones(2)}
    matrix = _compute_matrix(means, _eye(2))
    assert "a_vs_a" not in matrix
    assert "b_vs_b" not in matrix


def test_compute_matrix_non_negative():
    means = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}
    matrix = _compute_matrix(means, _eye(2))
    for vals in matrix.values():
        assert vals["mahalanobis"] >= 0


def test_compute_matrix_zero_for_identical_means():
    means = {"a": np.ones(3), "b": np.ones(3)}
    matrix = _compute_matrix(means, _eye(3))
    assert matrix["a_vs_b"]["mahalanobis"] == pytest.approx(0.0, abs=1e-9)


def test_compute_matrix_farther_is_larger():
    means = {
        "near": np.array([0.1, 0.0]),
        "far": np.array([5.0, 0.0]),
        "ref": np.zeros(2),
    }
    matrix = _compute_matrix(means, _eye(2))
    assert matrix["far_vs_ref"]["mahalanobis"] > matrix["near_vs_ref"]["mahalanobis"]


def test_compute_matrix_pair_count():
    means = {k: np.zeros(2) for k in ["a", "b", "c", "d"]}
    matrix = _compute_matrix(means, _eye(2))
    assert len(matrix) == 6
