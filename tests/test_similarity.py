import numpy as np

from jsd import calc_jsd


def test_jsd_is_zero_for_identical_distributions():
    dist = np.array([0.5, 0.5])
    assert calc_jsd("a", "b", dist, dist) == 0.0


def test_jsd_is_symmetric_and_positive():
    left = np.array([0.9, 0.1])
    right = np.array([0.1, 0.9])

    value_ab = calc_jsd("a", "b", left, right)
    value_ba = calc_jsd("b", "a", right, left)

    assert value_ab > 0.0
    assert np.isclose(value_ab, value_ba)
