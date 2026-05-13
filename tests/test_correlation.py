import json
from pathlib import Path

import numpy as np
import pytest

from correlation import SpearmanAnalysis


def test_spearman_perfect_positive():
    a = {"p1": 1.0, "p2": 2.0, "p3": 3.0}
    b = {"p1": 10.0, "p2": 20.0, "p3": 30.0}
    rho, _ = SpearmanAnalysis._spearman(a, b, ["p1", "p2", "p3"])
    assert rho == pytest.approx(1.0)


def test_spearman_perfect_negative():
    a = {"p1": 3.0, "p2": 2.0, "p3": 1.0}
    b = {"p1": 1.0, "p2": 2.0, "p3": 3.0}
    rho, _ = SpearmanAnalysis._spearman(a, b, ["p1", "p2", "p3"])
    assert rho == pytest.approx(-1.0)


def test_spearman_too_few_pairs_returns_nan():
    a = {"p1": 1.0}
    b = {"p1": 2.0}
    rho, pvalue = SpearmanAnalysis._spearman(a, b, ["p1"])
    assert np.isnan(rho)
    assert np.isnan(pvalue)


def test_spearman_ignores_pairs_missing_from_either():
    a = {"p1": 1.0, "p2": 2.0, "p3": 3.0}
    b = {"p1": 3.0, "p3": 1.0}
    rho, _ = SpearmanAnalysis._spearman(a, b, ["p1", "p2", "p3"])
    assert rho == pytest.approx(-1.0)



def test_z_normalize_constant_returns_zeros():
    values = {"p1": 5.0, "p2": 5.0, "p3": 5.0}
    result = SpearmanAnalysis._z_normalize(values)
    assert all(v == 0.0 for v in result.values())


def test_z_normalize_preserves_order():
    values = {"p1": 1.0, "p2": 2.0, "p3": 3.0}
    result = SpearmanAnalysis._z_normalize(values)
    assert result["p1"] < result["p2"] < result["p3"]


def test_z_normalize_mean_is_zero():
    values = {"p1": 1.0, "p2": 2.0, "p3": 3.0}
    result = SpearmanAnalysis._z_normalize(values)
    assert sum(result.values()) == pytest.approx(0.0, abs=1e-9)


def test_combine_uses_pair_intersection():
    sa = SpearmanAnalysis(Path("."), Path("."))
    metrics = {
        "m_a": {"p1": 1.0, "p2": 2.0, "p3": 3.0},
        "m_b": {"p1": 3.0, "p2": 1.0},
    }
    result = sa._combine(metrics, ["m_a", "m_b"])
    assert set(result.keys()) == {"p1", "p2"}


def test_combine_empty_when_components_absent():
    sa = SpearmanAnalysis(Path("."), Path("."))
    result = sa._combine({"only": {"p1": 1.0}}, ["missing_a", "missing_b"])
    assert result == {}


def test_combine_single_component_passes_through():
    sa = SpearmanAnalysis(Path("."), Path("."))
    metrics = {"m": {"p1": 1.0, "p2": -1.0}}
    result = sa._combine(metrics, ["m"])
    assert set(result.keys()) == {"p1", "p2"}


def test_compute_ensembles_skips_when_no_components_match():
    sa = SpearmanAnalysis(Path("."), Path("."))
    metrics = {"unrelated_metric": {"p1": 1.0, "p2": 2.0}}
    ensembles = sa._compute_ensembles(metrics)
    assert "ensemble_intervals" not in ensembles
    assert "ensemble_interval_mahal" not in ensembles
    assert "ensemble_top3" not in ensembles


def test_compute_ensembles_uses_partial_components():
    sa = SpearmanAnalysis(Path("."), Path("."))
    metrics = {"jsd_interval": {"p1": 1.0, "p2": 2.0}}
    ensembles = sa._compute_ensembles(metrics)
    assert "ensemble_intervals" in ensembles


def test_compute_ensembles_present_when_all_components_available():
    sa = SpearmanAnalysis(Path("."), Path("."))
    metrics = {
        "jsd_interval": {"p1": 1.0, "p2": 2.0},
        "interval_wasserstein": {"p1": 0.5, "p2": 1.5},
        "mahalanobis": {"p1": 0.3, "p2": 0.9},
    }
    ensembles = sa._compute_ensembles(metrics)
    assert "ensemble_intervals" in ensembles
    assert "ensemble_interval_mahal" in ensembles
    assert "ensemble_top3" in ensembles

def test_load_metrics_reads_jsd(tmp_path):
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    jsd = {"pitch_class": {"a_vs_b": 0.4}, "interval": {"a_vs_b": 0.2}}
    (sim_dir / "jsd_matrix.json").write_text(json.dumps(jsd), encoding="utf-8")

    metrics = SpearmanAnalysis(sim_dir, tmp_path)._load_metrics()
    assert "jsd_pitch_class" in metrics
    assert metrics["jsd_pitch_class"]["a_vs_b"] == pytest.approx(0.4)


def test_load_metrics_reads_mahalanobis(tmp_path):
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    mahal = {"mahalanobis_matrix": {"a_vs_b": {"mahalanobis": 1.5}}}
    (sim_dir / "mahalanobis_matrix.json").write_text(json.dumps(mahal), encoding="utf-8")

    metrics = SpearmanAnalysis(sim_dir, tmp_path)._load_metrics()
    assert "mahalanobis" in metrics
    assert metrics["mahalanobis"]["a_vs_b"] == pytest.approx(1.5)


def test_load_metrics_reads_euclidean_per_feature(tmp_path):
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    eucl = {"euclidean_matrix": {"a_vs_b": {"euclidean": 2.0, "euclidean_pitch_range": 1.0}}}
    (sim_dir / "euclidean_matrix.json").write_text(json.dumps(eucl), encoding="utf-8")

    metrics = SpearmanAnalysis(sim_dir, tmp_path)._load_metrics()
    assert "euclidean" in metrics
    assert "euclidean_pitch_range" in metrics


def test_load_fmd(tmp_path):
    sim_dir = tmp_path / "sim"
    sim_dir.mkdir()
    fmd = {"fmd_matrix": {"a_vs_b": {"fmd": 3.7}, "c_vs_d": {"fmd": None}}}
    (sim_dir / "fmd_matrix.json").write_text(json.dumps(fmd), encoding="utf-8")

    fmd_values = SpearmanAnalysis(sim_dir, tmp_path)._load_fmd()
    assert "a_vs_b" in fmd_values
    assert fmd_values["a_vs_b"] == pytest.approx(3.7)
    assert "c_vs_d" not in fmd_values
