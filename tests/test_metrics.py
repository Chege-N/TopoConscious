"""Unit tests for MuellerLyerCurrent."""
import numpy as np
import pytest
from topoconscious.metrics import MuellerLyerCurrent


def _make_diagram(n_bars=10, seed=0, persistence_scale=0.5):
    rng = np.random.default_rng(seed)
    births = rng.uniform(0, 1, n_bars)
    deaths = births + rng.uniform(0.01, persistence_scale, n_bars)
    return {1: np.column_stack([births, deaths])}


def test_self_distance_zero():
    """
    Self-distance must be 0.  When both inputs are the SAME diagram,
    sorted matching pairs each bar with itself -> cost=0, scale term=0,
    location term=0.
    """
    ml = MuellerLyerCurrent()
    dgm = _make_diagram(seed=0)
    # Use same object (tests reference path)
    d_ref = ml.distance(dgm, dgm, dim=1)
    assert d_ref == pytest.approx(0.0, abs=1e-8), f"Same-object self-distance={d_ref}"
    # Also test with identical copy (tests value path)
    import copy
    dgm2 = copy.deepcopy(dgm)
    d_copy = ml.distance(dgm, dgm2, dim=1)
    assert d_copy == pytest.approx(0.0, abs=1e-6), f"Deep-copy self-distance={d_copy}"


def test_empty_diagrams():
    ml = MuellerLyerCurrent()
    empty = {1: np.empty((0, 2))}
    assert ml.distance(empty, empty, dim=1) == 0.0


def test_distance_nonnegative():
    ml = MuellerLyerCurrent()
    d1 = _make_diagram(seed=1)
    d2 = _make_diagram(seed=2)
    assert ml.distance(d1, d2, dim=1) >= 0.0


def test_distance_increases_with_scale_difference():
    """More persistent diagram should have larger ML distance from baseline."""
    ml = MuellerLyerCurrent(alpha=0.5, beta=0.1)
    base = _make_diagram(persistence_scale=0.1, seed=5)
    large = _make_diagram(persistence_scale=2.0, seed=5)  # same births
    small = _make_diagram(persistence_scale=0.05, seed=5)
    d_large = ml.distance(base, large, dim=1)
    d_small = ml.distance(base, small, dim=1)
    assert d_large > d_small


def test_timeline_length():
    ml = MuellerLyerCurrent()
    diagrams = [_make_diagram(seed=i) for i in range(7)]
    tl = ml.timeline(diagrams, dim=1)
    assert len(tl) == 6
    assert (tl >= 0).all()


def test_validation_runner_synthetic():
    """Integration smoke-test: ValidationRunner with synthetic data."""
    from topoconscious.validation import ValidationRunner
    rng = np.random.default_rng(42)
    n_subj = 6
    # Half conscious (label=1), half unconscious (label=0)
    ts_conscious = [rng.standard_normal((60, 10)) + np.tile(rng.uniform(0.3,0.8,10),(60,1))
                    for _ in range(n_subj // 2)]
    ts_unconscious = [rng.standard_normal((60, 10)) * 0.5
                      for _ in range(n_subj // 2)]
    ts_list = ts_conscious + ts_unconscious
    labels = np.array([1]*(n_subj//2) + [0]*(n_subj//2))

    runner = ValidationRunner(output_dir="/tmp/topo_test_validation")
    res = runner.evaluate_dataset(ts_list, labels, dataset_name="synthetic")
    assert 0 <= res["auc_topo"] <= 1.0
    assert 0 <= res["auc_fc"]   <= 1.0
