"""Unit tests for PersistenceEngine."""
import numpy as np
import pytest
from topoconscious.topology import PersistenceEngine


@pytest.fixture
def engine():
    return PersistenceEngine(max_dim=1, n_landmarks=20, use_gpu=False)


def test_maxmin_landmarks_shape(engine):
    rng = np.random.default_rng(0)
    cloud = rng.random((100, 10))
    lm = engine.maxmin_landmarks(cloud)
    assert lm.shape == (20, 10)


def test_maxmin_landmarks_fewer_points(engine):
    rng = np.random.default_rng(0)
    cloud = rng.random((15, 10))
    lm = engine.maxmin_landmarks(cloud)
    assert lm.shape[0] == 15  # capped at n


def test_compute_returns_dicts(engine):
    rng = np.random.default_rng(1)
    window = rng.random((30, 10))
    dgms = engine.compute(window)
    assert isinstance(dgms, dict)
    for d in range(2):
        assert d in dgms
        assert dgms[d].ndim == 2
        assert dgms[d].shape[1] == 2


def test_wasserstein_timeline_length(engine):
    rng = np.random.default_rng(2)
    dgms = [engine.compute(rng.random((30, 10))) for _ in range(5)]
    wt = engine.wasserstein_timeline(dgms)
    assert len(wt) == 4


def test_wasserstein_timeline_nonnegative(engine):
    rng = np.random.default_rng(99)
    dgms = [engine.compute(rng.random((30, 10))) for _ in range(4)]
    wt = engine.wasserstein_timeline(dgms)
    assert (wt >= 0).all()


def test_signature_vectors_shape(engine):
    rng = np.random.default_rng(3)
    dgms = [engine.compute(rng.random((30, 10))) for _ in range(8)]
    sv = engine.signature_vectors(dgms)
    assert sv.shape[0] == 8
    assert sv.shape[1] > 0
