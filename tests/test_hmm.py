"""Unit tests for TopologicalHMM."""
import numpy as np
import pytest
from topoconscious.hmm import TopologicalHMM


def test_fit_decode_shapes():
    rng = np.random.default_rng(42)
    sig = rng.random((50, 7))
    model = TopologicalHMM(n_states=2, n_iter=20)
    result = model.fit_decode(sig)
    assert len(result["p_conscious"]) == 50
    assert len(result["state_sequence"]) == 50
    assert result["p_conscious"].min() >= 0.0
    assert result["p_conscious"].max() <= 1.0


def test_conscious_state_id_valid():
    rng = np.random.default_rng(7)
    sig = rng.random((60, 7))
    model = TopologicalHMM(n_states=2, n_iter=20)
    result = model.fit_decode(sig)
    assert result["conscious_state_id"] in [0, 1]


def test_score_after_fit():
    rng = np.random.default_rng(13)
    sig_train = rng.random((60, 7))
    sig_test  = rng.random((20, 7))
    model = TopologicalHMM(n_states=2, n_iter=20)
    model.fit_decode(sig_train)
    scores = model.score(sig_test)
    assert scores.shape == (20,)
    assert (scores >= 0).all() and (scores <= 1).all()
