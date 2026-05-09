"""Unit tests for TopologicalTransferEntropy."""
import numpy as np
from topoconscious.transfer_entropy import TopologicalTransferEntropy


def _dummy_diagrams(n_windows, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_windows):
        n_bars = rng.integers(3, 15)
        births = rng.uniform(0, 1, n_bars)
        deaths = births + rng.uniform(0.05, 0.5, n_bars)
        out.append({0: np.column_stack([births, deaths]),
                    1: np.column_stack([births, deaths])})
    return out


def test_te_matrix_shape():
    n_regions = 5
    ts = np.random.default_rng(0).random((50, n_regions))
    dgms = _dummy_diagrams(10)
    calc = TopologicalTransferEntropy(lag=1, n_bins=5)
    mat = calc.compute(dgms, ts)
    assert mat.shape == (n_regions, n_regions)


def test_te_diagonal_zero():
    n_regions = 4
    ts = np.random.default_rng(1).random((50, n_regions))
    dgms = _dummy_diagrams(10)
    calc = TopologicalTransferEntropy(lag=1, n_bins=5)
    mat = calc.compute(dgms, ts)
    assert np.allclose(np.diag(mat), 0.0)


def test_te_nonnegative():
    n_regions = 4
    ts = np.random.default_rng(2).random((50, n_regions))
    dgms = _dummy_diagrams(10)
    calc = TopologicalTransferEntropy(lag=1, n_bins=5)
    mat = calc.compute(dgms, ts)
    assert (mat >= 0).all()
