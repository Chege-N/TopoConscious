"""
Shared pytest fixtures for TopoConscious tests.
"""
import numpy as np
import pytest
from topoconscious.topology import PersistenceEngine


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="session")
def small_engine():
    """Lightweight engine for fast unit tests."""
    return PersistenceEngine(max_dim=1, n_landmarks=20, use_gpu=False)


@pytest.fixture(scope="session")
def synthetic_ts(rng):
    """300-volume, 10-region synthetic fMRI time series."""
    ts = rng.standard_normal((300, 10))
    # Inject correlated activity to simulate a conscious epoch (vols 0-150)
    for i in range(5):
        ts[:150, i] += 0.4 * ts[:150, 0]
    return ts


@pytest.fixture(scope="session")
def small_diagrams(small_engine, rng):
    """Pre-computed persistence diagrams for 8 windows (30 TRs, 10 regions)."""
    return [small_engine.compute(rng.random((30, 10))) for _ in range(8)]


def make_dummy_diagrams(n_windows: int = 10, seed: int = 0):
    """Helper: create dummy {0,1} persistence diagrams."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_windows):
        n_bars = rng.integers(3, 15)
        births = rng.uniform(0, 1, n_bars)
        deaths = births + rng.uniform(0.05, 0.5, n_bars)
        out.append({0: np.column_stack([births, deaths]),
                    1: np.column_stack([births, deaths])})
    return out
