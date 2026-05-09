"""
Topological Transfer Entropy (TTE):
Measures directional information flow between brain regions
based on the birth-time sequences of H1 cycles.

Uses the standard Transfer Entropy estimator:
  TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})
where X and Y are topological feature sequences per region.
"""
import numpy as np
from scipy.stats import entropy as scipy_entropy
from typing import List, Dict


class TopologicalTransferEntropy:

    def __init__(self, lag: int = 1, n_bins: int = 10):
        self.lag = lag
        self.n_bins = n_bins

    def compute(self, diagrams_list: List[Dict],
                ts_matrix: np.ndarray) -> np.ndarray:
        """
        For each pair of regions (i, j), extract the H1 total persistence
        time series and compute TE(i → j).

        ts_matrix: (n_volumes, n_regions) – used to assign diagrams to regions
        Returns: (n_regions, n_regions) TE matrix.
        """
        n_regions = ts_matrix.shape[1]
        n_windows = len(diagrams_list)

        # Build per-region H1 persistence time series
        # Each window covers window_size time points of all regions together;
        # we proxy region-level topology by projecting the full diagram
        # onto per-region birth contributions via region correlation with
        # the landmark selection.
        region_ts = self._extract_region_persistence(
            diagrams_list, ts_matrix, n_regions
        )  # (n_windows, n_regions)

        te_matrix = np.zeros((n_regions, n_regions))
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j:
                    te_matrix[i, j] = self._transfer_entropy(
                        region_ts[:, i], region_ts[:, j]
                    )
        return te_matrix

    def _extract_region_persistence(self, diagrams_list, ts_matrix, n_regions):
        """
        Proxy: for each window, correlate the per-region BOLD variance with
        the total H1 persistence to weight regional contributions.
        """
        n_windows = len(diagrams_list)
        region_ts = np.zeros((n_windows, n_regions))
        step = 5   # mirrors pipeline default

        for w, dgm in enumerate(diagrams_list):
            t_start = w * step
            t_end = t_start + 30
            t_end = min(t_end, ts_matrix.shape[0])
            window_ts = ts_matrix[t_start:t_end]
            var = np.var(window_ts, axis=0)          # (n_regions,)
            bars = dgm.get(1, np.empty((0, 2)))
            total_h1 = float(np.sum(bars[:, 1] - bars[:, 0])) if len(bars) > 0 else 0.0
            total_var = var.sum() + 1e-8
            region_ts[w] = var / total_var * total_h1

        return region_ts

    def _transfer_entropy(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        Estimate TE(x → y) using histogram binning.
        TE = H(y_t | y_{t-1}) - H(y_t | y_{t-1}, x_{t-1})
        """
        lag = self.lag
        n = len(y) - lag
        if n < 10:
            return 0.0

        yt  = y[lag:]
        yt1 = y[:-lag]
        xt1 = x[:-lag]

        bins = self.n_bins
        yt_b  = np.digitize(yt,  np.linspace(yt.min(),  yt.max()  + 1e-8, bins))
        yt1_b = np.digitize(yt1, np.linspace(yt1.min(), yt1.max() + 1e-8, bins))
        xt1_b = np.digitize(xt1, np.linspace(xt1.min(), xt1.max() + 1e-8, bins))

        # H(yt | yt1)
        h_yt_given_yt1 = self._cond_entropy(yt_b, yt1_b, bins)
        # H(yt | yt1, xt1)
        joint_idx = yt1_b * bins + xt1_b
        h_yt_given_joint = self._cond_entropy(yt_b, joint_idx, bins * bins)

        return max(0.0, h_yt_given_yt1 - h_yt_given_joint)

    @staticmethod
    def _cond_entropy(y: np.ndarray, cond: np.ndarray, n_cond: int) -> float:
        total = len(y)
        cond_entropy = 0.0
        for c in np.unique(cond):
            mask = cond == c
            p_c = mask.sum() / total
            vals, counts = np.unique(y[mask], return_counts=True)
            probs = counts / counts.sum()
            cond_entropy += p_c * (-np.sum(probs * np.log2(probs + 1e-12)))
        return cond_entropy
