"""
Persistent homology engine using GUDHI and Ripser.
Includes:
  - MaxMin landmark selection
  - Vietoris-Rips filtration
  - Persistence diagram computation (H0, H1, H2)
  - Wasserstein distance timeline
  - Topological signature vectors
"""
import numpy as np
from typing import List, Dict
import gudhi
from gudhi.wasserstein import wasserstein_distance
import ripser


class PersistenceEngine:
    def __init__(self, max_dim: int = 2, n_landmarks: int = 200,
                 use_gpu: bool = False, max_edge_length: float = 5.0):
        self.max_dim = max_dim
        self.n_landmarks = n_landmarks
        self.use_gpu = use_gpu
        self.max_edge = max_edge_length

    # ------------------------------------------------------------------
    def maxmin_landmarks(self, point_cloud: np.ndarray) -> np.ndarray:
        """
        MaxMin (farthest-point) sampling to select n_landmarks points.
        O(n * k) complexity.
        """
        n = len(point_cloud)
        k = min(self.n_landmarks, n)
        selected = [np.random.randint(n)]
        dists = np.full(n, np.inf)

        for _ in range(k - 1):
            d = np.linalg.norm(point_cloud - point_cloud[selected[-1]], axis=1)
            dists = np.minimum(dists, d)
            selected.append(int(np.argmax(dists)))

        return point_cloud[selected]

    # ------------------------------------------------------------------
    def compute(self, window: np.ndarray) -> Dict[int, np.ndarray]:
        """
        window: (T, n_regions) – treat each time-point as a point in R^n_regions.
        Returns dict {dim: array of (birth, death) pairs}.
        """
        landmarks = self.maxmin_landmarks(window)

        if self.use_gpu:
            diagrams = self._ripser_gpu(landmarks)
        else:
            result = ripser.ripser(landmarks, maxdim=self.max_dim,
                                   metric="euclidean")
            diagrams = {d: result["dgms"][d] for d in range(self.max_dim + 1)}

        # Remove infinite bars for H0 (keep only finite)
        for d in diagrams:
            finite_mask = np.isfinite(diagrams[d][:, 1])
            diagrams[d] = diagrams[d][finite_mask]

        return diagrams

    def _ripser_gpu(self, pts: np.ndarray) -> Dict[int, np.ndarray]:
        """GPU path via cupy (falls back to CPU if unavailable)."""
        try:
            import cupy as cp
            pts_gpu = cp.asarray(pts)
            result = ripser.ripser(cp.asnumpy(pts_gpu), maxdim=self.max_dim)
        except ImportError:
            result = ripser.ripser(pts, maxdim=self.max_dim)
        return {d: result["dgms"][d] for d in range(self.max_dim + 1)}

    # ------------------------------------------------------------------
    def wasserstein_timeline(self, diagrams_list: List[Dict]) -> np.ndarray:
        """
        Compute Wasserstein-2 distance between consecutive persistence
        diagrams (H1 only – the consciousness-relevant dimension).
        Returns array of length (len(diagrams_list) - 1).
        """
        distances = []
        for i in range(len(diagrams_list) - 1):
            d1 = diagrams_list[i].get(1, np.empty((0, 2)))
            d2 = diagrams_list[i + 1].get(1, np.empty((0, 2)))
            try:
                w = wasserstein_distance(d1, d2, order=2, internal_p=2)
            except Exception:
                w = 0.0
            distances.append(w)
        return np.array(distances)

    # ------------------------------------------------------------------
    def signature_vectors(self, diagrams_list: List[Dict]) -> np.ndarray:
        """
        Convert each persistence diagram to a fixed-length feature vector:
          - total_persistence_H0
          - total_persistence_H1
          - total_persistence_H2
          - max_persistence_H1
          - n_bars_H1
          - birth_death_ratio_H1 (mean)
          - entropy_H1
        Returns (n_windows, n_features).
        """
        feats = []
        for dgms in diagrams_list:
            v = []
            for d in range(self.max_dim + 1):
                bars = dgms.get(d, np.empty((0, 2)))
                pers = bars[:, 1] - bars[:, 0] if len(bars) > 0 else np.array([0.0])
                v.append(float(np.sum(pers)))            # total persistence
                if d == 1:
                    v.append(float(np.max(pers)) if len(pers) > 0 else 0.0)
                    v.append(float(len(bars)))
                    ratio = (bars[:, 1] / (bars[:, 0] + 1e-8)).mean() if len(bars) > 0 else 0.0
                    v.append(float(ratio))
                    # Persistent entropy
                    if np.sum(pers) > 0:
                        p = pers / np.sum(pers)
                        ent = -np.sum(p * np.log(p + 1e-12))
                    else:
                        ent = 0.0
                    v.append(ent)
            feats.append(v)
        return np.array(feats)
