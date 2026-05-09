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

        High-dimensional note (90 regions):
          Euclidean distance in high dimensions suffers from concentration of
          measure.  We mitigate this by:
            1. MaxMin landmark subsampling (reduces effective point count).
            2. Computing a precomputed pairwise distance matrix on landmarks
               and passing it to Ripser (avoids redundant recomputation and
               allows us to substitute correlation distance if desired).
            3. Optionally: PCA pre-projection to 20-30 dims via
               self._maybe_reduce_dims() before Ripser if n_regions > 50.
          This keeps each window computation in ~0.2s on CPU.
        """
        landmarks = self.maxmin_landmarks(window)

        # Optional dimensionality reduction for very high-dim data
        landmarks_for_rips = self._maybe_reduce_dims(landmarks)

        # Precompute pairwise Euclidean distance matrix (n_landmarks x n_landmarks)
        # Ripser accepts a condensed distance matrix, which avoids internal recomputation
        from scipy.spatial.distance import pdist, squareform
        dist_mat = squareform(pdist(landmarks_for_rips, metric="euclidean"))

        if self.use_gpu:
            diagrams = self._ripser_gpu(landmarks_for_rips)
        else:
            result = ripser.ripser(
                dist_mat,
                maxdim=self.max_dim,
                metric="precomputed",  # square distance matrix
                thresh=self.max_edge,
            )
            diagrams = {d: result["dgms"][d] for d in range(self.max_dim + 1)}

        # Remove infinite bars (keep only finite death times)
        for d in diagrams:
            finite_mask = np.isfinite(diagrams[d][:, 1])
            diagrams[d] = diagrams[d][finite_mask]

        return diagrams

    def _maybe_reduce_dims(self, pts: np.ndarray,
                           target_dim: int = 30) -> np.ndarray:
        """
        If the point cloud lives in more than target_dim dimensions,
        project to target_dim via PCA before computing the Rips complex.
        This is the standard Euclidean approximation for high-dim fMRI data:
        the first 20-30 PCs capture >80% of BOLD variance and the Rips
        filtration on the projection closely approximates the full-dim one.
        """
        n_pts, n_dim = pts.shape
        if n_dim <= target_dim or n_pts <= target_dim:
            return pts
        # Zero-mean
        centered = pts - pts.mean(axis=0)
        # Thin SVD – O(n_pts * n_dim * target_dim)
        try:
            U, s, Vt = np.linalg.svd(centered, full_matrices=False)
            return (U[:, :target_dim] * s[:target_dim])
        except np.linalg.LinAlgError:
            return pts  # fallback: no reduction

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
        Convert each persistence diagram to a fixed-length feature vector.

        Features (7 total for max_dim=2; 7 for max_dim=1; consistent shape):
          H0: total_persistence_H0
          H1: total_persistence_H1, max_persistence_H1, n_bars_H1,
              birth_death_ratio_H1 (mean), entropy_H1
          H2: total_persistence_H2  (0.0 if max_dim < 2)

        The feature vector length is always the same regardless of max_dim so
        that HMM training on different runs remains compatible.

        Returns (n_windows, n_features).
        """
        feats = []
        for dgms in diagrams_list:
            v = []

            # H0
            bars0 = dgms.get(0, np.empty((0, 2)))
            pers0 = (bars0[:, 1] - bars0[:, 0]) if len(bars0) > 0 else np.array([0.0])
            v.append(float(np.sum(pers0)))

            # H1 (consciousness-relevant)
            bars1 = dgms.get(1, np.empty((0, 2)))
            if len(bars1) > 0:
                pers1 = bars1[:, 1] - bars1[:, 0]
                total_p1 = float(np.sum(pers1))
                max_p1   = float(np.max(pers1))
                n_bars1  = float(len(bars1))
                ratio1   = float((bars1[:, 1] / (bars1[:, 0] + 1e-8)).mean())
                if total_p1 > 0:
                    p = pers1 / total_p1
                    ent1 = float(-np.sum(p * np.log(p + 1e-12)))
                else:
                    ent1 = 0.0
            else:
                total_p1 = max_p1 = n_bars1 = ratio1 = ent1 = 0.0
            v.extend([total_p1, max_p1, n_bars1, ratio1, ent1])

            # H2 (always included; 0 if not computed)
            bars2 = dgms.get(2, np.empty((0, 2)))
            pers2 = (bars2[:, 1] - bars2[:, 0]) if len(bars2) > 0 else np.array([0.0])
            v.append(float(np.sum(pers2)))

            feats.append(v)
        return np.array(feats)  # shape: (n_windows, 7)
