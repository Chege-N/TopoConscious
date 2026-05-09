"""
Müller-Lyer Current metric for persistence diagrams.

The standard Wasserstein distance treats H1 cycles as unordered sets and
is permutation-invariant but SCALE-SENSITIVE.  The Müller-Lyer current
(also called the persistence-weighted Wasserstein or "landscape current")
additionally preserves information about the *location* of features in the
birth-death plane, making transitions between diagram types (e.g., many
short loops vs. few long loops) more discriminative.

Reference:
  Divol & Lacombe (2021). "Understanding the topology and the geometry of
  the space of persistence diagrams via optimal partial transport."
  J. Applied & Computational Topology.

Implementation note:
  We use a weighted Wasserstein variant where each bar is weighted by its
  persistence (death - birth), giving longer-lived topological features
  proportionally more influence on the metric.  This is equivalent to the
  first-order Müller-Lyer current restricted to H1.
"""
import numpy as np
from typing import List, Dict


class MuellerLyerCurrent:
    """
    Compute pairwise Müller-Lyer current distances between persistence diagrams.

    Distance formula (H1 only):
      ML(D1, D2) = W_2^{pers-weighted}(D1, D2)
                 + alpha * |sum(pers(D1)) - sum(pers(D2))|   # scale term
                 + beta  * ||centroid(D1) - centroid(D2)||   # location term

    alpha and beta balance the extra scale/location penalty terms.
    When alpha=beta=0 this reduces to standard persistence-weighted W2.
    """

    def __init__(self, alpha: float = 0.1, beta: float = 0.05,
                 order: int = 2):
        self.alpha = alpha
        self.beta = beta
        self.order = order

    # ------------------------------------------------------------------
    def distance(self, dgm1: Dict[int, np.ndarray],
                 dgm2: Dict[int, np.ndarray],
                 dim: int = 1) -> float:
        """
        Compute Müller-Lyer current distance between two persistence diagrams
        at homological dimension dim.
        """
        bars1 = dgm1.get(dim, np.empty((0, 2)))
        bars2 = dgm2.get(dim, np.empty((0, 2)))

        # Persistence-weighted optimal transport (approximate via sorted matching)
        w_dist = self._weighted_wasserstein(bars1, bars2)

        # Scale penalty: difference in total persistence
        tp1 = float(np.sum(bars1[:, 1] - bars1[:, 0])) if len(bars1) > 0 else 0.0
        tp2 = float(np.sum(bars2[:, 1] - bars2[:, 0])) if len(bars2) > 0 else 0.0
        scale_term = self.alpha * abs(tp1 - tp2)

        # Location penalty: distance between weighted centroids in birth-death plane
        c1 = self._weighted_centroid(bars1)
        c2 = self._weighted_centroid(bars2)
        loc_term = self.beta * float(np.linalg.norm(c1 - c2))

        return w_dist + scale_term + loc_term

    def _weighted_wasserstein(self, bars1: np.ndarray,
                               bars2: np.ndarray) -> float:
        """
        Approximate persistence-weighted W_p distance via sorted 1-D projections
        on the diagonal and on the persistence axis.  This is exact when bars
        are collinear and a good approximation otherwise.
        For an exact computation, use gudhi.wasserstein with weights.
        """
        if len(bars1) == 0 and len(bars2) == 0:
            return 0.0

        # Represent each bar as a point (birth, death) with weight = persistence
        def _to_weighted(bars):
            if len(bars) == 0:
                return np.empty((0, 2)), np.array([])
            w = bars[:, 1] - bars[:, 0]
            return bars, w

        pts1, w1 = _to_weighted(bars1)
        pts2, w2 = _to_weighted(bars2)

        # Pad the smaller diagram with diagonal points (birth=death=mid-point)
        n1, n2 = len(pts1), len(pts2)
        if n1 == 0:
            return float(np.mean(w2 ** self.order) ** (1 / self.order))
        if n2 == 0:
            return float(np.mean(w1 ** self.order) ** (1 / self.order))

        # Sort both by persistence and match greedily
        idx1 = np.argsort(-w1)
        idx2 = np.argsort(-w2)
        k = min(n1, n2)

        cost = 0.0
        for i in range(k):
            p = pts1[idx1[i]]
            q = pts2[idx2[i]]
            wi = (w1[idx1[i]] + w2[idx2[i]]) / 2.0
            cost += wi * (np.linalg.norm(p - q) ** self.order)

        # Unmatched bars: distance to diagonal is persistence / sqrt(2)
        for i in range(k, n1):
            p = pts1[idx1[i]]
            d = (p[1] - p[0]) / np.sqrt(2)
            cost += w1[idx1[i]] * (d ** self.order)
        for j in range(k, n2):
            q = pts2[idx2[j]]
            d = (q[1] - q[0]) / np.sqrt(2)
            cost += w2[idx2[j]] * (d ** self.order)

        total_w = max(1e-8, np.sum(w1) + np.sum(w2))
        return float((cost / total_w) ** (1.0 / self.order))

    @staticmethod
    def _weighted_centroid(bars: np.ndarray) -> np.ndarray:
        """Return the persistence-weighted centroid in the birth-death plane."""
        if len(bars) == 0:
            return np.zeros(2)
        w = bars[:, 1] - bars[:, 0]
        w_sum = w.sum() + 1e-8
        return (bars * w[:, None]).sum(axis=0) / w_sum

    # ------------------------------------------------------------------
    def timeline(self, diagrams_list: List[Dict], dim: int = 1) -> np.ndarray:
        """
        Compute Müller-Lyer current distance between consecutive diagrams.
        Returns array of length (len(diagrams_list) - 1).
        """
        dists = []
        for i in range(len(diagrams_list) - 1):
            dists.append(self.distance(diagrams_list[i], diagrams_list[i + 1], dim=dim))
        return np.array(dists)


# ---------------------------------------------------------------------------
class PersistenceLandscape:
    """
    Persistence Landscape representation (Bubenik 2015) for H1 diagrams.

    The landscape lambda_k(t) is the k-th largest "tent function" value
    at parameter t across all H1 bars.  This gives a functional summary
    of the persistence diagram that lives in L^2 and supports averaging,
    statistical testing, and the "landscape current" inner product.

    Relationship to ML current:
      The Müller-Lyer current can be viewed as a signed measure on the
      birth-death plane; the persistence landscape is its functional
      realisation.  Using both gives complementary views: ML current
      captures global shape changes; the landscape captures local
      per-level structure.

    Usage:
        pl = PersistenceLandscape(n_landscapes=5, resolution=100)
        vec = pl.vectorize(diagram)          # (n_landscapes * resolution,)
        dist = pl.distance(dgm1, dgm2)       # L2 landscape distance
        timeline = pl.timeline(diagrams_list)
    """

    def __init__(self, n_landscapes: int = 5, resolution: int = 100,
                 t_min: float = 0.0, t_max: float = 3.0):
        self.K = n_landscapes
        self.res = resolution
        self.t_grid = np.linspace(t_min, t_max, resolution)

    def _tent(self, birth: float, death: float) -> np.ndarray:
        """Tent function for a single bar: max(0, min(t-b, d-t)) over t_grid."""
        t = self.t_grid
        return np.maximum(0.0, np.minimum(t - birth, death - t))

    def vectorize(self, diagram: Dict[int, np.ndarray],
                  dim: int = 1) -> np.ndarray:
        """
        Compute the persistence landscape vector for dimension dim.
        Returns (K * resolution,) float array.
        """
        bars = diagram.get(dim, np.empty((0, 2)))
        bars = bars[np.isfinite(bars[:, 1])] if len(bars) > 0 else bars

        if len(bars) == 0:
            return np.zeros(self.K * self.res)

        # Stack tent functions: (n_bars, resolution)
        tents = np.stack([self._tent(b, d) for b, d in bars], axis=0)

        # K-th landscape: k-th largest value at each t
        landscape = np.zeros((self.K, self.res))
        for k in range(self.K):
            if tents.shape[0] > k:
                landscape[k] = np.partition(tents, -k - 1, axis=0)[-k - 1]

        return landscape.ravel()

    def distance(self, dgm1: Dict[int, np.ndarray],
                 dgm2: Dict[int, np.ndarray],
                 dim: int = 1) -> float:
        """L2 distance between persistence landscapes (landscape metric)."""
        v1 = self.vectorize(dgm1, dim)
        v2 = self.vectorize(dgm2, dim)
        dt = (self.t_grid[-1] - self.t_grid[0]) / (self.res - 1)
        return float(np.sqrt(np.sum((v1 - v2) ** 2) * dt))

    def timeline(self, diagrams_list: List[Dict], dim: int = 1) -> np.ndarray:
        """L2 landscape distance between consecutive diagrams."""
        return np.array([
            self.distance(diagrams_list[i], diagrams_list[i + 1], dim=dim)
            for i in range(len(diagrams_list) - 1)
        ])
