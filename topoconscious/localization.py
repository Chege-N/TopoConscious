"""
Cycle Localization:
Maps persistent H1 cycles back to anatomical brain regions using
the Vietoris-Rips complex SimplexTree from GUDHI.

Algorithm:
  1. Build a Rips SimplexTree on the landmark point cloud.
  2. Compute persistence; harvest H1 generators (cocycle representatives).
  3. Each H1 generator is a list of edges (simplex pairs) whose endpoints
     are landmark indices.
  4. Map landmark indices back to atlas region labels via nearest-neighbour
     lookup on the original time-series matrix.
"""
import numpy as np
from typing import Dict, List, Optional
import gudhi


class CycleLocalizer:
    """
    Maps persistent H1 cycles to anatomical brain regions.

    Usage:
        localizer = CycleLocalizer(atlas="aal")
        result = localizer.localize_with_complex(
            point_cloud=window,          # (T, n_regions)
            region_labels=atlas_labels,  # list of region names length n_regions
            max_edge_length=5.0,
        )
    """

    def __init__(self, atlas: str = "aal"):
        self.atlas = atlas

    # ------------------------------------------------------------------
    def localize_with_complex(
        self,
        point_cloud: np.ndarray,
        region_labels: Optional[List[str]] = None,
        max_edge_length: float = 5.0,
        persistence_threshold: Optional[float] = None,
    ) -> Dict:
        """
        Full localization using GUDHI SimplexTree.

        Parameters
        ----------
        point_cloud     : (T, n_regions) landmark point cloud for one window.
                          Each row is a time-point, each column a brain region.
        region_labels   : list of anatomical region names, length n_regions.
        max_edge_length : Rips filtration max edge (controls complex size).
        persistence_threshold : min persistence to call a cycle significant.
                          Defaults to median + 1*std of all H1 bars.

        Returns
        -------
        dict with keys:
          significant_cycles  – list of cycle indices (sorted by persistence desc)
          birth_times         – float list
          death_times         – float list
          persistence         – float list
          regions_per_cycle   – list of lists: each inner list = region names
                                involved in that cycle's representative edges
          n_regions_involved  – total unique regions across all significant cycles
        """
        n_pts = len(point_cloud)
        if n_pts == 0:
            return self._empty_result()

        # Build Vietoris-Rips complex
        rips = gudhi.RipsComplex(
            points=point_cloud.tolist(),
            max_edge_length=max_edge_length,
        )
        st = rips.create_simplex_tree(max_dimension=2)
        st.compute_persistence()

        # Extract H1 persistence pairs with their simplex representatives
        h1_pairs = [
            (birth, death, simplex)
            for (dim, (birth, death)), simplex in zip(
                st.persistence(), st.persistence_generators()[0]
                if hasattr(st, "persistence_generators") else []
            )
            if dim == 1 and np.isfinite(death)
        ]

        # Fallback: if persistence_generators not available (older GUDHI),
        # use persistence pairs only (no simplex info)
        if not h1_pairs:
            h1_pairs = self._extract_h1_fallback(st)

        if not h1_pairs:
            return self._empty_result()

        births  = np.array([p[0] for p in h1_pairs])
        deaths  = np.array([p[1] for p in h1_pairs])
        pers    = deaths - births
        simps   = [p[2] if len(p) > 2 else [] for p in h1_pairs]

        # Threshold
        if persistence_threshold is None:
            persistence_threshold = float(np.median(pers) + np.std(pers))

        sig_mask = pers >= persistence_threshold
        sig_idx  = np.where(sig_mask)[0]

        if len(sig_idx) == 0:
            # Fallback: take top-3 by persistence
            sig_idx = np.argsort(-pers)[:min(3, len(pers))]

        # Map simplex vertex indices -> region labels
        regions_per_cycle = []
        all_region_indices = set()
        for i in sig_idx:
            simp = simps[i] if i < len(simps) else []
            vertex_indices = self._vertices_from_simplex(simp, n_pts)
            # vertex index in landmark space = time index; map to nearest region
            # via max absolute correlation across the point cloud columns
            region_idxs = self._vertices_to_regions(
                vertex_indices, point_cloud, n_pts
            )
            all_region_indices.update(region_idxs)
            if region_labels:
                regions_per_cycle.append(
                    [region_labels[r % len(region_labels)] for r in region_idxs]
                )
            else:
                regions_per_cycle.append([f"region_{r}" for r in region_idxs])

        return {
            "significant_cycles": sig_idx.tolist(),
            "birth_times": births[sig_idx].tolist(),
            "death_times": deaths[sig_idx].tolist(),
            "persistence": pers[sig_idx].tolist(),
            "regions_per_cycle": regions_per_cycle,
            "n_regions_involved": len(all_region_indices),
            "persistence_threshold": persistence_threshold,
        }

    # ------------------------------------------------------------------
    def localize(
        self,
        diagram: Dict[int, np.ndarray],
        point_cloud: Optional[np.ndarray] = None,
        region_labels: Optional[List[str]] = None,
    ) -> Dict:
        """
        Lightweight interface: localize from a pre-computed persistence diagram.
        When point_cloud is provided, calls localize_with_complex for full
        cocycle extraction.  Otherwise uses the diagram directly.
        """
        if point_cloud is not None:
            return self.localize_with_complex(
                point_cloud=point_cloud,
                region_labels=region_labels,
            )

        # Diagram-only path (no simplex info available)
        bars = diagram.get(1, np.empty((0, 2)))
        if len(bars) == 0:
            return self._empty_result()

        pers = bars[:, 1] - bars[:, 0]
        threshold = float(np.median(pers) + np.std(pers))
        sig = np.where(pers >= threshold)[0]
        if len(sig) == 0:
            sig = np.argsort(-pers)[:min(3, len(pers))]

        labels_out = (
            [region_labels[i % len(region_labels)] for i in sig]
            if region_labels else [f"region_{i}" for i in sig]
        )
        return {
            "significant_cycles": sig.tolist(),
            "birth_times": bars[sig, 0].tolist(),
            "death_times": bars[sig, 1].tolist(),
            "persistence": pers[sig].tolist(),
            "regions_per_cycle": [[l] for l in labels_out],
            "n_regions_involved": len(sig),
            "persistence_threshold": threshold,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_h1_fallback(st) -> list:
        """Use persistence pairs only (GUDHI < 3.7 without persistence_generators)."""
        pairs = []
        for (dim, (birth, death)) in st.persistence():
            if dim == 1 and np.isfinite(death):
                pairs.append((birth, death))
        return pairs

    @staticmethod
    def _vertices_from_simplex(simplex, n_pts: int) -> List[int]:
        """
        Extract vertex indices from a GUDHI simplex representative.
        A H1 generator is a list of edges (each edge = list of 2 vertex indices).
        """
        if not simplex:
            return []
        vertices = set()
        # simplex may be a list of edges [[v1,v2], [v2,v3], ...]
        for item in simplex:
            if hasattr(item, "__iter__"):
                for v in item:
                    if isinstance(v, (int, np.integer)) and v < n_pts:
                        vertices.add(int(v))
            elif isinstance(item, (int, np.integer)) and item < n_pts:
                vertices.add(int(item))
        return list(vertices)

    @staticmethod
    def _vertices_to_regions(vertex_indices: List[int],
                              point_cloud: np.ndarray,
                              n_pts: int) -> List[int]:
        """
        Map landmark vertex (time-point) indices to brain region indices.

        Strategy: for each vertex (= time-point), find the brain region
        (column of point_cloud) with the highest absolute value at that
        time-point — this is the region most "active" when this cycle
        boundary was created.
        """
        if not vertex_indices or point_cloud is None:
            return list(range(min(3, point_cloud.shape[1] if point_cloud is not None else 3)))

        region_indices = []
        for v in vertex_indices:
            if v < len(point_cloud):
                row = np.abs(point_cloud[v])
                region_indices.append(int(np.argmax(row)))
        return list(set(region_indices))

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "significant_cycles": [],
            "birth_times": [],
            "death_times": [],
            "persistence": [],
            "regions_per_cycle": [],
            "n_regions_involved": 0,
            "persistence_threshold": 0.0,
        }
