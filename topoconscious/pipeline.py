"""
Main orchestration pipeline for TopoConscious.
"""
import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from pathlib import Path
from typing import Optional

from .preprocessing import Preprocessor
from .topology import PersistenceEngine
from .hmm import TopologicalHMM
from .transfer_entropy import TopologicalTransferEntropy
from .localization import CycleLocalizer
from .visualization import TopoVisualizer


class TopoConsciousPipeline:
    """
    End-to-end pipeline:
      fMRI NIfTI (BIDS) → preprocessing → sliding-window point clouds
      → persistent homology (H0/H1/H2) → Wasserstein timeline
      → HMM consciousness probability → cycle localization → report
    """

    def __init__(
        self,
        bids_dir: str,
        output_dir: str,
        window_size: int = 30,
        step: int = 5,
        n_landmarks: int = 200,
        max_homology_dim: int = 2,
        tr: float = 2.0,
        use_gpu: bool = False,
        atlas: str = "aal",
    ):
        self.bids_dir = Path(bids_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        self.step = step
        self.n_landmarks = n_landmarks
        self.max_dim = max_homology_dim
        self.tr = tr
        self.use_gpu = use_gpu
        self.atlas = atlas

        self.preprocessor = Preprocessor(atlas=atlas, tr=tr)
        self.topo_engine = PersistenceEngine(
            max_dim=max_homology_dim, n_landmarks=n_landmarks, use_gpu=use_gpu
        )
        self.hmm = TopologicalHMM(n_states=2)
        self.te_calc = TopologicalTransferEntropy()
        self.localizer = CycleLocalizer(atlas=atlas)
        self.visualizer = TopoVisualizer(output_dir=str(self.output_dir))

    # ------------------------------------------------------------------
    def run(self, subject_ids: Optional[list] = None):
        """Run the full pipeline for all (or specified) subjects."""
        subjects = subject_ids or self._discover_subjects()
        all_results = {}

        for sub in subjects:
            print(f"\n[TopoConscious] Processing subject: {sub}")
            result = self._process_subject(sub)
            all_results[sub] = result
            self._save_subject_results(sub, result)

        self._aggregate_and_report(all_results)
        return all_results

    # ------------------------------------------------------------------
    def _process_subject(self, subject_id: str) -> dict:
        # 1. Load & preprocess
        bold_path = self._find_bold(subject_id)
        ts_matrix = self.preprocessor.load_and_extract(bold_path)
        # ts_matrix: shape (n_volumes, n_regions) e.g. (300, 90)

        # 2. Sliding-window point clouds
        windows = self._sliding_windows(ts_matrix)

        # 3. Persistent homology per window
        diagrams = [self.topo_engine.compute(w) for w in windows]

        # 4. Wasserstein distance timeline
        wass_timeline = self.topo_engine.wasserstein_timeline(diagrams)

        # 5. Topological signature vectors for HMM
        sig_vectors = self.topo_engine.signature_vectors(diagrams)

        # 6. HMM decoding → consciousness probability
        hmm_result = self.hmm.fit_decode(sig_vectors)

        # 7. Transfer entropy between regional PH features
        te_matrix = self.te_calc.compute(diagrams, ts_matrix)

        # 8. Cycle localization for each transition
        transitions = self._find_transitions(hmm_result["state_sequence"])
        localization = {}
        for t in transitions:
            localization[t] = self.localizer.localize(diagrams[t])

        # 9. Visualize
        self.visualizer.plot_consciousness_timeline(
            hmm_result, wass_timeline, subject_id
        )
        self.visualizer.plot_te_matrix(te_matrix, subject_id)

        return {
            "subject": subject_id,
            "n_windows": len(windows),
            "diagrams": diagrams,
            "wasserstein_timeline": wass_timeline,
            "hmm": hmm_result,
            "te_matrix": te_matrix,
            "transitions": transitions,
            "localization": localization,
        }

    # ------------------------------------------------------------------
    def _sliding_windows(self, ts: np.ndarray) -> list:
        """Return list of (window_size, n_regions) arrays."""
        n_vols = ts.shape[0]
        windows = []
        t = 0
        while t + self.window_size <= n_vols:
            windows.append(ts[t : t + self.window_size])
            t += self.step
        return windows

    def _find_transitions(self, states: np.ndarray) -> list:
        transitions = []
        for i in range(1, len(states)):
            if states[i] != states[i - 1]:
                transitions.append(i)
        return transitions

    def _discover_subjects(self) -> list:
        return sorted(
            [d.name for d in self.bids_dir.iterdir() if d.name.startswith("sub-")]
        )

    def _find_bold(self, subject_id: str) -> Path:
        pattern = list(
            self.bids_dir.glob(f"{subject_id}/func/*_bold.nii*")
        )
        if not pattern:
            raise FileNotFoundError(f"No BOLD file for {subject_id}")
        return pattern[0]

    def _save_subject_results(self, subject_id: str, result: dict):
        out = self.output_dir / subject_id
        out.mkdir(exist_ok=True)
        np.save(out / "wasserstein_timeline.npy", result["wasserstein_timeline"])
        np.save(out / "te_matrix.npy", result["te_matrix"])
        pd.DataFrame(
            {
                "window": range(len(result["hmm"]["p_conscious"])),
                "p_conscious": result["hmm"]["p_conscious"],
                "state": result["hmm"]["state_sequence"],
            }
        ).to_csv(out / "consciousness_probability.csv", index=False)
        print(f"  Saved results → {out}")

    def _aggregate_and_report(self, all_results: dict):
        summary_rows = []
        for sub, res in all_results.items():
            mean_p = float(np.mean(res["hmm"]["p_conscious"]))
            summary_rows.append({"subject": sub, "mean_p_conscious": mean_p})
        df = pd.DataFrame(summary_rows)
        df.to_csv(self.output_dir / "group_summary.csv", index=False)
        print(f"\n[TopoConscious] Group summary saved to {self.output_dir}/group_summary.csv")


def cli_entry():
    parser = argparse.ArgumentParser(description="TopoConscious pipeline CLI")
    parser.add_argument("--bids-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--window-size", type=int, default=30)
    parser.add_argument("--step", type=int, default=5)
    parser.add_argument("--landmarks", type=int, default=200)
    parser.add_argument("--atlas", type=str, default="aal", choices=["aal", "schaefer100", "destrieux"])
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    pipe = TopoConsciousPipeline(
        bids_dir=args.bids_dir,
        output_dir=args.output_dir,
        window_size=args.window_size,
        step=args.step,
        n_landmarks=args.landmarks,
        atlas=args.atlas,
        use_gpu=args.gpu,
    )
    pipe.run()
