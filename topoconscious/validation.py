"""
Validation module: ROC/AUC evaluation on labelled datasets.

Datasets supported:
  - propofol: awake (label=1) vs. anaesthesia (label=0)
  - sleep:    REM (label=1) vs. NREM (label=0)
  - doc:      MCS (label=1) vs. UWS (label=0)

Usage:
    runner = ValidationRunner(pipeline)
    results = runner.evaluate_all(datasets_config)
    runner.plot_roc_curves(results, output_dir="results/validation")
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .pipeline import TopoConsciousPipeline
from .topology import PersistenceEngine
from .hmm import TopologicalHMM


class ValidationRunner:
    """
    Runs the TopoConscious pipeline on labelled datasets and computes AUC,
    comparing topological features against a static FC baseline.
    """

    def __init__(self, pipeline: Optional[TopoConsciousPipeline] = None,
                 output_dir: str = "results/validation"):
        self.pipeline = pipeline
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    def evaluate_dataset(
        self,
        ts_list: List[np.ndarray],
        labels: np.ndarray,
        dataset_name: str = "dataset",
    ) -> Dict:
        """
        Evaluate on a list of fMRI time series with binary labels.

        Parameters
        ----------
        ts_list : list of (n_volumes, n_regions) arrays
        labels  : binary array, 1=conscious 0=unconscious, shape (n_subjects,)
        dataset_name : str for logging

        Returns
        -------
        dict with keys: auc_topo, auc_fc, fpr_topo, tpr_topo, fpr_fc, tpr_fc
        """
        engine = PersistenceEngine(max_dim=1, n_landmarks=50, use_gpu=False)
        hmm_model = TopologicalHMM(n_states=2, n_iter=100)

        topo_scores = []
        fc_scores   = []

        for ts in ts_list:
            # --- Topological score ---
            n_vols = ts.shape[0]
            window_size, step = 30, 5
            windows = [ts[t:t+window_size] for t in range(0, n_vols - window_size + 1, step)]
            diagrams = [engine.compute(w) for w in windows]
            sig = engine.signature_vectors(diagrams)

            try:
                result = hmm_model.fit_decode(sig)
                topo_scores.append(float(np.mean(result["p_conscious"])))
            except Exception:
                # HMM can fail if data is degenerate
                topo_scores.append(float(np.mean(sig[:, 1])))  # fallback: mean H1

            # --- Static FC baseline (mean correlation coefficient) ---
            fc_mat = np.corrcoef(ts.T)
            np.fill_diagonal(fc_mat, 0)
            fc_scores.append(float(np.mean(np.abs(fc_mat))))

        topo_scores = np.array(topo_scores)
        fc_scores   = np.array(fc_scores)
        labels      = np.array(labels)

        fpr_t, tpr_t, _ = roc_curve(labels, topo_scores)
        fpr_f, tpr_f, _ = roc_curve(labels, fc_scores)
        auc_t = auc(fpr_t, tpr_t)
        auc_f = auc(fpr_f, tpr_f)

        print(f"  [{dataset_name}] AUC (TopoConscious) = {auc_t:.3f} | "
              f"AUC (static FC) = {auc_f:.3f}")

        return {
            "dataset": dataset_name,
            "auc_topo": auc_t,
            "auc_fc": auc_f,
            "fpr_topo": fpr_t,
            "tpr_topo": tpr_t,
            "fpr_fc": fpr_f,
            "tpr_fc": tpr_f,
            "topo_scores": topo_scores,
            "fc_scores": fc_scores,
            "labels": labels,
        }

    # ------------------------------------------------------------------
    def evaluate_all(
        self,
        datasets: Dict[str, Tuple[List[np.ndarray], np.ndarray]],
    ) -> Dict[str, Dict]:
        """
        Evaluate multiple labelled datasets.

        datasets: { dataset_name: (ts_list, labels) }
        """
        results = {}
        for name, (ts_list, labels) in datasets.items():
            print(f"\n[Validation] Evaluating dataset: {name}")
            results[name] = self.evaluate_dataset(ts_list, labels, name)
        self._save_summary(results)
        return results

    # ------------------------------------------------------------------
    def plot_roc_curves(self, results: Dict[str, Dict],
                        output_dir: Optional[str] = None):
        """
        Plot ROC curves for all datasets, comparing topo vs. static FC.
        """
        out = Path(output_dir or self.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        n = len(results)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), squeeze=False)

        for ax, (name, res) in zip(axes[0], results.items()):
            ax.plot(res["fpr_topo"], res["tpr_topo"],
                    color="#2196F3", lw=2,
                    label=f"TopoConscious (AUC={res['auc_topo']:.2f})")
            ax.plot(res["fpr_fc"], res["tpr_fc"],
                    color="#FF9800", lw=2, linestyle="--",
                    label=f"Static FC (AUC={res['auc_fc']:.2f})")
            ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
            ax.set_title(name, fontsize=12)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.02])

            # Shade AUC region
            ax.fill_between(res["fpr_topo"], res["tpr_topo"],
                            alpha=0.12, color="#2196F3")

        fig.suptitle("TopoConscious Validation – ROC Curves", fontsize=14, y=1.02)
        plt.tight_layout()
        path = out / "roc_curves.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [Validation] ROC curves saved -> {path}")
        return str(path)

    # ------------------------------------------------------------------
    def _save_summary(self, results: Dict[str, Dict]):
        rows = []
        for name, res in results.items():
            rows.append({
                "dataset": name,
                "auc_topoconscious": round(res["auc_topo"], 4),
                "auc_static_fc": round(res["auc_fc"], 4),
                "delta_auc": round(res["auc_topo"] - res["auc_fc"], 4),
                "meets_criterion_090": res["auc_topo"] >= 0.90,
            })
        df = pd.DataFrame(rows)
        df.to_csv(self.output_dir / "validation_summary.csv", index=False)
        print(f"\n[Validation] Summary -> {self.output_dir}/validation_summary.csv")
        print(df.to_string(index=False))
