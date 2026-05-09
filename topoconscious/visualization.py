"""
Visualization module:
  - Consciousness probability time course plot
  - Transfer entropy heatmap
  - Interactive Jupyter widget for PD exploration
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from typing import Dict


class TopoVisualizer:

    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------
    def plot_consciousness_timeline(self, hmm_result: Dict,
                                     wass_timeline: np.ndarray,
                                     subject_id: str):
        fig = plt.figure(figsize=(14, 6))
        gs = gridspec.GridSpec(2, 1, hspace=0.4)

        ax1 = fig.add_subplot(gs[0])
        t = np.arange(len(hmm_result["p_conscious"]))
        ax1.fill_between(t, 0, hmm_result["p_conscious"],
                         alpha=0.7, color="#2196F3", label="P(conscious)")
        ax1.axhline(0.5, color="red", linestyle="--", linewidth=0.8)
        ax1.set_ylabel("P(conscious)", fontsize=11)
        ax1.set_title(f"TopoConscious – {subject_id}", fontsize=13)
        ax1.set_ylim(0, 1)
        ax1.legend(loc="upper right")

        ax2 = fig.add_subplot(gs[1])
        ax2.plot(np.arange(len(wass_timeline)), wass_timeline,
                 color="#FF5722", linewidth=1.2, label="Wasserstein dist (H1)")
        ax2.set_xlabel("Window index", fontsize=11)
        ax2.set_ylabel("W₂ distance", fontsize=11)
        ax2.legend(loc="upper right")

        out_path = os.path.join(self.output_dir, subject_id,
                                "consciousness_timeline.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] Saved timeline → {out_path}")

    # ------------------------------------------------------------------
    def plot_te_matrix(self, te_matrix: np.ndarray, subject_id: str):
        fig, ax = plt.subplots(figsize=(9, 8))
        im = ax.imshow(te_matrix, aspect="auto", cmap="hot", origin="upper")
        plt.colorbar(im, ax=ax, label="Transfer Entropy (bits)")
        ax.set_title(f"Topological Transfer Entropy – {subject_id}", fontsize=13)
        ax.set_xlabel("Target region", fontsize=11)
        ax.set_ylabel("Source region", fontsize=11)

        out_path = os.path.join(self.output_dir, subject_id, "te_matrix.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [viz] Saved TE matrix → {out_path}")

    # ------------------------------------------------------------------
    def persistence_widget(self, diagrams_list: list, tr: float = 2.0):
        """
        Returns an ipywidgets interactive widget for scrubbing through
        persistence diagrams over time.
        Usage: display(visualizer.persistence_widget(diagrams_list))
        """
        import ipywidgets as widgets
        from IPython.display import display
        import io, base64

        def _render(window_idx):
            dgms = diagrams_list[window_idx]
            fig, axes = plt.subplots(1, 3, figsize=(13, 4))
            colors = ["#4CAF50", "#2196F3", "#9C27B0"]
            for d in range(min(3, len(dgms))):
                bars = dgms.get(d, np.empty((0, 2)))
                ax = axes[d]
                if len(bars) > 0:
                    ax.scatter(bars[:, 0], bars[:, 1],
                               color=colors[d], alpha=0.7, s=20)
                lim = max(1.0,
                          max(bars[:, 1].max() if len(bars) > 0 else 1.0
                              for bars in [dgms.get(dd, np.empty((0,2)))
                                           for dd in range(3)] if len(bars) > 0))
                ax.plot([0, lim], [0, lim], "k--", linewidth=0.8)
                ax.set_title(f"H{d} – Window {window_idx} "
                             f"(t={window_idx*5*tr:.0f}s)", fontsize=10)
                ax.set_xlabel("Birth")
                ax.set_ylabel("Death")
            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", dpi=100)
            buf.seek(0)
            plt.close(fig)
            img = widgets.Image(value=buf.read(), format="png",
                                width=780, height=260)
            return img

        slider = widgets.IntSlider(
            value=0, min=0, max=len(diagrams_list) - 1, step=1,
            description="Window:", continuous_update=False,
            layout=widgets.Layout(width="600px")
        )
        out = widgets.Output()

        def on_change(change):
            with out:
                out.clear_output(wait=True)
                display(_render(change["new"]))

        slider.observe(on_change, names="value")
        with out:
            display(_render(0))

        return widgets.VBox([slider, out])
