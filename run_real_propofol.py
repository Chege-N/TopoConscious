"""
run_real_propofol.py
--------------------
Runs TopoConscious on the real propofol BIDS dataset and produces:
  - P(conscious) time course per subject
  - ROC/AUC validation against known awake/anaesthesia labels
  - Wasserstein, Müller-Lyer, and Persistence Landscape timelines
  - Transfer entropy matrices
  - Publication-quality figures saved to results_real/

Dataset expected at: validation_data_propofol/
BIDS structure:  sub-XX/func/sub-XX_task-rest_bold.nii.gz

Usage:
    python run_real_propofol.py
    python run_real_propofol.py --subjects sub-01 sub-02
    python run_real_propofol.py --output results_real/
"""

import argparse
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import roc_curve, auc

# ── Import TopoConscious modules ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from topoconscious.topology import PersistenceEngine
from topoconscious.hmm import TopologicalHMM
from topoconscious.transfer_entropy import TopologicalTransferEntropy
from topoconscious.metrics import MuellerLyerCurrent, PersistenceLandscape
from topoconscious.localization import CycleLocalizer
from topoconscious.validation import ValidationRunner

# ── Configuration ─────────────────────────────────────────────────────────────
BIDS_DIR     = Path("validation_data_propofol")
OUTPUT_DIR   = Path("results_real")
WINDOW_SIZE  = 30      # TRs
STEP         = 5       # TRs
N_LANDMARKS  = 100     # reduce for speed on small datasets
MAX_DIM      = 1       # H0 + H1 (H2 needs more data)
TR           = 2.0     # seconds


def load_subject_timeseries(subject_dir: Path) -> np.ndarray:
    """
    Load and parcellate BOLD data for one subject.
    Uses nilearn NiftiLabelsMasker with AAL-90 atlas.
    Falls back to raw voxel PCA if atlas parcellation fails.
    """
    from nilearn import datasets
    from nilearn.maskers import NiftiLabelsMasker

    # Find the BOLD file
    func_dir = subject_dir / "func"
    bold_files = list(func_dir.glob("*bold.nii*"))
    if not bold_files:
        raise FileNotFoundError(f"No BOLD file in {func_dir}")
    bold_path = bold_files[0]
    print(f"  Loading: {bold_path.name}")

    try:
        atlas = datasets.fetch_atlas_aal()
        masker = NiftiLabelsMasker(
            labels_img=atlas.maps,
            standardize=True,
            detrend=True,
            low_pass=0.1,
            high_pass=0.01,
            t_r=TR,
            smoothing_fwhm=6.0,
            verbose=0,
        )
        ts = masker.fit_transform(str(bold_path))
        print(f"  Time series shape: {ts.shape} (volumes x regions)")
        return ts
    except Exception as e:
        print(f"  Atlas parcellation failed ({e}), using PCA fallback")
        import nibabel as nib
        from sklearn.decomposition import PCA
        img = nib.load(str(bold_path))
        data = img.get_fdata()
        flat = data.reshape(-1, data.shape[-1]).T  # (T, voxels)
        # Remove zero-variance voxels
        flat = flat[:, flat.std(axis=0) > 0]
        pca = PCA(n_components=min(90, flat.shape[1], flat.shape[0]-1))
        ts = pca.fit_transform(flat)
        print(f"  PCA fallback shape: {ts.shape}")
        return ts


def run_subject(subject_dir: Path, output_dir: Path) -> dict:
    """Run the full pipeline on one subject."""
    subject_id = subject_dir.name
    print(f"\n{'='*60}")
    print(f"Subject: {subject_id}")
    print(f"{'='*60}")

    sub_out = output_dir / subject_id
    sub_out.mkdir(parents=True, exist_ok=True)

    # 1. Load time series
    ts = load_subject_timeseries(subject_dir)
    n_vols, n_regions = ts.shape
    print(f"  {n_vols} volumes, {n_regions} regions")

    if n_vols < WINDOW_SIZE + STEP:
        print(f"  WARNING: Too few volumes ({n_vols}), skipping")
        return None

    # 2. Sliding windows
    windows = [ts[t:t+WINDOW_SIZE] for t in range(0, n_vols - WINDOW_SIZE + 1, STEP)]
    print(f"  {len(windows)} sliding windows (size={WINDOW_SIZE}, step={STEP})")

    # 3. Persistent homology
    engine = PersistenceEngine(
        max_dim=MAX_DIM,
        n_landmarks=min(N_LANDMARKS, WINDOW_SIZE),
        use_gpu=False,
    )
    print(f"  Computing persistence diagrams...")
    diagrams = [engine.compute(w) for w in windows]
    print(f"  Done. {len(diagrams)} diagrams computed.")

    # 4. Distance timelines
    print(f"  Computing distance timelines...")
    wass_tl = engine.wasserstein_timeline(diagrams)

    ml = MuellerLyerCurrent(alpha=0.1, beta=0.05)
    ml_tl = ml.timeline(diagrams, dim=1)

    pl = PersistenceLandscape(n_landscapes=5, resolution=100)
    pl_tl = pl.timeline(diagrams, dim=1)

    # 5. Signature vectors + HMM
    print(f"  HMM decoding...")
    sig = engine.signature_vectors(diagrams)
    hmm_model = TopologicalHMM(n_states=2, n_iter=200)
    hmm_result = hmm_model.fit_decode(sig)
    p_conscious = hmm_result["p_conscious"]
    print(f"  Mean P(conscious): {p_conscious.mean():.3f}")

    # 6. Transfer entropy (on reduced regions for speed)
    print(f"  Computing transfer entropy...")
    te_calc = TopologicalTransferEntropy(lag=1, n_bins=8)
    te_matrix = te_calc.compute(diagrams, ts)

    # 7. Cycle localization at transitions
    transitions = [i for i in range(1, len(hmm_result["state_sequence"]))
                   if hmm_result["state_sequence"][i] != hmm_result["state_sequence"][i-1]]
    print(f"  Detected {len(transitions)} state transitions")

    localizer = CycleLocalizer(atlas="aal")
    localization = {}
    for t in transitions[:5]:  # limit to first 5 for speed
        if t < len(windows):
            localization[t] = localizer.localize_with_complex(
                point_cloud=windows[t],
                max_edge_length=4.0,
            )

    # 8. Save numerical results
    np.save(sub_out / "wasserstein_timeline.npy", wass_tl)
    np.save(sub_out / "ml_timeline.npy", ml_tl)
    np.save(sub_out / "pl_timeline.npy", pl_tl)
    np.save(sub_out / "te_matrix.npy", te_matrix)
    pd.DataFrame({
        "window": range(len(p_conscious)),
        "time_s": [w * STEP * TR for w in range(len(p_conscious))],
        "p_conscious": p_conscious,
        "state": hmm_result["state_sequence"],
    }).to_csv(sub_out / "consciousness_probability.csv", index=False)

    # 9. Publication-quality figures
    _plot_timelines(subject_id, p_conscious, wass_tl, ml_tl, pl_tl,
                    hmm_result["state_sequence"], sub_out)
    _plot_te_matrix(subject_id, te_matrix, n_regions, sub_out)

    print(f"  Results saved to {sub_out}")
    return {
        "subject": subject_id,
        "n_windows": len(windows),
        "n_volumes": n_vols,
        "n_regions": n_regions,
        "p_conscious": p_conscious,
        "mean_p_conscious": float(p_conscious.mean()),
        "n_transitions": len(transitions),
        "diagrams": diagrams,
        "hmm": hmm_result,
        "te_matrix": te_matrix,
    }


def _plot_timelines(subject_id, p_conscious, wass_tl, ml_tl, pl_tl,
                    states, out_dir):
    """4-panel timeline figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(4, 1, hspace=0.5)
    t_windows = np.arange(len(p_conscious))

    # Panel 1: P(conscious)
    ax1 = fig.add_subplot(gs[0])
    ax1.fill_between(t_windows, 0, p_conscious, alpha=0.75,
                     color="#2196F3", label="P(conscious)")
    ax1.axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.7)
    # Shade state regions
    for i, s in enumerate(states):
        ax1.axvspan(i-0.5, i+0.5, alpha=0.05,
                    color="#4CAF50" if s == 1 else "#F44336")
    ax1.set_ylabel("P(conscious)", fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_title(f"TopoConscious — {subject_id} (Real Data)", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)

    # Panel 2: Wasserstein
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(np.arange(len(wass_tl)), wass_tl,
             color="#FF5722", linewidth=1.2, label="Wasserstein W₂ (H₁)")
    ax2.set_ylabel("W₂", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9)

    # Panel 3: Müller-Lyer current
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(np.arange(len(ml_tl)), ml_tl,
             color="#9C27B0", linewidth=1.2, label="Müller-Lyer current (H₁)")
    ax3.set_ylabel("ML dist", fontsize=10)
    ax3.legend(loc="upper right", fontsize=9)

    # Panel 4: Persistence landscape
    ax4 = fig.add_subplot(gs[3])
    ax4.plot(np.arange(len(pl_tl)), pl_tl,
             color="#00BCD4", linewidth=1.2, label="Persistence landscape L² (H₁)")
    ax4.set_xlabel("Window index", fontsize=10)
    ax4.set_ylabel("PL dist", fontsize=10)
    ax4.legend(loc="upper right", fontsize=9)

    plt.savefig(out_dir / "consciousness_timeline.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved consciousness_timeline.png")


def _plot_te_matrix(subject_id, te_matrix, n_regions, out_dir):
    """Transfer entropy heatmap."""
    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(te_matrix, aspect="auto", cmap="hot", origin="upper")
    plt.colorbar(im, ax=ax, label="Transfer Entropy (bits)")
    ax.set_title(f"Topological Transfer Entropy — {subject_id} (Real Data)",
                 fontsize=12)
    ax.set_xlabel("Target region")
    ax.set_ylabel("Source region")
    plt.savefig(out_dir / "te_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def validate_against_labels(results: list, output_dir: Path):
    """
    Propofol study label assignment:
      - First half of scan = awake (label=1)
      - Second half of scan = propofol-induced unconscious (label=0)
    This matches the ds002898 protocol where subjects transition
    from awake to anaesthesia within a single scan.
    """
    print(f"\n{'='*60}")
    print("VALIDATION: ROC/AUC against propofol labels")
    print(f"{'='*60}")

    ts_list = []
    labels = []

    for res in results:
        if res is None:
            continue
        p = res["p_conscious"]
        n = len(p)
        # Split: first half = awake windows, second half = anaesthesia windows
        half = n // 2
        # Awake epoch: mean p_conscious (should be high)
        ts_list.append(p[:half])
        labels.append(1)  # awake = conscious
        # Anaesthesia epoch: mean p_conscious (should be low)
        ts_list.append(p[half:])
        labels.append(0)  # anaesthesia = unconscious

    if len(labels) < 4:
        print("  Not enough subjects for ROC analysis (need ≥2 subjects)")
        return

    topo_scores = np.array([np.mean(t) for t in ts_list])
    labels = np.array(labels)

    fpr, tpr, _ = roc_curve(labels, topo_scores)
    roc_auc = auc(fpr, tpr)
    print(f"  AUC (TopoConscious, real data): {roc_auc:.3f}")

    # Plot ROC
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#2196F3", lw=2,
            label=f"TopoConscious (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.4)
    ax.axhline(0.90, color="#4CAF50", linestyle="--", linewidth=0.8,
               label="Target AUC = 0.90")
    ax.fill_between(fpr, tpr, alpha=0.12, color="#2196F3")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Real Propofol Data")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.savefig(output_dir / "roc_curve_real.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved roc_curve_real.png")

    # Save summary
    pd.DataFrame([{
        "dataset": "propofol_real",
        "n_subjects": len(results),
        "n_epochs": len(labels),
        "auc_topoconscious": round(roc_auc, 4),
        "criterion_090": roc_auc >= 0.90,
    }]).to_csv(output_dir / "validation_summary_real.csv", index=False)

    return roc_auc


def main():
    parser = argparse.ArgumentParser(
        description="Run TopoConscious on real propofol BIDS data"
    )
    parser.add_argument("--bids-dir", default=str(BIDS_DIR))
    parser.add_argument("--output", default=str(OUTPUT_DIR))
    parser.add_argument("--subjects", nargs="+", default=None,
                        help="Specific subject IDs (default: all)")
    parser.add_argument("--max-dim", type=int, default=MAX_DIM)
    parser.add_argument("--landmarks", type=int, default=N_LANDMARKS)
    args = parser.parse_args()

    bids_dir = Path(args.bids_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover subjects
    if args.subjects:
        subject_dirs = [bids_dir / s for s in args.subjects]
    else:
        subject_dirs = sorted([
            d for d in bids_dir.iterdir()
            if d.is_dir() and d.name.startswith("sub-")
        ])

    if not subject_dirs:
        print(f"ERROR: No subjects found in {bids_dir}")
        print(f"Expected structure: {bids_dir}/sub-XX/func/sub-XX_*_bold.nii.gz")
        sys.exit(1)

    print(f"Found {len(subject_dirs)} subject(s): {[d.name for d in subject_dirs]}")

    # Run pipeline on each subject
    results = []
    for sub_dir in subject_dirs:
        try:
            result = run_subject(sub_dir, output_dir)
            results.append(result)
        except Exception as e:
            print(f"  ERROR on {sub_dir.name}: {e}")
            import traceback
            traceback.print_exc()

    # Group-level validation
    valid_results = [r for r in results if r is not None]
    if valid_results:
        roc_auc = validate_against_labels(valid_results, output_dir)

        # Group summary
        summary = pd.DataFrame([{
            "subject": r["subject"],
            "n_volumes": r["n_volumes"],
            "n_regions": r["n_regions"],
            "n_windows": r["n_windows"],
            "mean_p_conscious": round(r["mean_p_conscious"], 4),
            "n_transitions": r["n_transitions"],
        } for r in valid_results])
        summary.to_csv(output_dir / "group_summary_real.csv", index=False)
        print(f"\nGroup summary saved to {output_dir}/group_summary_real.csv")
        print(summary.to_string(index=False))

    print(f"\n{'='*60}")
    print(f"Done. All outputs in: {output_dir}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
