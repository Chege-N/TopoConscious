# 🧠 TopoConscious

> **Persistent Homology Pipeline for Neural Correlates of Consciousness**

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![GUDHI](https://img.shields.io/badge/topology-GUDHI%203.8%2B-purple)](https://gudhi.inria.fr)
[![Ripser](https://img.shields.io/badge/homology-Ripser%200.6%2B-orange)](https://ripser.scikit-tda.org)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20101572.svg)](https://doi.org/10.5281/zenodo.20101572)
[!PyPI](https://img.shields.io/pypi/v/topoconscious.svg)

TopoConscious is a Python package that mathematically distinguishes **conscious** from **unconscious** brain states using topological data analysis of fMRI time series. It models each sliding time window of fMRI activity as a point cloud in high-dimensional space, computes persistent homology (H₀, H₁, H₂), and tracks topological changes over time to decode consciousness probability.

---

## 🔬 The Problem

The neural correlates of consciousness (NCC) remain one of neuroscience's deepest unsolved problems. Existing approaches use static functional connectivity matrices or power spectra — they miss the **dynamic topological reorganisation** that may underlie conscious experience.

**The hypothesis:** Conscious states exhibit greater persistence of 1-dimensional cycles (loops) in the functional connectivity manifold — reflecting dynamic integration — while unconscious states are dominated by 0-dimensional clusters (segregation).

---

## 🏗️ Architecture

```
fMRI NIfTI (BIDS)
        │
        ▼
┌─────────────────┐
│  Preprocessing  │  Atlas parcellation (AAL-90 / Schaefer-100 / Destrieux)
│                 │  Band-pass 0.01–0.1 Hz, detrend, smooth 6mm FWHM
└────────┬────────┘
         │  (n_volumes × n_regions) time-series matrix
         ▼
┌─────────────────┐
│ Sliding Windows │  window=30 TRs, step=5 TRs → ~54 windows per scan
└────────┬────────┘
         │  list of (30 × n_regions) point clouds
         ▼
┌─────────────────┐
│ MaxMin Sampling │  Farthest-point landmark selection → 200 pts
│                 │  Reduces O(n²) to O(k²), keeps topology intact
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Persistent      │  Vietoris-Rips filtration via GUDHI + Ripser
│ Homology Engine │  H₀ (clusters), H₁ (loops), H₂ (voids)
│                 │  PCA pre-projection for 90-dim → 30-dim approximation
└────────┬────────┘
         │  persistence diagrams {dim: [(birth, death), ...]}
         ▼
┌─────────────────────────────────────────────────────┐
│              Topological Metrics                     │
│                                                     │
│  Wasserstein W₂   — standard diagram distance       │
│  Müller-Lyer      — scale + location aware metric   │  ← Novel
│  Current          — persistence-weighted OT         │
│  Persistence      — functional L² landscape         │
│  Landscape        — k-th largest tent functions     │
└────────┬────────────────────────────────────────────┘
         │  distance timelines
         ▼
┌─────────────────┐
│ Signature       │  7-feature vector per window:
│ Vectors         │  total_pers H0/H1/H2, max_pers H1,
│                 │  n_bars H1, birth/death ratio, entropy H1
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Gaussian HMM    │  2-state (conscious / unconscious)
│                 │  Viterbi decoding + posterior P(conscious)
│                 │  Covariance fallback: full → diag → spherical
└────────┬────────┘
         │  P(conscious) time course
         ▼
┌─────────────────┐
│ Topological     │  TE(X→Y) = H(Yₜ|Yₜ₋₁) - H(Yₜ|Yₜ₋₁,Xₜ₋₁)
│ Transfer        │  C++/OpenMP extension for O(n²) region pairs
│ Entropy         │  Measures directional information flow
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Cycle           │  GUDHI SimplexTree cocycle representatives
│ Localization    │  Maps H₁ generators → anatomical regions
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Output          │  P(conscious).csv, wasserstein_timeline.npy,
│                 │  ml_timeline.npy, pl_timeline.npy, te_matrix.npy
│                 │  consciousness_timeline.png, te_matrix.png
└─────────────────┘
```

---

## 📦 Installation

### Requirements
- Python 3.9+
- C++ compiler with OpenMP support (GCC 9+ recommended)
- Optional: CUDA 12 + cupy for GPU acceleration

```bash
# Clone the repository
git clone https://github.com/Chege-N/TopoConscious.git
cd TopoConscious

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate          # Linux/macOS
# venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt

# Install TopoConscious (builds the C++ extension)
pip install -e .
```

### GPU Support (Optional)
```bash
# Uncomment in requirements.txt:
# cupy-cuda12x>=12.0
pip install cupy-cuda12x
```

---

## 🚀 Quick Start

### Python API

```python
from topoconscious import TopoConsciousPipeline

pipe = TopoConsciousPipeline(
    bids_dir="data/ds002345",
    output_dir="results/",
    window_size=30,       # TRs per window
    step=5,               # TR step between windows
    n_landmarks=200,      # MaxMin landmark count
    max_homology_dim=2,   # Compute H0, H1, H2
    tr=2.0,               # Repetition time (seconds)
    atlas="aal",          # aal | schaefer100 | destrieux
    use_gpu=False,
)

results = pipe.run()
# results["sub-01"]["hmm"]["p_conscious"]  → array of shape (n_windows,)
```

### Command Line Interface

```bash
topoconscious \
  --bids-dir data/ds002345 \
  --output-dir results/ \
  --window-size 30 \
  --step 5 \
  --landmarks 200 \
  --atlas aal \
  --max-dim 2 \
  --tr 2.0 \
  --gpu                  # optional
```

### FastAPI Backend

```bash
# Start the REST server
uvicorn backend.app:app --reload --port 8000

# Trigger pipeline run
curl -X POST http://localhost:8000/run
```

---

## 🔭 Key Innovations

### 1. Müller-Lyer Current Metric
Standard Wasserstein distance is permutation-invariant but loses information about *where* features live in the birth-death plane. The Müller-Lyer current adds:

```
ML(D₁, D₂) = W₂ᵖᵉʳˢ⁻ʷᵉⁱᵍʰᵗᵉᵈ(D₁, D₂)
            + α · |Σpers(D₁) − Σpers(D₂)|    ← scale penalty
            + β · ‖centroid(D₁) − centroid(D₂)‖  ← location penalty
```

This makes transitions between *many short loops* (unconscious) and *few long loops* (conscious) maximally discriminative.

### 2. Persistence Landscape (Landscape Current)
The k-th landscape function λₖ(t) is the k-th largest tent value at parameter t across all H₁ bars. Lives in L² — supports averaging, statistical testing, and inner products:

```python
from topoconscious.metrics import PersistenceLandscape
pl = PersistenceLandscape(n_landscapes=5, resolution=100)
vec = pl.vectorize(diagram)       # (500,) feature vector
dist = pl.distance(dgm1, dgm2)    # L² landscape distance
```

### 3. Topological Transfer Entropy (C++/OpenMP)
Directional information flow between brain regions' H₁ persistence time series, parallelised across all O(n²) region pairs:

```
TE(X→Y) = H(Yₜ | Yₜ₋₁) − H(Yₜ | Yₜ₋₁, Xₜ₋₁)
```

The C++ extension computes the full 90×90 matrix in milliseconds vs. minutes in Python.

### 4. Full Cycle Localization
Using GUDHI `SimplexTree` cocycle representatives, each significant H₁ generator is traced back to the specific brain regions forming its boundary — enabling anatomical interpretation of which regions are responsible for each consciousness transition.

---

## 📊 Validation

Three public datasets target the success criterion of **AUC > 0.90** (vs. static FC baseline ~0.75):

| Dataset | Condition | Source |
|---------|-----------|--------|
| Propofol | Awake vs. anaesthesia | MIT/Tufts (OpenNeuro ds002898) |
| Sleep | REM vs. NREM | OpenNeuro ds000201 |
| Disorders of Consciousness | MCS vs. UWS | Liège dataset |

```python
from topoconscious.validation import ValidationRunner

runner = ValidationRunner(output_dir="results/validation")
results = runner.evaluate_dataset(ts_list, labels, "propofol")
# results["auc_topo"]  → e.g. 0.94
# results["auc_fc"]    → e.g. 0.76
runner.plot_roc_curves(results)
```

---

## 📓 Notebooks

| Notebook | Description |
|----------|-------------|
| `01_demo_pipeline.ipynb` | End-to-end run on synthetic 300-volume fMRI data |
| `02_widget_explorer.ipynb` | Interactive ipywidgets scrubber for persistence diagrams |
| `03_validation_roc.ipynb` | ROC/AUC evaluation with synthetic labelled datasets |
| `04_real_bids_data.ipynb` | Full walkthrough: DataLad download → BIDS → results |

---

## 🧪 Tests

```bash
# Run full test suite
pytest tests/ -v

# Run specific module tests
pytest tests/test_topology.py -v
pytest tests/test_metrics.py -v        # includes ValidationRunner smoke test
pytest tests/test_transfer_entropy.py -v
```

Test coverage:
- `test_topology.py` — landmark sampling, diagram shapes, Wasserstein timeline
- `test_hmm.py` — fit/decode shapes, p_conscious bounds, score() after fit
- `test_transfer_entropy.py` — matrix shape, diagonal=0, non-negativity
- `test_metrics.py` — ML current correctness, self-distance=0, ValidationRunner

---

## 📁 Project Structure

```
TopoConscious/
├── topoconscious/
│   ├── __init__.py              # Package exports
│   ├── pipeline.py              # Main orchestrator
│   ├── preprocessing.py         # NIfTI/BIDS loader, atlas parcellation
│   ├── topology.py              # PersistenceEngine (GUDHI + Ripser)
│   ├── hmm.py                   # TopologicalHMM (Gaussian, 2-state)
│   ├── transfer_entropy.py      # TopologicalTransferEntropy
│   ├── localization.py          # CycleLocalizer (GUDHI SimplexTree)
│   ├── visualization.py         # TopoVisualizer (matplotlib + ipywidgets)
│   ├── metrics.py               # MuellerLyerCurrent + PersistenceLandscape
│   ├── validation.py            # ValidationRunner (ROC/AUC)
│   ├── run_pipeline.py          # Convenience entry script
│   └── ext/
│       └── topo_te.cpp          # C++17/OpenMP transfer entropy extension
├── backend/
│   └── app.py                   # FastAPI REST endpoint (POST /run)
├── notebooks/
│   ├── 01_demo_pipeline.ipynb
│   ├── 02_widget_explorer.ipynb
│   ├── 03_validation_roc.ipynb
│   └── 04_real_bids_data.ipynb
├── tests/
│   ├── conftest.py              # Shared session-scoped fixtures
│   ├── test_topology.py
│   ├── test_hmm.py
│   ├── test_transfer_entropy.py
│   └── test_metrics.py
├── data/                        # Place BIDS datasets here
├── results/                     # Pipeline outputs written here
├── CITATION.cff                 # Academic citation metadata
├── LICENSE                      # Apache 2.0
├── requirements.txt
└── setup.py
```

---

## ⚙️ Configuration Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `bids_dir` | — | Path to BIDS dataset root |
| `output_dir` | — | Where results are written |
| `window_size` | `30` | TRs per sliding window |
| `step` | `5` | TR step between windows |
| `n_landmarks` | `200` | MaxMin landmark count |
| `max_homology_dim` | `2` | Max homology dim (0=H₀, 1=H₁, 2=H₂) |
| `tr` | `2.0` | Repetition time in seconds |
| `atlas` | `aal` | `aal` / `schaefer100` / `destrieux` |
| `use_gpu` | `False` | Use cupy-accelerated Ripser |

---

## 📖 References

1. Edelsbrunner, H. & Harer, J. (2010). *Computational Topology: An Introduction.* AMS.
2. Bauer, U. (2021). Ripser: efficient computation of Vietoris-Rips persistence barcodes. *J. Applied & Computational Topology.*
3. Divol, V. & Lacombe, T. (2021). Understanding topology and geometry of persistence diagrams via optimal partial transport. *J. Applied & Computational Topology.*
4. Bubenik, P. (2015). Statistical topological data analysis using persistence landscapes. *JMLR 16.*
5. GUDHI Project. *Geometry Understanding in Higher Dimensions.* https://gudhi.inria.fr
6. Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters 85(2).*

---

## 📄 Citation

If you use TopoConscious in your research, please cite:

```bibtex
@software{topoconscious2026,
  title   = {TopoConscious: Persistent Homology Pipeline for Neural Correlates of Consciousness},
  author  = {Chege, N.},
  year    = {2026},
  version = {0.1.0},
  license = {Apache-2.0},
  url     = {https://github.com/Chege-N/TopoConscious}
}
```

---

## 🤝 Contributing

Pull requests are welcome. For major changes please open an issue first.

```bash
# Run tests before submitting
pytest tests/ -v
# Check code style
flake8 topoconscious/ --max-line-length=100
```

---

## 📜 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.
