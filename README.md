# TopoConscious

A Python pipeline for detecting neural correlates of consciousness via persistent homology of fMRI/MEG time series.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

```python
from topoconscious import TopoConsciousPipeline

pipe = TopoConsciousPipeline(
    bids_dir="data/ds002345",
    output_dir="results/",
    window_size=30,
    step=5,
    n_landmarks=200,
    max_homology_dim=2
)
pipe.run()
```

## Validation Datasets

1. **Propofol study** – MIT/Tufts awake vs. anaesthesia
2. **Sleep staging** – REM vs. NREM (OpenNeuro ds000201)
3. **Disorders of Consciousness** – MCS vs. UWS (Liège dataset)

## Project Structure

```
topoconscious/
├── topoconscious/
│   ├── __init__.py
│   ├── pipeline.py
│   ├── preprocessing.py
│   ├── topology.py
│   ├── hmm.py
│   ├── transfer_entropy.py
│   ├── localization.py
│   ├── visualization.py
│   ├── metrics.py           ← Müller-Lyer current + PersistenceLandscape
│   ├── validation.py        ← ROC/AUC evaluation vs static FC
│   └── ext/
│       └── topo_te.cpp
├── notebooks/
│   ├── 01_demo_pipeline.ipynb
│   ├── 02_widget_explorer.ipynb
│   ├── 03_validation_roc.ipynb
│   └── 04_real_bids_data.ipynb  ← full real-data walkthrough
├── tests/
│   ├── test_topology.py
│   ├── test_hmm.py
│   ├── test_transfer_entropy.py
│   ├── test_metrics.py          ← MuellerLyerCurrent + ValidationRunner
│   └── conftest.py              ← shared fixtures
├── data/
│   └── .gitkeep
├── results/
│   └── .gitkeep
├── setup.py
├── requirements.txt
└── README.md
```
