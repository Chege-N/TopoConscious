"""TopoConscious: Persistent Homology Pipeline for Consciousness Detection."""
from .pipeline import TopoConsciousPipeline
from .topology import PersistenceEngine
from .hmm import TopologicalHMM
from .transfer_entropy import TopologicalTransferEntropy
from .metrics import MuellerLyerCurrent, PersistenceLandscape
from .validation import ValidationRunner
from .localization import CycleLocalizer

__version__ = "0.1.0"
__all__ = [
    "TopoConsciousPipeline",
    "PersistenceEngine",
    "TopologicalHMM",
    "TopologicalTransferEntropy",
    "MuellerLyerCurrent",
    "PersistenceLandscape",
    "ValidationRunner",
    "CycleLocalizer",
]
