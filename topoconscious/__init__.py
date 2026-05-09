"""TopoConscious: Persistent Homology Pipeline for Consciousness Detection."""
from .pipeline import TopoConsciousPipeline
from .topology import PersistenceEngine
from .hmm import TopologicalHMM
from .transfer_entropy import TopologicalTransferEntropy

__version__ = "0.1.0"
__all__ = [
    "TopoConsciousPipeline",
    "PersistenceEngine",
    "TopologicalHMM",
    "TopologicalTransferEntropy",
]
