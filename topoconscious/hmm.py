"""
Hidden Markov Model for consciousness state decoding.
States: 0 = unconscious, 1 = conscious
Emissions: topological signature vectors (Gaussian mixture).
"""
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from typing import Dict


class TopologicalHMM:
    """
    Gaussian HMM with 2 states fit on topological signature vectors.
    Provides posterior probability of the 'conscious' state per window.
    """

    def __init__(self, n_states: int = 2, n_iter: int = 200,
                 covariance_type: str = "full", random_state: int = 42):
        self.n_states = n_states
        self.n_iter = n_iter
        self.cov_type = covariance_type
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self._conscious_state = 1   # resolved after fitting

    def fit_decode(self, sig_vectors: np.ndarray) -> Dict:
        """
        Fit HMM on signature vectors and decode the most likely state sequence.
        Returns dict with keys: state_sequence, p_conscious, log_likelihood.
        """
        X = self.scaler.fit_transform(sig_vectors)
        lengths = [len(X)]

        self.model = hmm.GaussianHMM(
            n_components=self.n_states,
            covariance_type=self.cov_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
        )
        self.model.fit(X, lengths)

        log_likelihood, state_seq = self.model.decode(X, algorithm="viterbi")
        posteriors = self.model.predict_proba(X)

        # Heuristic: the conscious state has higher total H1 persistence on average
        # (index 1 of signature = total_persistence_H1)
        means = self.model.means_
        self._conscious_state = int(np.argmax(means[:, 1]))

        p_conscious = posteriors[:, self._conscious_state]

        return {
            "state_sequence": state_seq,
            "p_conscious": p_conscious,
            "posteriors": posteriors,
            "log_likelihood": log_likelihood,
            "conscious_state_id": self._conscious_state,
        }

    def score(self, sig_vectors: np.ndarray) -> np.ndarray:
        """Return p_conscious for new data (requires prior fit)."""
        X = self.scaler.transform(sig_vectors)
        posteriors = self.model.predict_proba(X)
        return posteriors[:, self._conscious_state]
