"""Define the noisy abstract model class."""
import editdistance
import numpy as np

import flexs
from flexs.types import SEQUENCES_TYPE


class NoisyAbstractModel(flexs.Model):
    r"""
    Behaves like a ground truth model.

    It corrupts a ground truth model with noise, which is modulated by distance
    to already measured sequences.

    Specifically, $\hat{f}(x) = \alpha^d f(x) + (1 - \alpha^d) \epsilon$ where
    $\epsilon$ is drawn from an exponential distribution with mean $f(x)$
    $d$ is the edit distance to the closest measured neighbor,
    and $\alpha$ is the signal strength.
    """

    def __init__(
        self,
        landscape: flexs.Landscape,
        signal_strength: float = 0.9,
    ):
        """
        Create a noisy abstract model.

        Args:
            landscape: The ground truth landscape.
            signal_strength: A value between 0 and 1 representing the
                true signal strength.

        """
        super().__init__(f"NAMb_ss{signal_strength}")

        self.landscape = landscape
        self.ss = signal_strength
        self.cache = {}

    def _get_min_distance(self, sequence):
        # Special case if cache is empty
        if len(self.cache) == 0:
            return 0, sequence

        new_dist = np.inf
        closest = None

        for seq in self.cache:
            dist = editdistance.eval(sequence, seq)

            if dist == 1:
                return dist, seq

            if dist < new_dist:
                new_dist = dist
                closest = seq

        return new_dist, closest

    def train(self, sequences: SEQUENCES_TYPE, labels: np.ndarray):
        """
        Training step simply stores sequences and labels in a
        dictionary for future lookup.
        """
        self.cache.update(zip(sequences, labels))

    def _fitness_function(self, sequences):
        sequences = np.array(sequences)
        fitnesses = np.empty(len(sequences))

        # We use cached evaluations so that the model gives deterministic outputs
        cached = np.array([seq in self.cache for seq in sequences])
        fitnesses[cached] = np.array([self.cache[seq] for seq in sequences[cached]])

        new_fitnesses = []
        for seq in sequences[~cached]:

            # Otherwise, fitness = alpha * true_fitness + (1 - alpha) * noise
            # where alpha = signal_strength ^ (dist to nearest neighbor)
            # and noise is the nearest neighbor's fitness plus exponentially
            # distributed noise
            distance, neighbor_seq = self._get_min_distance(seq)

            signal = self.landscape.get_fitness([seq]).item()
            neighbor_fitness = self.landscape.get_fitness([neighbor_seq]).item()
            if neighbor_fitness >= 0:
                noise = np.random.exponential(scale=neighbor_fitness)
            else:
                noise = np.random.choice(list(self.cache.values()))

            alpha = self.ss ** distance
            new_fitnesses.append(alpha * signal + (1 - alpha) * noise)

        fitnesses[~cached] = new_fitnesses

        # Update cache with new sequences and their predicted fitnesses
        self.cache.update(zip(sequences[~cached], fitnesses[~cached]))

        return np.array(fitnesses)
