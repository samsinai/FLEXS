import flexs
import numpy as np


class NullModel(flexs.Model):
    def __init__(self, cache=True, seed=None):
        super().__init__(name="NullModel")

        self.average_fitness = 0.05
        self.rng = np.random.default_rng()
        self.cache = {}

    def train(self, sequences, labels):
        self.average_fitness = np.mean(labels)
        self.cache.update(zip(sequences, labels))

    def _fitness_function(self, sequences):
        sequences = np.array(sequences)
        fitnesses = np.empty(len(sequences))

        # We use cached evaluations so that the model gives deterministic outputs
        cached = np.array([seq in self.cache for seq in sequences])
        fitnesses[cached] = np.array([self.cache[seq] for seq in sequences[cached]])

        # Fitnesses are exponentially distributed with mean at the average fitness
        fitnesses[~cached] = self.rng.exponential(
            scale=self.average_fitness, size=np.count_nonzero(~cached)
        )

        # Update cache with new sequences and their predicted fitnesses
        self.cache.update(zip(sequences[~cached], fitnesses[~cached]))

        return fitnesses
