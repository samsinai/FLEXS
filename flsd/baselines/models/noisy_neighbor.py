import editdistance
import numpy as np
import random
from sklearn.metrics import explained_variance_score, r2_score
from scipy.stats import pearsonr

import flsd


class NoisyNeighbor(flsd.Model):
    """
    Behaves like a ground truth model.

    It corrupts a ground truth model with noise, which is modulated by distance
    to already measured sequences.
    """

    def __init__(
        self,
        landscape,
        signal_strength=0.9,
        cache=True,
        landscape_id=-1,
        start_id=-1,
    ):
        self.landscape = landscape
        self.measured_sequences = {}  # save the measured sequences for the model
        self.model_sequences = {}  # cache the sequences for later queries
        self.cost = 0
        self.evals = 0
        self.ss = signal_strength
        self.cache = cache
        self.model_type = f"NAMb_ss{self.ss}"
        self.landscape_id = landscape_id
        self.start_id = start_id
        self.fitnesses = []
        self.r2 = signal_strength ** 2  # this is a proxy

    def _get_min_distance(self, sequence):
        new_dist = np.inf
        closest = None

        for seq in self.measured_sequences:
            dist = editdistance.eval(sequence, seq)

            if dist == 1:
                return dist, seq

            if dist < new_dist:
                new_dist = dist
                closest = seq

        return new_dist, closest

    def train(self, sequences, labels):
        self.measured_sequences.update(zip(sequences, labels))

    def get_fitness(self, sequences):
        fitnesses = []

        for seq in sequences:

            # If we have already measured the sequence, just return cached value
            if seq in self.measured_sequences:
                fitnesses.append(self.measured_sequences[seq])
                continue

            # Otherwise, fitness = alpha * true_fitness + (1 - alpha) * noise
            # where alpha = signal_strength ^ (dist to nearest neighbor)
            # and noise is the nearest neighbor's fitness plus exponentially distributed noise
            distance, neighbor_seq = self._get_min_distance(seq)

            signal = self.landscape.get_fitness([seq]).item()
            neighbor_fitness = self.landscape.get_fitness([neighbor_seq]).item()
            noise = np.random.exponential(scale=neighbor_fitness)

            alpha = self.ss ** distance
            fitnesses.append(alpha * signal + (1 - alpha) * noise)

        return np.array(fitnesses)
